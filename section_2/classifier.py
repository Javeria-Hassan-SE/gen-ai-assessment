"""Task 2.2 — Two-agent sentence classification pipeline (Anthropic or OpenAI).

Classifier Agent batches all sentences. Embedded Sentence Agent runs only for
Incomplete sentences. Routing is enforced in Python, not prompts.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import anthropic

try:
    import openai as _openai
    _OPENAI_AVAILABLE = True
except ImportError:
    _openai = None  # type: ignore
    _OPENAI_AVAILABLE = False

from prompts import CLASSIFIER_SYSTEM_PROMPT, EMBEDDED_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
MAX_BATCH_SIZE = 25
MAX_RETRIES = 4
BASE_BACKOFF = 1.0
MAX_EMBEDDED_WORKERS = 8

VALID_CLASSIFICATIONS = frozenset(
    ["Simple", "Compound", "Complex", "Compound-Complex", "Incomplete"]
)
COMPLEXITY_RANK = {
    "Compound-Complex": 4,
    "Complex": 3,
    "Compound": 2,
    "Simple": 1,
    "Incomplete": 0,
}

_CLASSIFIER_TOOL = {
    "name": "classify_sentences",
    "description": (
        "Return the syntactic classification for each sentence. "
        "Return one entry per sentence, in the same order as received."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {
                            "type": "integer",
                            "description": "0-based position in the input list",
                        },
                        "classification": {
                            "type": "string",
                            "enum": list(VALID_CLASSIFICATIONS),
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "One sentence explaining the classification",
                        },
                    },
                    "required": ["index", "classification"],
                },
            }
        },
        "required": ["classifications"],
    },
}

_EMBEDDED_TOOL = {
    "name": "report_embedded_sentence",
    "description": (
        "Report whether an embedded complete clause was found in the fragment."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "found": {
                "type": "boolean",
                "description": "True if a recoverable subject+predicate was found",
            },
            "embedded_sentence": {
                "type": "string",
                "description": "Verbatim extract of the embedded clause (only when found=true)",
            },
            "classification": {
                "type": "string",
                "enum": ["Simple", "Compound", "Complex", "Compound-Complex"],
                "description": "Classification of the embedded sentence (only when found=true)",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation",
            },
        },
        "required": ["found"],
    },
}

_OPENAI_CLASSIFIER_TOOL = {
    "type": "function",
    "function": {
        "name": _CLASSIFIER_TOOL["name"],
        "description": _CLASSIFIER_TOOL["description"],
        "parameters": _CLASSIFIER_TOOL["input_schema"],
    },
}

_OPENAI_EMBEDDED_TOOL = {
    "type": "function",
    "function": {
        "name": _EMBEDDED_TOOL["name"],
        "description": _EMBEDDED_TOOL["description"],
        "parameters": _EMBEDDED_TOOL["input_schema"],
    },
}


class ClassificationResult:
    """Per-sentence output from the pipeline."""

    __slots__ = (
        "sentence",
        "classification",
        "agent_path",
        "embedded_sentence",
        "original_flag",
        "reasoning",
    )

    def __init__(
        self,
        sentence: str,
        classification: str,
        agent_path: list[str],
        embedded_sentence: Optional[str] = None,
        original_flag: Optional[str] = None,
        reasoning: str = "",
    ):
        self.sentence = sentence
        self.classification = classification
        self.agent_path = agent_path
        self.embedded_sentence = embedded_sentence
        self.original_flag = original_flag
        self.reasoning = reasoning

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "classification": self.classification,
            "agent_path": self.agent_path,
            "embedded_sentence": self.embedded_sentence,
            "original_flag": self.original_flag,
            "reasoning": self.reasoning,
        }

    def __repr__(self) -> str:
        return (
            f"ClassificationResult(classification={self.classification!r}, "
            f"agent_path={self.agent_path}, sentence={self.sentence!r})"
        )


class SentenceClassificationPipeline:
    """Thread-safe two-agent sentence classification pipeline (Anthropic or OpenAI)."""

    def __init__(
        self,
        client=None,
        model: str = DEFAULT_MODEL,
        provider: str = PROVIDER_ANTHROPIC,
        api_key: Optional[str] = None,
    ):
        self._provider = provider
        self._model = model
        self._lock = threading.Lock()

        if client is not None:
            self._client = client
        elif provider == PROVIDER_OPENAI:
            if not _OPENAI_AVAILABLE:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                )
            self._client = _openai.OpenAI(api_key=api_key)
        else:
            self._client = (
                anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
            )

    def classify(self, sentences: list[str]) -> list[ClassificationResult]:
        """Classify a list of sentences.  Thread-safe."""
        if not sentences:
            return []

        raw: list[dict] = self._batch_classify(sentences)

        received_idx = {r["index"] for r in raw}
        missing = [i for i in range(len(sentences)) if i not in received_idx]
        if missing:
            logger.warning(
                "Classifier returned %d missing results — retrying individually",
                len(missing),
            )
            for idx in missing:
                recovered = self._batch_classify([sentences[idx]])
                if recovered:
                    recovered[0]["index"] = idx
                    raw.append(recovered[0])
                else:
                    raw.append(
                        {
                            "index": idx,
                            "classification": "Incomplete",
                            "reasoning": "missing from classifier response",
                        }
                    )

        raw.sort(key=lambda r: r["index"])

        incomplete_items = [r for r in raw if r["classification"] == "Incomplete"]
        embedded_results: dict[int, dict] = {}
        if incomplete_items:
            with ThreadPoolExecutor(
                max_workers=min(len(incomplete_items), MAX_EMBEDDED_WORKERS)
            ) as pool:
                future_to_idx = {
                    pool.submit(self._run_embedded_agent, sentences[r["index"]]): r["index"]
                    for r in incomplete_items
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        embedded_results[idx] = future.result()
                    except Exception as exc:
                        logger.error("Embedded agent failed for index %d: %s", idx, exc)
                        embedded_results[idx] = {"found": False}

        results: list[ClassificationResult] = []
        for r in raw:
            idx = r["index"]
            sentence = sentences[idx]
            classification = r["classification"]
            agent_path: list[str] = ["classifier"]
            embedded_sentence: Optional[str] = None
            original_flag: Optional[str] = None

            if classification == "Incomplete" and idx in embedded_results:
                emb = embedded_results[idx]
                agent_path = ["classifier", "embedded"]

                if emb.get("found"):
                    emb_class = emb.get("classification", "")
                    if emb_class in VALID_CLASSIFICATIONS and emb_class != "Incomplete":
                        embedded_sentence = emb.get("embedded_sentence")
                        original_flag = "Incomplete"
                        classification = emb_class

            results.append(
                ClassificationResult(
                    sentence=sentence,
                    classification=classification,
                    agent_path=agent_path,
                    embedded_sentence=embedded_sentence,
                    original_flag=original_flag,
                    reasoning=r.get("reasoning", ""),
                )
            )

        return results

    def _batch_classify(self, sentences: list[str]) -> list[dict]:
        all_results: list[dict] = []
        for start in range(0, len(sentences), MAX_BATCH_SIZE):
            chunk = sentences[start: start + MAX_BATCH_SIZE]
            chunk_results = self._call_classifier_chunk(chunk, offset=start)
            all_results.extend(chunk_results)
        return all_results

    def _call_classifier_chunk(self, chunk: list[str], offset: int) -> list[dict]:
        if self._provider == PROVIDER_OPENAI:
            return self._call_classifier_chunk_openai(chunk, offset)
        return self._call_classifier_chunk_anthropic(chunk, offset)

    def _call_classifier_chunk_anthropic(self, chunk: list[str], offset: int) -> list[dict]:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(chunk))
        user_msg = (
            f"Classify the following {len(chunk)} sentence(s). "
            f"Return one classification per sentence.\n\n{numbered}"
        )
        response = self._call_with_retry(
            self._client.messages.create,
            model=self._model,
            max_tokens=2048,
            system=CLASSIFIER_SYSTEM_PROMPT,
            tools=[_CLASSIFIER_TOOL],
            tool_choice={"type": "tool", "name": "classify_sentences"},
            messages=[{"role": "user", "content": user_msg}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "classify_sentences":
                raw = block.input.get("classifications", [])
                for item in raw:
                    item["index"] = item["index"] + offset
                return raw
        return []

    def _call_classifier_chunk_openai(self, chunk: list[str], offset: int) -> list[dict]:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(chunk))
        user_msg = (
            f"Classify the following {len(chunk)} sentence(s). "
            f"Return one classification per sentence.\n\n{numbered}"
        )
        response = self._call_with_retry(
            self._client.chat.completions.create,
            model=self._model,
            max_tokens=2048,
            tools=[_OPENAI_CLASSIFIER_TOOL],
            tool_choice={"type": "function", "function": {"name": "classify_sentences"}},
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            data = json.loads(tool_calls[0].function.arguments)
            raw = data.get("classifications", [])
            for item in raw:
                item["index"] = item["index"] + offset
            return raw
        return []

    def _run_embedded_agent(self, sentence: str) -> dict:
        if self._provider == PROVIDER_OPENAI:
            return self._run_embedded_agent_openai(sentence)
        return self._run_embedded_agent_anthropic(sentence)

    def _run_embedded_agent_anthropic(self, sentence: str) -> dict:
        user_msg = f'Analyze this incomplete sentence for an embedded complete clause:\n\n"{sentence}"'
        response = self._call_with_retry(
            self._client.messages.create,
            model=self._model,
            max_tokens=512,
            system=EMBEDDED_SYSTEM_PROMPT,
            tools=[_EMBEDDED_TOOL],
            tool_choice={"type": "tool", "name": "report_embedded_sentence"},
            messages=[{"role": "user", "content": user_msg}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == "report_embedded_sentence":
                result = dict(block.input)
                if result.get("found") and result.get("classification") not in VALID_CLASSIFICATIONS:
                    result["found"] = False
                return result
        return {"found": False}

    def _run_embedded_agent_openai(self, sentence: str) -> dict:
        user_msg = f'Analyze this incomplete sentence for an embedded complete clause:\n\n"{sentence}"'
        response = self._call_with_retry(
            self._client.chat.completions.create,
            model=self._model,
            max_tokens=512,
            tools=[_OPENAI_EMBEDDED_TOOL],
            tool_choice={"type": "function", "function": {"name": "report_embedded_sentence"}},
            messages=[
                {"role": "system", "content": EMBEDDED_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            result = json.loads(tool_calls[0].function.arguments)
            if result.get("found") and result.get("classification") not in VALID_CLASSIFICATIONS:
                result["found"] = False
            return result
        return {"found": False}

    def _call_with_retry(self, func, *args, **kwargs):
        """Call func with exponential back-off on rate limits (Anthropic or OpenAI)."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except anthropic.RateLimitError:
                if attempt == MAX_RETRIES:
                    raise
                wait = BASE_BACKOFF * (2 ** attempt)
                logger.warning(
                    "Anthropic rate limit — backing off %.1fs (attempt %d/%d)",
                    wait, attempt + 1, MAX_RETRIES,
                )
                time.sleep(wait)
            except anthropic.APIStatusError as exc:
                if exc.status_code == 429:
                    if attempt == MAX_RETRIES:
                        raise
                    wait = BASE_BACKOFF * (2 ** attempt)
                    logger.warning("HTTP 429 (Anthropic) — backing off %.1fs", wait)
                    time.sleep(wait)
                else:
                    raise
            except Exception as exc:
                # OpenAI exceptions handled dynamically to avoid hard import dependency
                if _OPENAI_AVAILABLE:
                    if isinstance(exc, _openai.RateLimitError):
                        if attempt == MAX_RETRIES:
                            raise
                        wait = BASE_BACKOFF * (2 ** attempt)
                        logger.warning(
                            "OpenAI rate limit — backing off %.1fs (attempt %d/%d)",
                            wait, attempt + 1, MAX_RETRIES,
                        )
                        time.sleep(wait)
                        continue
                    if (
                        isinstance(exc, _openai.APIStatusError)
                        and exc.status_code == 429
                    ):
                        if attempt == MAX_RETRIES:
                            raise
                        wait = BASE_BACKOFF * (2 ** attempt)
                        logger.warning("HTTP 429 (OpenAI) — backing off %.1fs", wait)
                        time.sleep(wait)
                        continue
                raise
        raise RuntimeError("Exceeded max retries")


def classify_sentences(
    sentences: list[str],
    client=None,
    model: str = DEFAULT_MODEL,
    provider: str = PROVIDER_ANTHROPIC,
    api_key: Optional[str] = None,
) -> list[dict]:
    """Classify a list of sentences. Returns list of dicts."""
    pipeline = SentenceClassificationPipeline(
        client=client, model=model, provider=provider, api_key=api_key
    )
    return [r.to_dict() for r in pipeline.classify(sentences)]
