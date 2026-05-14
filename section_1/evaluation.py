"""Task 1.3 — Evaluation metrics: CER, WER, Sentence F1, Composite Score, NSFPR."""

from __future__ import annotations

import editdistance
import numpy as np

from config import EVAL_WEIGHTS, SENTENCE_OVERLAP_THRESHOLD

try:
    from spellchecker import SpellChecker as _SpellChecker
    _SPELLCHECKER_AVAILABLE = True
except ImportError:
    _SpellChecker = None
    _SPELLCHECKER_AVAILABLE = False

try:
    import spacy as _spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _spacy = None
    _SPACY_AVAILABLE = False


def _normalise(text: str) -> str:
    # Internal whitespace and non-standard spellings are intentionally preserved —
    # correcting 'siad' to 'said' improves CER while breaking verbatim fidelity.
    return text.lower().strip()


def compute_cer(ground_truth: str, predicted: str) -> float:
    """Character-level edit distance normalised by ground-truth length, clamped to [0, 1]."""
    gt = _normalise(ground_truth)
    pred = _normalise(predicted)
    if not gt:
        return 0.0 if not pred else 1.0
    return min(1.0, editdistance.eval(gt, pred) / len(gt))


def compute_wer(ground_truth: str, predicted: str) -> float:
    """Word-level edit distance normalised by ground-truth word count, clamped to [0, 1]."""
    gt_words = _normalise(ground_truth).split()
    pred_words = _normalise(predicted).split()
    if not gt_words:
        return 0.0 if not pred_words else 1.0
    return min(1.0, editdistance.eval(gt_words, pred_words) / len(gt_words))


def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def compute_sentence_f1(
    gt_sentences: list[str],
    pred_sentences: list[str],
    threshold: float = SENTENCE_OVERLAP_THRESHOLD,
) -> dict:
    """Sentence-level precision, recall, and F1 via Jaccard token overlap.

    A predicted sentence counts as correct if its token overlap with any GT sentence
    meets the threshold. Also returns gt_recovered_count for CLI display.
    """
    if not gt_sentences:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "gt_recovered_count": 0}

    gt_sets = [set(_normalise(s).split()) for s in gt_sentences]
    pred_sets = [set(_normalise(s).split()) for s in pred_sentences] if pred_sentences else []

    pred_matched = [
        any(_jaccard(p, g) >= threshold for g in gt_sets) for p in pred_sets
    ]
    gt_recovered_flags = [
        any(_jaccard(g, p) >= threshold for p in pred_sets) for g in gt_sets
    ]

    precision = sum(pred_matched) / len(pred_matched) if pred_matched else 0.0
    recall = sum(gt_recovered_flags) / len(gt_recovered_flags)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gt_recovered_count": sum(gt_recovered_flags),
    }


def compute_composite(cer: float, wer: float, sent_f1: float) -> float:
    """Weighted composite: (1 - CER)*0.50 + (1 - WER)*0.30 + SentF1*0.20."""
    return round(
        (1.0 - cer) * 0.50 + (1.0 - wer) * 0.30 + sent_f1 * 0.20,
        4,
    )


def compute_nsfpr(ground_truth: str, predicted: str) -> float | None:
    """Proportion of intentional misspellings preserved verbatim in prediction.

    Returns None if pyspellchecker is not installed.
    """
    if not _SPELLCHECKER_AVAILABLE:
        return None

    spell = _SpellChecker()
    gt_words = _normalise(ground_truth).split()
    pred_words_set = set(_normalise(predicted).split())

    nonstandard = [w for w in gt_words if spell.unknown([w])]
    if not nonstandard:
        return 1.0

    preserved = sum(1 for w in nonstandard if w in pred_words_set)
    return round(preserved / len(nonstandard), 4)


def segment_sentences_for_eval(text: str) -> list[str]:
    """Segment text into sentences using spaCy rule-based sentencizer."""
    if not _SPACY_AVAILABLE:
        return [text] if text.strip() else []

    nlp = _spacy.blank("en")
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def evaluate_sample(sample: dict) -> dict:
    """Compute all metrics for a single {sample_id, ground_truth, predicted} record.

    Sentences are auto-segmented so the JSON doesn't need pre-split sentence lists.
    """
    gt = sample.get("ground_truth", "")
    pred = sample.get("predicted", "")
    gt_sents = segment_sentences_for_eval(gt) if gt else []
    pred_sents = segment_sentences_for_eval(pred) if pred else []

    cer = compute_cer(gt, pred)
    wer = compute_wer(gt, pred)
    sf1 = compute_sentence_f1(gt_sents, pred_sents)
    composite = compute_composite(cer, wer, sf1["f1"])
    nsfpr = compute_nsfpr(gt, pred)

    result: dict = {
        "sample_id": sample.get("sample_id", "unknown"),
        "cer": cer,
        "wer": wer,
        "sent_precision": sf1["precision"],
        "sent_recall": sf1["recall"],
        "sent_f1": sf1["f1"],
        "composite_score": composite,
    }
    if nsfpr is not None:
        result["nsfpr"] = nsfpr
    return result


def evaluate_all(samples: list[dict]) -> tuple[list[dict], dict]:
    """Evaluate all samples and return (per_sample_results, aggregate_row)."""
    results = [evaluate_sample(s) for s in samples]

    numeric_keys = ["cer", "wer", "sent_precision", "sent_recall", "sent_f1", "composite_score"]
    if results and "nsfpr" in results[0]:
        numeric_keys.append("nsfpr")

    aggregate: dict = {k: round(float(np.mean([r[k] for r in results if k in r])), 4) for k in numeric_keys}
    aggregate["sample_id"] = "AGGREGATE (mean)"

    return results, aggregate
