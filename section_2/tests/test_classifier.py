"""Tests for the two-agent sentence classification pipeline.

Covers:
  T1 — Normal batch (no Incompletes): only Classifier Agent runs
  T2 — Batch with several Incomplete sentences: Embedded Agent triggered
  T3 — Embedded Agent finds no embedded sentence (found=False)
  T4 — Embedded Agent finds multiple embedded sentences (highest complexity wins)
  T5 — Rate-limit retry (RateLimitError on first attempt, succeeds on second)
  T6 — Missing items in Classifier Agent response (recovery path)
"""

from __future__ import annotations

import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import anthropic as _anthropic

from classifier import (
    SentenceClassificationPipeline,
    ClassificationResult,
    classify_sentences,
    VALID_CLASSIFICATIONS,
)

def _make_tool_block(name: str, input_data: dict):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_data
    return block


def _make_response(blocks):
    resp = MagicMock()
    resp.content = blocks
    return resp


def _classifier_response(classifications: list[dict]):
    return _make_response([
        _make_tool_block("classify_sentences", {"classifications": classifications})
    ])


def _embedded_response(result: dict):
    return _make_response([
        _make_tool_block("report_embedded_sentence", result)
    ])


class TestNormalBatchNoIncomplete(unittest.TestCase):
    """T1 — All sentences classified without triggering Embedded Agent."""

    def setUp(self):
        self.client = MagicMock()
        self.pipeline = SentenceClassificationPipeline(client=self.client)

    def test_simple_sentences_only_classifier_runs(self):
        sentences = ["he ran home", "she likes cats", "they went to school"]
        self.client.messages.create.return_value = _classifier_response([
            {"index": 0, "classification": "Simple", "reasoning": "one clause"},
            {"index": 1, "classification": "Simple", "reasoning": "one clause"},
            {"index": 2, "classification": "Simple", "reasoning": "one clause"},
        ])

        results = self.pipeline.classify(sentences)

        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.classification, "Simple")
            self.assertEqual(r.agent_path, ["classifier"])
            self.assertIsNone(r.embedded_sentence)
        # Embedded agent never called — only 1 API call total
        self.assertEqual(self.client.messages.create.call_count, 1)

    def test_mixed_non_incomplete_classifications(self):
        sentences = [
            "he ran and she walked",
            "she went because it rained",
            "he ran and she walked because it rained",
        ]
        self.client.messages.create.return_value = _classifier_response([
            {"index": 0, "classification": "Compound"},
            {"index": 1, "classification": "Complex"},
            {"index": 2, "classification": "Compound-Complex"},
        ])

        results = self.pipeline.classify(sentences)

        self.assertEqual(results[0].classification, "Compound")
        self.assertEqual(results[1].classification, "Complex")
        self.assertEqual(results[2].classification, "Compound-Complex")
        # Still only one API call
        self.assertEqual(self.client.messages.create.call_count, 1)

    def test_empty_input(self):
        results = self.pipeline.classify([])
        self.assertEqual(results, [])
        self.client.messages.create.assert_not_called()

    def test_result_order_preserved(self):
        sentences = ["A", "B", "C"]
        self.client.messages.create.return_value = _classifier_response([
            {"index": 2, "classification": "Simple"},
            {"index": 0, "classification": "Compound"},
            {"index": 1, "classification": "Complex"},
        ])
        results = self.pipeline.classify(sentences)
        # Must match input order even when response arrives out of order
        self.assertEqual(results[0].sentence, "A")
        self.assertEqual(results[0].classification, "Compound")
        self.assertEqual(results[1].sentence, "B")
        self.assertEqual(results[1].classification, "Complex")
        self.assertEqual(results[2].sentence, "C")
        self.assertEqual(results[2].classification, "Simple")


class TestBatchWithIncomplete(unittest.TestCase):
    """T2 — Incomplete sentences trigger Embedded Agent."""

    def setUp(self):
        self.client = MagicMock()
        self.pipeline = SentenceClassificationPipeline(client=self.client)

    def test_single_incomplete_triggers_embedded(self):
        sentences = ["he ran home", "becaus I go Home and"]

        # Classifier call
        classifier_resp = _classifier_response([
            {"index": 0, "classification": "Simple"},
            {"index": 1, "classification": "Incomplete"},
        ])
        # Embedded call
        embedded_resp = _embedded_response({
            "found": True,
            "embedded_sentence": "I go Home",
            "classification": "Simple",
        })
        self.client.messages.create.side_effect = [classifier_resp, embedded_resp]

        results = self.pipeline.classify(sentences)

        self.assertEqual(results[0].classification, "Simple")
        self.assertEqual(results[0].agent_path, ["classifier"])

        self.assertEqual(results[1].classification, "Simple")
        self.assertEqual(results[1].agent_path, ["classifier", "embedded"])
        self.assertEqual(results[1].embedded_sentence, "I go Home")
        self.assertEqual(results[1].original_flag, "Incomplete")

        self.assertEqual(self.client.messages.create.call_count, 2)

    def test_multiple_incompletes_all_upgraded(self):
        sentences = [
            "becaus I go Home and",
            "and then the big",
            "she went because it rained",
        ]

        classifier_resp = _classifier_response([
            {"index": 0, "classification": "Incomplete"},
            {"index": 1, "classification": "Incomplete"},
            {"index": 2, "classification": "Complex"},
        ])
        emb_resp_0 = _embedded_response({
            "found": True,
            "embedded_sentence": "I go Home",
            "classification": "Simple",
        })
        emb_resp_1 = _embedded_response({"found": False})

        # classifier first, then embedded calls for idx 0 and 1
        self.client.messages.create.side_effect = [
            classifier_resp, emb_resp_0, emb_resp_1
        ]

        results = self.pipeline.classify(sentences)

        self.assertEqual(results[0].classification, "Simple")
        self.assertEqual(results[0].original_flag, "Incomplete")

        self.assertEqual(results[1].classification, "Incomplete")
        self.assertIsNone(results[1].original_flag)

        self.assertEqual(results[2].classification, "Complex")
        self.assertEqual(results[2].agent_path, ["classifier"])


class TestEmbeddedFindsNothing(unittest.TestCase):
    """T3 — Embedded Agent returns found=False; classification stays Incomplete."""

    def setUp(self):
        self.client = MagicMock()
        self.pipeline = SentenceClassificationPipeline(client=self.client)

    def test_found_false_keeps_incomplete(self):
        sentences = ["and then the big"]
        self.client.messages.create.side_effect = [
            _classifier_response([{"index": 0, "classification": "Incomplete"}]),
            _embedded_response({"found": False}),
        ]

        results = self.pipeline.classify(sentences)

        self.assertEqual(results[0].classification, "Incomplete")
        self.assertEqual(results[0].agent_path, ["classifier", "embedded"])
        self.assertIsNone(results[0].embedded_sentence)
        self.assertIsNone(results[0].original_flag)

    def test_invalid_embedded_classification_treated_as_not_found(self):
        """If embedded returns an invalid classification string, treat as found=False."""
        sentences = ["when the rain fell on"]
        self.client.messages.create.side_effect = [
            _classifier_response([{"index": 0, "classification": "Incomplete"}]),
            _embedded_response({
                "found": True,
                "embedded_sentence": "the rain fell on",
                "classification": "INVALID_LABEL",
            }),
        ]

        results = self.pipeline.classify(sentences)

        # Invalid classification → pipeline ignores upgrade
        self.assertEqual(results[0].classification, "Incomplete")


class TestEmbeddedFindsMultiple(unittest.TestCase):
    """T4 — Embedded Agent selects the highest-complexity embedded sentence."""

    def setUp(self):
        self.client = MagicMock()
        self.pipeline = SentenceClassificationPipeline(client=self.client)

    def test_highest_complexity_wins(self):
        """When prompt instructs model to return the highest-complexity embedded
        sentence, the pipeline accepts whatever the model returns."""
        sentences = ["becaus she went and he stayed becaus it rained"]
        self.client.messages.create.side_effect = [
            _classifier_response([{"index": 0, "classification": "Incomplete"}]),
            _embedded_response({
                "found": True,
                "embedded_sentence": "she went and he stayed becaus it rained",
                "classification": "Compound-Complex",
                "reasoning": "highest complexity found",
            }),
        ]

        results = self.pipeline.classify(sentences)

        self.assertEqual(results[0].classification, "Compound-Complex")
        self.assertEqual(results[0].embedded_sentence,
                         "she went and he stayed becaus it rained")

    def test_compound_complex_beats_simple(self):
        sentences = ["fragment text here"]
        self.client.messages.create.side_effect = [
            _classifier_response([{"index": 0, "classification": "Incomplete"}]),
            _embedded_response({
                "found": True,
                "embedded_sentence": "she went and he stayed because it rained",
                "classification": "Compound-Complex",
            }),
        ]

        results = self.pipeline.classify(sentences)
        self.assertEqual(results[0].classification, "Compound-Complex")

    def test_embedded_incomplete_not_upgraded(self):
        """Rule 6: if the embedded sentence itself is Incomplete, keep Incomplete."""
        sentences = ["some fragment"]
        self.client.messages.create.side_effect = [
            _classifier_response([{"index": 0, "classification": "Incomplete"}]),
            # Embedded returns found=True but classification=Incomplete (violates schema)
            # Pipeline should not upgrade in this case
            _embedded_response({
                "found": True,
                "embedded_sentence": "some fragment part",
                "classification": "Incomplete",
            }),
        ]

        results = self.pipeline.classify(sentences)
        # Incomplete is not a valid upgrade target
        self.assertEqual(results[0].classification, "Incomplete")


class TestRateLimitRetry(unittest.TestCase):
    """T5 — RateLimitError on first attempt; succeeds on second."""

    def setUp(self):
        self.client = MagicMock()
        self.pipeline = SentenceClassificationPipeline(client=self.client)

    @patch("classifier.time.sleep")
    def test_rate_limit_retried_and_succeeds(self, mock_sleep):
        sentences = ["he ran home"]

        success_resp = _classifier_response([
            {"index": 0, "classification": "Simple"}
        ])
        self.client.messages.create.side_effect = [
            _anthropic.RateLimitError.__new__(_anthropic.RateLimitError),
            success_resp,
        ]
        # Patch RateLimitError to be raise-able without __init__ args
        with patch.object(
            self.client.messages, "create",
            side_effect=[_anthropic.RateLimitError("rate limited", response=MagicMock(), body={}), success_resp]
        ):
            results = self.pipeline.classify(sentences)

        self.assertEqual(results[0].classification, "Simple")
        # sleep was called for the back-off
        mock_sleep.assert_called_once()

    @patch("classifier.time.sleep")
    def test_rate_limit_exhausted_raises(self, mock_sleep):
        sentences = ["he ran home"]

        err = _anthropic.RateLimitError("rate limited", response=MagicMock(), body={})
        self.client.messages.create.side_effect = [err] * 10

        with self.assertRaises((_anthropic.RateLimitError, RuntimeError)):
            self.pipeline.classify(sentences)

    @patch("classifier.time.sleep")
    def test_http_429_retried(self, mock_sleep):
        sentences = ["she walked"]

        mock_response = MagicMock()
        mock_response.status_code = 429
        err = _anthropic.APIStatusError("429", response=mock_response, body={})
        success_resp = _classifier_response([{"index": 0, "classification": "Simple"}])

        self.client.messages.create.side_effect = [err, success_resp]

        results = self.pipeline.classify(sentences)
        self.assertEqual(results[0].classification, "Simple")
        mock_sleep.assert_called_once()

    @patch("classifier.time.sleep")
    def test_non_429_api_error_raises_immediately(self, mock_sleep):
        sentences = ["she walked"]

        mock_response = MagicMock()
        mock_response.status_code = 500
        err = _anthropic.APIStatusError("500", response=mock_response, body={})
        self.client.messages.create.side_effect = err

        with self.assertRaises(_anthropic.APIStatusError):
            self.pipeline.classify(sentences)

        # No back-off sleep for non-429
        mock_sleep.assert_not_called()


class TestMissingItemsRecovery(unittest.TestCase):
    """T6 — Classifier omits some sentences; pipeline retries individually."""

    def setUp(self):
        self.client = MagicMock()
        self.pipeline = SentenceClassificationPipeline(client=self.client)

    def test_missing_item_recovered_on_retry(self):
        sentences = ["he ran", "she walked", "they played"]

        # First call returns only index 0 and 2 (index 1 missing)
        first_resp = _classifier_response([
            {"index": 0, "classification": "Simple"},
            {"index": 2, "classification": "Simple"},
        ])
        # Retry for missing index 1
        retry_resp = _classifier_response([
            {"index": 0, "classification": "Compound"},
        ])
        self.client.messages.create.side_effect = [first_resp, retry_resp]

        results = self.pipeline.classify(sentences)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].classification, "Simple")
        self.assertEqual(results[1].classification, "Compound")  # recovered
        self.assertEqual(results[2].classification, "Simple")

    def test_permanently_missing_falls_back_to_incomplete(self):
        sentences = ["he ran", "???"]

        # First call returns only index 0
        first_resp = _classifier_response([
            {"index": 0, "classification": "Simple"},
        ])
        # Retry for index 1 returns nothing
        empty_resp = _make_response([])
        self.client.messages.create.side_effect = [first_resp, empty_resp]

        results = self.pipeline.classify(sentences)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].classification, "Simple")
        # Fallback for permanently missing
        self.assertEqual(results[1].classification, "Incomplete")


class TestConvenienceFunction(unittest.TestCase):
    """classify_sentences() wraps pipeline and returns list of dicts."""

    def test_returns_list_of_dicts(self):
        client = MagicMock()
        client.messages.create.return_value = _classifier_response([
            {"index": 0, "classification": "Simple", "reasoning": "one clause"},
        ])
        results = classify_sentences(["he ran home"], client=client)

        self.assertIsInstance(results, list)
        self.assertIsInstance(results[0], dict)
        self.assertIn("sentence", results[0])
        self.assertIn("classification", results[0])
        self.assertIn("agent_path", results[0])


class TestClassificationResultToDict(unittest.TestCase):
    """ClassificationResult.to_dict() returns all required keys."""

    def test_to_dict_keys(self):
        r = ClassificationResult(
            sentence="he ran",
            classification="Simple",
            agent_path=["classifier"],
            embedded_sentence=None,
            original_flag=None,
            reasoning="one clause",
        )
        d = r.to_dict()
        expected_keys = {
            "sentence", "classification", "agent_path",
            "embedded_sentence", "original_flag", "reasoning",
        }
        self.assertEqual(set(d.keys()), expected_keys)

    def test_to_dict_values(self):
        r = ClassificationResult(
            sentence="becaus I go Home and",
            classification="Simple",
            agent_path=["classifier", "embedded"],
            embedded_sentence="I go Home",
            original_flag="Incomplete",
            reasoning="embedded clause found",
        )
        d = r.to_dict()
        self.assertEqual(d["sentence"], "becaus I go Home and")
        self.assertEqual(d["classification"], "Simple")
        self.assertEqual(d["agent_path"], ["classifier", "embedded"])
        self.assertEqual(d["embedded_sentence"], "I go Home")
        self.assertEqual(d["original_flag"], "Incomplete")


if __name__ == "__main__":
    unittest.main()
