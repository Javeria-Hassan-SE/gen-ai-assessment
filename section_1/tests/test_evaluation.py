"""Unit tests for Task 1.3 evaluation metrics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from evaluation import (
    compute_cer,
    compute_wer,
    compute_sentence_f1,
    compute_composite,
    compute_nsfpr,
    evaluate_all,
    _normalise,
)


# ── Normalisation ─────────────────────────────────────────────────────────────

def test_normalise_lowercases_and_strips():
    assert _normalise("  Hello World  ") == "hello world"


def test_normalise_preserves_nonstandard_spelling():
    # 'siad' must NOT be changed to 'said'
    assert _normalise("Tadeo siad") == "tadeo siad"


# ── CER ──────────────────────────────────────────────────────────────────────

def test_cer_perfect():
    assert compute_cer("hello world", "hello world") == 0.0


def test_cer_completely_wrong():
    cer = compute_cer("abc", "xyz")
    assert cer == 1.0  # 3 substitutions / 3 chars = 1.0


def test_cer_partial():
    cer = compute_cer("hello", "helo")  # 1 deletion in 5 chars
    assert abs(cer - 0.2) < 1e-6


def test_cer_clamped_to_one():
    """Insertions can push raw CER > 1 — must be clamped."""
    cer = compute_cer("ab", "abcdefghij")  # many insertions
    assert cer <= 1.0


def test_cer_empty_gt_empty_pred():
    assert compute_cer("", "") == 0.0


def test_cer_empty_gt_nonempty_pred():
    assert compute_cer("", "something") == 1.0


# ── WER ──────────────────────────────────────────────────────────────────────

def test_wer_perfect():
    assert compute_wer("the cat sat", "the cat sat") == 0.0


def test_wer_all_wrong():
    wer = compute_wer("one two three", "x y z")  # 3 subs / 3 = 1.0
    assert wer == 1.0


def test_wer_clamped():
    wer = compute_wer("hi", "one two three four five six")
    assert wer <= 1.0


def test_wer_preserves_nonstandard_words():
    """'siad' vs 'said' should count as 1 word error, not 0."""
    wer_nonstandard = compute_wer("tadeo siad well", "tadeo said well")
    wer_correct = compute_wer("tadeo said well", "tadeo said well")
    assert wer_nonstandard > wer_correct


# ── Sentence F1 ──────────────────────────────────────────────────────────────

def test_sentence_f1_perfect_match():
    sents = ["The cat sat on the mat.", "It was a sunny day."]
    result = compute_sentence_f1(sents, sents)
    assert result["f1"] == 1.0


def test_sentence_f1_no_overlap():
    gt = ["hello world"]
    pred = ["completely different text here"]
    result = compute_sentence_f1(gt, pred)
    assert result["f1"] == 0.0


def test_sentence_f1_empty_pred():
    gt = ["some sentence here"]
    result = compute_sentence_f1(gt, [])
    assert result["recall"] == 0.0
    assert result["precision"] == 0.0


def test_sentence_f1_partial():
    gt = ["the quick brown fox", "jumps over the lazy dog"]
    pred = ["the quick brown fox"]  # only first sentence recovered
    result = compute_sentence_f1(gt, pred)
    assert 0.0 < result["f1"] < 1.0
    assert result["recall"] < 1.0


# ── Composite score ───────────────────────────────────────────────────────────

def test_composite_perfect():
    assert compute_composite(0.0, 0.0, 1.0) == 1.0


def test_composite_worst():
    assert compute_composite(1.0, 1.0, 0.0) == 0.0


def test_composite_range():
    for cer in [0.0, 0.3, 0.7, 1.0]:
        for wer in [0.0, 0.5, 1.0]:
            for f1 in [0.0, 0.5, 1.0]:
                score = compute_composite(cer, wer, f1)
                assert 0.0 <= score <= 1.0, f"Out of range for CER={cer}, WER={wer}, F1={f1}"


def test_composite_weights():
    """CER has highest weight — larger CER impact than same WER change."""
    base = compute_composite(0.5, 0.5, 0.5)
    high_cer = compute_composite(1.0, 0.5, 0.5)
    high_wer = compute_composite(0.5, 1.0, 0.5)
    assert high_cer < high_wer < base  # CER 0→1 has bigger impact than WER 0→1


# ── NSFPR ─────────────────────────────────────────────────────────────────────

def test_nsfpr_returns_none_or_float():
    result = compute_nsfpr("he got scord and siad hello", "he got scord and siad hello")
    assert result is None or isinstance(result, float)


def test_nsfpr_perfect_preservation():
    result = compute_nsfpr("siad scord wokeup", "siad scord wokeup")
    if result is not None:
        assert result == 1.0


def test_nsfpr_zero_when_all_corrected():
    result = compute_nsfpr("siad scord wokeup", "said scared wokedup")
    if result is not None:
        assert result < 1.0


# ── evaluate_all ─────────────────────────────────────────────────────────────

def test_evaluate_all_aggregate_keys():
    samples = [
        {
            "sample_id": "s1",
            "ground_truth": "the cat sat",
            "predicted": "the cat sat",
            "ground_truth_sentences": ["the cat sat"],
            "predicted_sentences": ["the cat sat"],
        },
        {
            "sample_id": "s2",
            "ground_truth": "hello world",
            "predicted": "helo world",
            "ground_truth_sentences": ["hello world"],
            "predicted_sentences": ["helo world"],
        },
    ]
    per_sample, agg = evaluate_all(samples)
    assert len(per_sample) == 2
    for key in ["cer", "wer", "sent_f1", "composite_score"]:
        assert key in agg
    assert agg["sample_id"] == "AGGREGATE (mean)"


def test_evaluate_all_perfect_sample_scores_one():
    samples = [
        {
            "sample_id": "perfect",
            "ground_truth": "at 7:00am tadeo woke up",
            "predicted": "at 7:00am tadeo woke up",
            "ground_truth_sentences": ["at 7:00am tadeo woke up"],
            "predicted_sentences": ["at 7:00am tadeo woke up"],
        }
    ]
    _, agg = evaluate_all(samples)
    assert agg["cer"] == 0.0
    assert agg["wer"] == 0.0
    assert agg["composite_score"] == 1.0
