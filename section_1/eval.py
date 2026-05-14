"""Task 1.3 — eval.py
CLI evaluation script for the handwriting OCR pipeline.

Usage:
    python eval.py --input ground_truth.json --predicted predicted.json

Input files:
    ground_truth.json  — array of {sample_id, ground_truth} objects
    predicted.json     — array of {sample_id, predicted} objects

Output:
    Formatted evaluation report to stdout matching the reference spec.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluation import (
    compute_cer,
    compute_wer,
    compute_sentence_f1,
    compute_composite,
    compute_nsfpr,
    segment_sentences_for_eval,
)

WIDTH = 80
SEP = "=" * WIDTH


def _fmt_pct(val: float) -> str:
    return f"{val:.1%}"


def _load_json(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _build_lookup(items: list[dict], key_field: str, value_field: str) -> dict[str, str]:
    return {item[key_field]: item[value_field] for item in items if value_field in item}


def evaluate(gt_path: str, pred_path: str) -> None:
    gt_items = _load_json(gt_path)
    pred_items = _load_json(pred_path)
    pred_lookup = _build_lookup(pred_items, "sample_id", "predicted")

    results: list[dict] = []

    print(SEP)
    print("OCR EVALUATION REPORT")
    print(SEP)

    for gt in gt_items:
        sid = gt["sample_id"]
        gt_text = gt.get("ground_truth", "")
        pred_text = pred_lookup.get(sid, "")

        if not pred_text.strip():
            print(f"\nSAMPLE: {sid}")
            print("  (no prediction found — skipped)")
            continue

        gt_sents = segment_sentences_for_eval(gt_text)
        pred_sents = segment_sentences_for_eval(pred_text)

        cer = compute_cer(gt_text, pred_text)
        wer = compute_wer(gt_text, pred_text)
        sf1_result = compute_sentence_f1(gt_sents, pred_sents)
        sf1 = sf1_result["f1"]
        composite = compute_composite(cer, wer, sf1)
        nsfpr = compute_nsfpr(gt_text, pred_text)

        gt_recovered = sf1_result.get("gt_recovered_count", 0)
        gt_total = len(gt_sents)

        results.append(
            {
                "sample_id": sid,
                "cer": cer,
                "wer": wer,
                "sent_f1": sf1,
                "composite": composite,
                "nsfpr": nsfpr,
            }
        )

        print(f"\nSAMPLE: {sid}")
        print(f"  CER:          {cer:.3f}   ({_fmt_pct(cer)} character error rate)")
        print(f"  WER:          {wer:.3f}   ({_fmt_pct(wer)} word error rate)")
        print(
            f"  Sentence F1:  {sf1:.3f}   "
            f"({gt_recovered} of {gt_total} ground truth sentences recovered)"
        )
        print(f"  Composite:    {composite:.3f}")
        if nsfpr is not None:
            print(f"  NSFPR:        {nsfpr:.3f}   (non-standard form preservation rate)")

    if not results:
        print("\nNo samples evaluated.")
        return

    print(f"\n{SEP}")
    print("AGGREGATE SUMMARY")
    print(SEP)

    def _mean(key: str) -> float:
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else 0.0

    mean_cer = _mean("cer")
    mean_wer = _mean("wer")
    mean_f1 = _mean("sent_f1")
    mean_composite = _mean("composite")

    best = max(results, key=lambda r: r["composite"])
    worst = min(results, key=lambda r: r["composite"])

    print(f"  Mean CER:          {mean_cer:.3f}")
    print(f"  Mean WER:          {mean_wer:.3f}")
    print(f"  Mean Sentence F1:  {mean_f1:.3f}")
    print(f"  Mean Composite:    {mean_composite:.3f}")
    print(f"  Best:   {best['sample_id']:<35} (Composite: {best['composite']:.3f})")
    print(f"  Worst:  {worst['sample_id']:<35} (Composite: {worst['composite']:.3f})")
    print(f"  Weights: CER={0.50}  WER={0.30}  SentF1={0.20}")

    nsfpr_vals = [r["nsfpr"] for r in results if r.get("nsfpr") is not None]
    if nsfpr_vals:
        print(f"  Mean NSFPR:        {sum(nsfpr_vals)/len(nsfpr_vals):.3f}   (non-standard form preservation)")

    print(SEP)
    print(
        "NOTE: Numbers above depend on OCR engine performance.\n"
        "      The composite formula is: (1-CER)*0.50 + (1-WER)*0.30 + SentF1*0.20"
    )
    print(SEP)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate OCR predictions against ground truth."
    )
    parser.add_argument("--input", required=True, help="Path to ground_truth.json")
    parser.add_argument("--predicted", required=True, help="Path to predicted.json")
    args = parser.parse_args()
    evaluate(args.input, args.predicted)


if __name__ == "__main__":
    main()
