# Task 3.3 — The Fidelity vs. Correction Trade-off

## Why CER and WER Fail to Detect Silent Normalisation

Standard OCR accuracy metrics measure edit distance between predicted output and a ground truth reference. A lower CER means fewer character-level changes between the two strings. The critical flaw for verbatim transcription is that **CER treats all edits as equivalent regardless of direction.**

When an OCR engine changes `"siad"` to `"said"`, this counts as two character substitutions — which actually **improves** CER if the ground truth reference was `"said"`. The engine has silently normalised an intentional misspelling and the metric rewarded it. In a verbatim transcription pipeline this is a correctness failure masquerading as a quality improvement. CER and WER cannot distinguish between the engine introducing noise and the engine silently correcting the source text.

This problem compounds at the LLM classification layer. If OCR partially fails and produces a near-correct non-standard form, a downstream LLM optimised for fluency will complete the normalisation. By the time the text reaches the classifier, both the original form and the silent correction look clean. CER is low. WER is low. The verbatim requirement has been violated invisibly, with no signal in any standard metric.

---

## Specific Preprocessing and Post-processing Safeguards

**First: never pass OCR output through any LLM for cleaning before classification.** Enforce this architecturally — OCR output is written directly to the database and read directly by the Classification Queue with no intermediate processing step where a language model can touch the text. The architecture makes accidental normalisation structurally impossible, not just policy-prohibited.

**Second: build a post-OCR vocabulary audit.** Flag any output word that is a standard English correction of a known non-standard input form. `"said"` appearing where `"siad"` was expected is a silent-correction signal, not an OCR success. This audit runs as a post-processing check against the domain lexicon, not as a correction step.

**Third: confidence thresholding with preservation bias.** When OCR confidence is low on a word, prefer the raw engine output over any post-processing correction. A low-confidence non-standard form is more faithful to the source than a high-confidence normalised one for this use case. The system must be biased toward transcription accuracy, not linguistic correctness.

---

## The NSFPR Metric and CI/CD Integration

**Non-Standard Form Preservation Rate (NSFPR)** is defined as:

```
NSFPR = (non-standard forms preserved verbatim) / (non-standard forms present in ground truth)
```

Before deployment, create a **reference lexicon** of known non-standard forms from the document domain. After OCR, compute NSFPR against this lexicon. A score below **0.90** triggers investigation — meaning more than 10% of expected non-standard forms were silently altered.

The **CI/CD build gate** requires all three conditions to pass on every commit:

```
CER  ≤ baseline
WER  ≤ baseline
NSFPR ≥ 0.90
```

CER and WER can both improve and the build still **fails** if NSFPR drops below threshold. This enforces the invariant that preprocessing improvements must not come at the cost of verbatim fidelity. Every commit to the preprocessing pipeline runs this gate automatically against reference test pages with known ground truth. A preprocessing change that makes OCR more accurate on standard text while silently normalising non-standard forms is treated as a regression, not an improvement.
