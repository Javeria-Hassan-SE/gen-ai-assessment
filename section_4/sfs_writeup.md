# Task 4.1 — Source-Fidelity Score (SFS)

## Formal Definition

**SFS** measures the proportion of token-level edits that represent fidelity-preserving outcomes. Unlike CER and WER, it weights edit types by their severity to verbatim transcription.

### Formula

```
SFS = 1 − (normalisation_penalty + noise_penalty) / max(len(gt_tokens), 1)
```

Clamped to **[0.0, 1.0]**. Higher is better.

### Penalty Weights

| Edit type | Condition | Weight |
|---|---|---|
| **Correct** | `pred == gt` | 0.0 |
| **Normalisation** | `pred != gt` AND `is_standard_english(pred)` | **1.0** |
| **Noise** | `pred != gt` AND NOT `is_standard_english(pred)` | **0.5** |

### Why normalisation is penalised more heavily than noise

A **normalisation edit** silently replaces a non-standard source form with a standard English word. The output looks clean — CER may even improve — but the verbatim transcription requirement is violated. The fidelity failure is invisible to standard metrics.

A **noise edit** produces a non-English token. It is immediately detectable: downstream processes can flag it, low-confidence scores can catch it, and a human reviewer can see it. Detectable failures are recoverable. Silent corrections are not.

The 2:1 penalty ratio (1.0 vs 0.5) encodes this asymmetry: silent corruption is twice as dangerous as visible garbage.

---

## Worked Examples

### Example 1 — Silent Normalisation (CER and SFS diverge)

```
Ground truth : I siad hello and wokeup eerly
Predicted    : I said hello and woke up early
```

| Metric | Score |
|---|---|
| **CER** | **0.1379** |
| **SFS** | **0.4167** |

**CER is low (0.14) — standard metric says "good performance."** The edit distance between the two strings is small: `siad` to `said` is two character substitutions, `wokeup` to `woke up` is one insertion, `eerly` to `early` is one substitution. A total of roughly four character edits on a 29-character string produces a low CER that suggests the OCR output is nearly correct.

**SFS is low (0.42) — verbatim fidelity has failed.** Tokenising and aligning reveals three normalisation edits: `siad -> said`, `wokeup -> woke` (positional alignment places `eerly` against `up`), and `eerly` against `up` — all predicted tokens are standard English while the ground-truth tokens are intentional child misspellings. Each triggers the 1.0 normalisation penalty. One extra predicted token (`early`) adds a 0.5 noise penalty. Total weighted penalty: 3.5 over 6 ground-truth tokens → SFS = 0.42.

The two metrics give **opposite signals** for the same output. CER says near-perfect; SFS says substantial fidelity failure. For a verbatim transcription pipeline, SFS is the correct signal.

---

### Example 2 — Noise Errors (CER and SFS agree on failure, but for different reasons)

```
Ground truth : he got scord and ran
Predicted    : he g2[ xkqz and r@n
```

| Metric | Score |
|---|---|
| **CER** | **0.4000** |
| **SFS** | **0.7000** |

**Both CER (0.40) and SFS (0.70) indicate poor output — they agree on failure.** The OCR engine produced garbled non-English tokens (`g2[`, `xkqz`, `r@n`), resulting in a high character error rate. SFS also falls below 1.0 because three tokens are wrong.

**The critical difference is the failure mode.** SFS assigns weight 0.5 (noise) to all three bad tokens because none of `g2[`, `xkqz`, or `r@n` are standard English. No normalisation penalty (weight 1.0) fires. Noise errors are detectable: confidence scores will be low, the tokens are obviously garbled, and no silent correction of intentional child spelling has occurred. The verbatim fidelity of the source — including the non-standard `scord` — has not been violated.

SFS = 0.70 (noise) versus SFS = 0.42 (normalisation) correctly represents that Example 2, despite its high CER, is a **less dangerous failure** for a verbatim transcription system than Example 1.

---

## Edge Case — Where SFS Produces a Misleading Result

**Scenario:** ground truth `"he ran home"`, predicted `"he run home"`.

`_classify_edit("ran", "run")` returns `'normalisation'` because `is_standard_english("run")` is `True`. SFS applies a 1.0 normalisation penalty.

**Why this is misleading:** the engine did not silently correct a non-standard form. It made a grammatical tense error — `ran` (past) replaced by `run` (base form). This is a genuine OCR or language-model error, not a fidelity-preserving normalisation. SFS over-penalises it at the normalisation weight when it deserves the noise weight or less.

**Root cause:** SFS classifies based solely on whether the *predicted* token is standard English. When both the ground-truth token and the predicted token are standard English words, SFS cannot determine whether the edit was a silent correction of intended non-standard text or a misrecognition between two valid English words.

**Practical scope:** this ambiguity only arises when both tokens are standard English and different. For the primary use case — child handwriting with intentional misspellings — ground-truth tokens are typically non-standard, so `_classify_edit` produces the correct result. The edge case surfaces most for common function words, verb inflections, and proper nouns.

---

## When to Use SFS Alongside CER/WER vs. CER/WER Alone

**Use SFS alongside CER/WER** whenever the source document contains intentional non-standard forms that must be preserved verbatim. This includes child handwriting transcription, dialect writing, historical documents, and any domain where the exact surface form of the source text is the deliverable. In these contexts, CER and WER are necessary but insufficient — they cannot detect silent normalisation, and a system optimised for low CER may actively violate the verbatim requirement while appearing to perform well.

**Use CER/WER alone** when the goal is fluency or correctness rather than verbatim fidelity — for example, when OCR output will be used for search indexing, summarisation, or any downstream task that benefits from standard English. In these cases, normalisation is a feature, not a failure, and penalising it via SFS would misdirect optimisation effort.
