# Task 4.2 — Hybrid OCR Fallback Strategy

## Defining and Measuring the Confidence Threshold

EasyOCR returns a confidence score per detected text region in **[0.0, 1.0]**. The threshold is not a single fixed value — it is **calibrated per document type** using a held-out validation set.

**Calibration process:**
1. Take 50 images with known ground truth from the target document type.
2. Run traditional OCR only.
3. Compute per-region CER for each detected bounding box.
4. Plot CER against confidence score as a scatter plot.
5. Find the confidence value below which CER exceeds **0.30** — this becomes the initial threshold.
6. Typical value for child handwriting on ruled paper: **0.45 – 0.60**.

The threshold is stored as a configurable environment variable (`OCR_CONFIDENCE_THRESHOLD`), not hardcoded. It is re-calibrated quarterly or whenever a new document type is onboarded.

**Fallback trigger unit — region-level, not word-level:**

If any word in a detected region has confidence below the threshold, the **entire bounding box** is flagged for LLM fallback. Character-level OCR errors in one word frequently affect adjacent word segmentation. Word-level flagging produces fragmented, misaligned crops that are harder for the LLM to interpret than a full region crop.

---

## Merging Traditional and LLM Outputs for the Same Image

The merge strategy is **region-based**, not document-based. The LLM never sees the full page.

**Per-page process:**

1. **High-confidence regions:** use traditional OCR output verbatim. No LLM call.
2. **Low-confidence regions:** send the cropped region image to the vision LLM. Use LLM output for that region only.
3. **Reading order reconstruction:** sort all regions — both traditional and LLM-produced — by bounding box Y coordinate, then X coordinate. This restores natural reading order across the merged output.
4. **Provenance tagging:** store a source flag per region:
   ```json
   { "source": "traditional" }
   { "source": "llm_fallback" }
   ```
   This flag is mandatory for auditing, for NSFPR computation per source, and for evaluating whether the fallback is improving or degrading output.

**Only flagged crops are sent to the LLM.** This minimises API cost, minimises the surface area where the LLM can normalise text, and keeps the traditional engine as the primary transcription source for the majority of content.

---

## Guardrails to Prevent the LLM from Correcting Verbatim Text

### Guardrail 1 — Explicit System Prompt Instruction

The LLM system prompt states verbatim:

> "You are a transcription assistant. Your only job is to read exactly what is written in the image, character by character. Do not correct spelling. Do not fix grammar. Do not normalise non-standard words. If the handwriting says 'siad' transcribe 'siad'. If it says 'wokeup' transcribe 'wokeup'. Reproduce exactly what you see."

### Guardrail 2 — Post-LLM Normalisation Audit

After receiving LLM output for a region, run the same **NSFPR check** used in the main pipeline. If the LLM output contains standard English corrections of words that appeared non-standard in the traditional OCR output for the same region, flag the region for **human review** rather than accepting the LLM output silently. The LLM output is rejected; the traditional OCR output — however low-confidence — is retained as the safer option.

### Guardrail 3 — Traditional Output as Anchor

For each low-confidence region, compare LLM output token by token against the traditional OCR output. Where the traditional engine produced a token with confidence **above 0.30** (even if the region as a whole fell below the fallback threshold), **prefer the traditional token** over the LLM token. The LLM is trusted only for tokens where the traditional engine had zero confidence or produced clear garbage — specifically, non-alphanumeric output or isolated single characters with confidence below **0.20**.

### Guardrail 4 — Structural Enforcement via Orchestrator

**The orchestrator decides which regions go to the LLM — never the LLM itself.** The routing logic is Python code that reads confidence scores from the traditional OCR output and routes bounding boxes accordingly. The LLM cannot influence high-confidence regions because it never receives them. This is an architectural guarantee, not a prompt-level instruction.

---

## Evaluating Whether Hybrid Outperforms Traditional-Only Baseline

Run evaluation on the same held-out 50-image validation set used for threshold calibration. Compute four metrics for both **traditional-only** and **hybrid** pipelines.

### Metric 1 — CER per Region Type

Compute CER separately for:
- **High-confidence regions** — identical for both pipelines (traditional used in both cases); serves as a sanity check.
- **Low-confidence regions** — differs between pipelines; this is where hybrid must demonstrate improvement.

If hybrid does not improve CER on low-confidence regions, the threshold is miscalibrated (too many regions flagged) or the LLM prompt is normalising instead of transcribing.

### Metric 2 — NSFPR (Non-Standard Form Preservation Rate)

Compute NSFPR across all regions for both pipelines. **Hybrid must not decrease NSFPR relative to traditional-only.** If it does, the LLM guardrails are insufficient — the LLM is silently correcting non-standard forms that the traditional engine preserved, even imperfectly.

### Metric 3 — Cost per Image

LLM API calls have a per-call cost. Measure average API cost per image under the hybrid pipeline. If cost exceeds **3× the traditional-only baseline** without a meaningful CER improvement on low-confidence regions, the threshold is too low — too many regions are being sent to the LLM unnecessarily.

### Metric 4 — Regions Improved vs. Degraded

For each low-confidence region, classify the hybrid outcome as:

| Outcome | Condition |
|---|---|
| **Improved** | Hybrid region CER < traditional region CER |
| **Neutral** | Same |
| **Degraded** | Hybrid region CER > traditional region CER (LLM made it worse) |

Report the ratio across all fallback regions. If **degraded > 10%** of fallback regions, the LLM is actively harming output quality and the guardrails need strengthening before the hybrid pipeline can be deployed.

### Decision Rule

Hybrid is considered to **outperform traditional-only** if and only if all three conditions hold:

1. CER on low-confidence regions improves by **at least 15%**
2. **NSFPR does not decrease**
3. **Degraded region ratio is below 10%**

All three conditions must pass. A CER improvement that comes with a NSFPR decrease or a high degradation ratio is not an acceptable trade-off for a verbatim transcription system.
