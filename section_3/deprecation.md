# Task 3.2 — Classification Model Deprecation

## Deliverable 1 — Step-by-Step Migration Plan

### Weeks 1–2: Benchmark Setup

Before touching any model, build a **frozen evaluation dataset** of 200 human-verified sentences:

| Category | Count | Notes |
|---|---|---|
| Simple | 80 | Includes non-standard spelling variants |
| Compound / Complex | 60 | Mixed, covering both categories |
| Incomplete | 40 | 20% share — matches confusion matrix distribution |
| Edge cases | 20 | Previously misclassified examples from production |

**Freeze this dataset.** It does not change during the migration.

Run the current deprecated model against it and record **baseline metrics:**
1. Classification accuracy per category
2. Incomplete routing accuracy (most sensitive)
3. Embedded Agent trigger rate
4. End-to-end latency per sentence
5. API cost per 1,000 sentences

---

### Weeks 3–4: Benchmark All Three Candidates

Run each candidate against the frozen dataset using **identical prompts** — do not modify prompts at this stage. This isolates model performance from prompt performance.

If a candidate scores poorly with the current prompts, attempt a **prompt tuning second pass** before eliminating it.

Record the same five metrics for each candidate.

**Decision framework (priority order):**

1. **Incomplete routing accuracy** must be ≥ current baseline — this is the most sensitive category
2. **Overall classification accuracy** must be ≥ current baseline
3. **End-to-end latency** must not increase by more than 20%
4. **Cost per 1,000 sentences** must not increase by more than 30%
5. If multiple candidates pass all four criteria, choose the one with the best Incomplete precision

---

### Weeks 5–6: Prompt Tuning on Winner

Take the winning candidate and tune prompts if needed. Re-run the frozen benchmark to confirm scores hold after prompt changes.

**This is the only phase where prompts change.**

---

### Weeks 7–8: Shadow Mode

Run the new model in **shadow mode** — every production request is processed by both the old and new model simultaneously. Old model output is returned to the client. New model output is stored silently for comparison.

Track **disagreement rate** — if the new model disagrees with the old model on more than **15%** of sentences, investigate before proceeding to canary.

---

### Weeks 9–10: Canary Deployment

Route **5%** of real traffic to the new model. Monitor disagreement rate, error rate, and latency.

Progression gates — each step requires **48 hours of stability** before advancing:

```
5% → 25% → 50% → 100%
```

---

### Weeks 11–12: Full Cutover and Cleanup

- 100% of traffic on the new model
- Keep old model endpoint available for **2 weeks** as emergency rollback
- Remove old model after 2 weeks with no issues

---

## Deliverable 2 — Rollout Strategy

### Shadow Mode First, Then Canary

Use shadow mode before canary. Do not split traffic from the start.

**Reason for shadow mode first:** Shadow mode costs extra API calls (both models run on every request) but produces **zero risk to production output**. It provides 2 weeks of real-world data comparison before any client ever sees new model output. For a classification system where errors affect downstream analysis, this is worth the additional API cost.

### Migrate Both Agents Independently, Not Simultaneously

Migrate the **Classifier Agent first**. Stabilise fully — minimum 2 weeks in production — then migrate the **Embedded Sentence Agent**.

**Reason:** If both agents are migrated simultaneously and something breaks, the failure cannot be isolated to one agent. The Classifier Agent handles 100% of traffic and has the richer confusion matrix history, making it the higher-risk migration. The Embedded Sentence Agent handles only sentences classified as `Incomplete` (approximately 8.5% of traffic based on 17/200 Incomplete sentences in the confusion matrix). Its blast radius is smaller, so it is the safer second migration.

---

## Deliverable 3 — Minimum Test Suite Before Any Production Traffic

**12 tests minimum across three categories.**

---

### Category Coverage Tests (5 tests)

One clean example per category — new model must match current model on all five:

| Test | Input type | Expected label |
|---|---|---|
| T1 | Clean Simple sentence | `Simple` |
| T2 | Clean Compound sentence | `Compound` |
| T3 | Clean Complex sentence | `Complex` |
| T4 | Clean Compound-Complex sentence | `Compound-Complex` |
| T5 | Clean Incomplete fragment | `Incomplete` |

---

### Incomplete Routing Path Tests (3 tests)

**Test 6 — Embedded sentence found:**
- Input: sentence classified as `Incomplete` where an embedded complete clause exists
- Assert: `final_label` equals the embedded sentence type, **not** `Incomplete`
- Assert: `agent_path` contains both `["classifier", "embedded"]`
- Assert: `original_flag` is `"Incomplete"`

**Test 7 — Embedded sentence not found:**
- Input: sentence classified as `Incomplete` where no embedded clause exists
- Assert: `final_label` is `Incomplete`
- Assert: `agent_path` shows Embedded Agent was called
- Assert: `original_flag` is `"Incomplete"`

**Test 8 — Non-Incomplete sentence does not trigger Embedded Agent:**
- Input: sentence classified as anything other than `Incomplete`
- Assert: Embedded Agent is **never called**
- Assert: `agent_path` contains only `["classifier"]`
- This test verifies that routing enforcement in code works correctly with the new model — a different model must not accidentally trigger the Embedded Agent for non-Incomplete sentences

---

### Edge Case Tests (4 tests)

**Test 9 — Non-standard spelling, complete Simple sentence:**
- Input: `"I siad hello"` (misspelled, but subject + predicate present)
- Assert: `Simple` — must not be classified `Incomplete` due to misspelling

**Test 10 — Simple/Compound boundary with coordinating conjunction:**
- Input: sentence with "and" joining two verb phrases under one subject
- Assert: `Simple` — tests the highest-error boundary from the confusion matrix
- Verifies the model does not over-split VP coordination into a Compound classification

**Test 11 — Complex/Compound-Complex boundary with subordinating clause:**
- Input: sentence with one independent clause and one subordinate clause
- Assert: `Complex` — verifies the model does not over-promote to `Compound-Complex`

**Test 12 — Genuine Incomplete fragment with no embedded sentence:**
- Input: `"because and then"`
- Assert: `final_label` is `Incomplete`
- Assert: Embedded Agent was called (confirm `agent_path`)
- Assert: Embedded Agent returned `found = false`
