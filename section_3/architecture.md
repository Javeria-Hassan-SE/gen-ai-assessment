# Task 3.1 — End-to-End System Architecture

## Deliverable 1 — System Architecture

### Data Flow

**1. REST API Layer (FastAPI)**

- Accepts `POST /upload`
- Validates file type (PDF, PNG, JPEG)
- Assigns UUID `job_id`
- Stores raw image to **Object Storage** (S3 or equivalent)
- Publishes message to **Upload Queue**
- Returns `{ job_id, status: "queued" }` immediately

**2. Upload Queue (Redis Streams or SQS)**

Message payload:
```
{ job_id, image_path, timestamp }
```

**3. OCR Worker Pool (3–5 stateless workers)**

Each worker:
1. Pulls message from Upload Queue
2. Downloads image from Object Storage
3. Runs `pdf2image` for PDFs (one page at a time — never the full document in memory)
4. Runs `preprocess_image()` per page
5. Runs OCR engine
6. Excludes metadata region
7. Runs spaCy sentence segmentation
8. Stores OCR result to **PostgreSQL** (`ocr_results` table)
9. Publishes message to **Classification Queue**

---

> ### ⬛ BOUNDARY: OCR ends here — Classification begins here

---

**4. Classification Queue (separate queue from Upload Queue)**

Message payload:
```
{ job_id, page_num, sentences: [...] }
```

Keeping this queue separate from the Upload Queue means OCR and classification fail independently. If the LLM API goes down, OCR jobs continue completing and results accumulate in the Classification Queue until the API recovers.

**5. Classifier Agent Worker (2–3 workers)**

Each worker:
1. Pulls batches from Classification Queue
2. Calls Classifier Agent API
3. Checks response for completeness — retries missing sentences individually
4. Pushes sentences classified `Incomplete` to **Embedded Agent Sub-queue**
5. Stores batch results to **PostgreSQL** (`classifications` table)

**6. Embedded Agent Sub-queue**

- One sentence at a time
- Decouples Embedded Agent processing from the main classification batch

**7. Embedded Sentence Agent Worker**

1. Pulls one sentence from Embedded Agent Sub-queue
2. Calls Embedded Agent API
3. Applies routing rules **in code** (not in prompt)
4. Stores final classification to **PostgreSQL** (`classifications` table)

**8. PostgreSQL Database — Three Tables**

| Table | Columns |
|---|---|
| `jobs` | `job_id`, `status`, `created_at`, `completed_at` |
| `ocr_results` | `job_id`, `page`, `raw_text`, `sentences_json`, `confidence_scores`, `metadata_json` |
| `classifications` | `job_id`, `sentence_id`, `text`, `final_label`, `agent_path`, `embedded_sentence`, `original_flag` |

**9. REST API Retrieval Endpoints**

- `GET /results/{job_id}` — returns status and all classifications
- `GET /results/{job_id}/sentences` — supports filtering by label, confidence, agent path
- `GET /health` — liveness check
- `GET /metrics` — Prometheus metrics endpoint

---

### Design Decisions

- **Two separate queues** (Upload Queue and Classification Queue) so OCR and classification fail independently — if the LLM API goes down, OCR jobs keep completing and queue depth accumulates safely until recovery.
- **Workers are stateless** — each worker pulls, processes, writes to DB, pushes to next queue. No local state. Horizontal scaling is trivial: add containers, point them at the queue.
- **Raw images stored in Object Storage**, never in the database — only the storage path is written to the DB. Keeps the database lean and avoids blob storage anti-patterns.
- **Job status always queryable via API** reading from the database — clients never poll a worker directly. Workers update job status in DB; the API reads it.
- **Target throughput: 500 images/hour** (≈8.3/minute, 1 every 7 seconds) — moderate load. Does not require Kafka or heavy distributed infrastructure. Redis Streams or SQS is sufficient.

---

## Deliverable 2 — Two Most Likely Failure Points

### Failure Point 1 — LLM API Rate Limiting or Outage (Classification Layer)

**Why most likely:** 500 images/hour at 5 sentences per image = 2,500+ classification API calls/hour minimum. LLM APIs have documented rate limits and periodic outages.

**Detection:**
- Every LLM API call logs HTTP status — Prometheus counter `classification_api_errors_total` labelled by error type (`429`, `500`, `timeout`)
- Alert fires if `classification_queue_depth` grows beyond 2× normal over 5 minutes
- Dead letter queue captures messages failing after 3 retries — alert fires immediately if DLQ is non-empty

**Recovery:**
- Exponential backoff with jitter on every API call (already implemented in the Classifier and Embedded Agent workers)
- **Circuit breaker:** if error rate exceeds 50% over 60 seconds, stop pulling from Classification Queue, wait 5 minutes, retry with a single probe request before resuming
- OCR results are already stored in the database before classification is attempted — **zero OCR work is lost** during an LLM outage
- When the API recovers, workers resume from the Classification Queue automatically — jobs complete late but correctly, without reprocessing

---

### Failure Point 2 — OCR Worker Memory Crash on Large PDFs

**Why second most likely:** The preprocessing pipeline loads full page images as NumPy arrays at 300 DPI. A 20-page PDF at 300 DPI is approximately 800 MB per worker. Multiple large PDFs hitting the queue simultaneously causes out-of-memory crashes.

**Detection:**
- Each worker sends a heartbeat to Redis every 30 seconds — if heartbeat stops, the orchestrator marks the worker dead and triggers a replacement
- Prometheus gauge `ocr_worker_memory_mb` — alert fires if any worker exceeds 80% of its container memory limit
- **Message visibility timeout:** if a worker pulls a message and does not acknowledge within 5 minutes, the queue makes it visible again for another worker to pick up
- Jobs stuck in `processing` status for more than 10 minutes trigger a stale-job alert

**Recovery:**
- Process pages one at a time — never load the entire PDF into memory. Process one page, write result to DB, free memory, process next page.
- Container memory limit enforced at deployment level (2 GB per OCR worker container)
- If a worker crashes mid-job, the message re-appears in the queue and another worker picks it up — no manual intervention required
- Files over 50 pages are split into chunks at upload time and processed as separate jobs
- After 3 failed attempts, message moves to dead letter queue and triggers a manual inspection alert

---

## Deliverable 3 — Real-time OCR Quality Instrumentation for Verbatim Transcription

### Signal 1 — Confidence Score Distribution Monitoring

Track the rolling distribution of per-word OCR confidence scores across all jobs as a **Prometheus histogram** (`ocr_word_confidence_histogram`). Alert if the mean confidence drops more than 15% compared to the 7-day rolling baseline. Catches scanner quality degradation, new image formats entering the system, and preprocessing regressions before they accumulate into systemic errors.

### Signal 2 — Non-Standard Form Preservation Rate (NSFPR)

Maintain a **reference lexicon** of known non-standard forms from the document domain (child handwriting misspellings, invented words, non-standard constructions). After each OCR run, compute:

```
NSFPR = (non-standard forms preserved verbatim) / (non-standard forms present in ground truth)
```

Alert if NSFPR drops below **0.90**. This directly measures silent normalisation — the core fidelity risk that CER and WER cannot detect. A system that auto-corrects "siad" to "said" scores well on CER but fails this metric.

### Signal 3 — Out-of-Vocabulary Rate

Track the percentage of OCR output words that are **out-of-vocabulary** relative to standard English, as a rolling per-job metric. A healthy OOV rate is expected for verbatim transcription — it confirms non-standard forms are being preserved. Two alert conditions:

- **OOV rate drops suddenly toward zero** — normalisation is occurring; non-standard forms are being silently corrected
- **OOV rate spikes suddenly upward** — OCR engine is producing noise or garbled output

Both directions indicate a problem. Neither is safe to ignore.

### Signal 4 — Low-Confidence Region Trend

Track the percentage of sentences per job that contain at least one low-confidence word, as a **rolling 24-hour metric**. Alert conditions:

- **Sudden increase** — image quality has changed (different scanner, different lighting, different document format)
- **Sudden decrease** — confidence thresholds may be misconfigured, or the OCR engine is producing overconfident wrong output

### CI/CD Integration

Every commit to `preprocessing.py` triggers a pipeline that runs the preprocessing changes against a set of **reference test pages with known ground truth**. The pipeline computes CER, WER, and NSFPR. The **build fails** if:

- NSFPR drops more than 5% from baseline, **or**
- CER increases more than 10% from baseline

This prevents preprocessing improvements from accidentally normalising non-standard text. Quality improvements must not come at the cost of verbatim fidelity.
