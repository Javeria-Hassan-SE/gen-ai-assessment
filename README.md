# GenAI Engineer Assessment — Aegasis Lab

This repo contains my submission for the GenAI Engineer technical assessment. There are four sections, each in its own folder. Below I've explained what each section does, the approach I took, how to run everything, and honest notes on what works and what doesn't out of the box.

---

## Section 1 — Handwriting OCR Pipeline

**What it is:** A full pipeline for reading scanned handwritten student work — preprocessing the image, extracting text with OCR, and evaluating how accurately the text was read.

**Approach:**

- **Preprocessing (Task 1.1):** The image goes through a fixed sequence of steps: convert to grayscale → deskew (straighten tilted scans) → CLAHE contrast boost → median blur to remove noise → adaptive binarization → remove horizontal ruled lines → remove vertical margin line → reconnect broken strokes. The key decision was to run CLAHE *before* the median blur so faint pencil strokes get their contrast boosted before noise suppression can smooth them away.
- **OCR and segmentation (Task 1.2):** EasyOCR reads the preprocessed image word by word, each word comes with a confidence score. Words below the threshold are flagged as low-confidence. Words are then sorted into reading order by Y and X coordinates. The header region (top 20%) is treated as metadata (student name, date, class) and excluded from the body text. Sentences are split using spaCy's rule-based sentencizer — no LLM.
- **Evaluation (Task 1.3):** Four metrics are computed against ground truth. CER (character error rate) and WER (word error rate) measure raw accuracy. Sentence F1 measures how many sentences were recovered. NSFPR (Non-Standard Form Preservation Rate) is the extra metric — it specifically checks whether intentional child misspellings like `siad` or `scord` were preserved verbatim, which standard CER misses entirely.

**Two OCR engines are available: EasyOCR and TrOCR.**

### Why EasyOCR doesn't always work well on handwriting

EasyOCR was originally designed for scene text — signs, labels, printed documents. It does work on handwriting but its confidence scores on child writing tend to be all over the place. The bigger issue is that there's a known compatibility problem with `torchvision`: when EasyOCR initializes its CRAFT text detector, it needs `torchvision::nms` to be registered. If torchvision is only partially initialized at that point (which happens easily in a Streamlit app where imports run in different orders), you get the error `operator torchvision::nms does not exist`. The fix is to `import torchvision` at the very top of `app.py` before anything else — this forces torchvision's C extension to load fully before EasyOCR touches it. That's the reason for that slightly odd import at the top of `section_1/app.py`.

### Why TrOCR isn't easy to test

TrOCR (`microsoft/trocr-base-handwritten`) is actually much better suited to this task. It was specifically fine-tuned on the IAM handwriting dataset so it understands cursive and messy script much better than EasyOCR. However, it requires downloading about 400 MB of model weights from Hugging Face the first time you use it. That download needs an internet connection and takes a couple of minutes. On top of that, it needs `transformers`, `torch`, and `torchvision` all installed. If any of those are missing you'll get an ImportError. The pipeline handles this gracefully — if the download hasn't happened yet you'll see a warning in the sidebar, and there's a "Pre-load model" button to trigger the download manually before running OCR.

**Streamlit app:**
```
cd section_1
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # only needed once
streamlit run app.py
```

**Run tests:**
```
cd section_1
pytest tests/
```

**Note on Poppler (Windows):** The PDF-to-image conversion uses Poppler. Download the Windows build from [https://github.com/oschwartz10612/poppler-windows/releases](https://github.com/oschwartz10612/poppler-windows/releases) and extract it into a `poppler/` folder at the project root. The code will find it automatically — no PATH changes needed.

---

## Section 2 — Sentence Classification Pipeline

**What it is:** A two-agent LLM pipeline that classifies sentences from student writing into five grammatical types: Simple, Compound, Complex, Compound-Complex, or Incomplete.

**Approach:**

- **Two agents, not one:** The Classifier Agent gets all sentences in one API call and labels each one. If any sentence comes back as `Incomplete`, the Embedded Sentence Agent runs on that sentence specifically — its job is to look inside the fragment and find a complete clause that was hiding in there. The routing between agents is a Python `if` statement, not a prompt instruction. The model can't decide to call a different agent; the code decides.
- **Hard constraint respected:** The classifier must never "fix" the sentence. `becaus I go Home and` gets classified as-written, not cleaned up. This is enforced by the system prompt.
- **Dual provider support:** You can use either Anthropic Claude or OpenAI GPT. The frontend has a radio button to switch between them. Tool-use (structured output) is used for both — Anthropic uses `input_schema`, OpenAI uses the `function` wrapper format. Both force the model to use the tool rather than replying in free text.
- **Rate limit handling:** Exponential back-off with up to 4 retries. If a sentence is missing from the classifier's response, the pipeline retries it individually rather than failing.

**Streamlit app:**
```
cd section_2
pip install -r requirements.txt
streamlit run app.py
```
Enter your OpenAI or Anthropic API key in the sidebar. No key stored anywhere — you paste it fresh each session (or set it as an environment variable before launching).

**Run tests** (uses mocked API — no real key needed):
```
cd section_2
pytest tests/
```

---

## Section 3 — Architecture & Design Documents

Three markdown documents in `section_3/`. No code to run.

- **`architecture.md`** — Full system design for the production handwriting OCR + classification pipeline. Covers the REST API layer, upload queue, OCR workers, classification queue, database schema, instrumentation signals (confidence histogram, NSFPR drift, OOV rate), and the two most likely failure modes in production.
- **`deprecation.md`** — A 12-week migration plan for replacing an old single-agent classifier with the new two-agent pipeline. Uses a shadow-then-canary rollout strategy with a feature flag so the old system can be rolled back instantly if anything breaks. Includes 12 minimum test cases to pass before the migration is considered complete.
- **`fidelity_tradeoff.md`** — An essay on why standard CER and WER are not enough for a verbatim transcription system, and how NSFPR closes the gap. Explains the three safeguards needed to stop an LLM from silently correcting intentional misspellings, and what the CI/CD gate should look like.

---

## Section 4 — Bonus: Source-Fidelity Score

**What it is:** A new evaluation metric (SFS) and a design document for a hybrid OCR fallback strategy.

**Approach:**

- **`sfs_metric.py` — Source-Fidelity Score (Task 4.1):** SFS weights edit types differently. A "normalisation edit" is when the OCR engine silently replaces a non-standard word (`siad`) with the correct English word (`said`). That gets a penalty of 1.0 because it's invisible — the output looks clean but verbatim fidelity is gone. A "noise edit" is when the engine produces garbled output (`g2[`). That gets a penalty of 0.5 because at least it's detectable. The formula is `SFS = 1 - (norm_penalty + noise_penalty) / len(gt_tokens)`, clamped to [0, 1]. Higher is better. The write-up (`sfs_writeup.md`) walks through two worked examples showing exactly where CER and SFS give opposite signals.
- **`hybrid_ocr.md` — Hybrid OCR Fallback (Task 4.2):** Instead of running the LLM on the whole page, only low-confidence regions get sent to a vision LLM. The confidence threshold is calibrated per document type using a 50-image holdout set. Four guardrails prevent the LLM from correcting misspellings: an explicit prompt instruction, a post-LLM NSFPR audit, token-level anchoring against the traditional OCR output, and architectural routing in the orchestrator so the LLM physically can't see high-confidence regions.

**Run the SFS demo:**
```
cd section_4
pip install editdistance
python sfs_metric.py
```

---

## How to run the full repo

Each section is independent. Start with whichever section you want to test.

```
# Clone
git clone https://github.com/Javeria-Hassan-SE/gen-ai-assessment.git
cd gen-ai-assessment

# Section 1 — OCR pipeline
cd section_1
pip install -r requirements.txt
streamlit run app.py

# Section 2 — Sentence classifier
cd ../section_2
pip install -r requirements.txt
streamlit run app.py

# Section 4 — SFS metric
cd ../section_4
pip install editdistance
python sfs_metric.py
```

**Python version:** 3.8+ required (spaCy 3.6 and numpy 1.24 are pinned for Python 3.8 compatibility in section 1).

**GPU:** Everything runs on CPU. If you have CUDA, check "Use GPU" in the section 1 sidebar — it speeds up EasyOCR detection noticeably.

**API keys:** Section 2 needs either an Anthropic or OpenAI key. Paste it in the sidebar, or set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` as an environment variable before launching Streamlit.

---

## Project structure

```
.
├── section_1/          OCR pipeline (preprocessing, EasyOCR, evaluation, Streamlit app)
│   ├── app.py
│   ├── preprocessing.py
│   ├── ocr_pipeline.py
│   ├── evaluation.py
│   ├── config.py
│   ├── utils.py
│   ├── eval.py
│   ├── ground_truth.json
│   ├── requirements.txt
│   └── tests/
├── section_2/          Two-agent sentence classifier
│   ├── app.py
│   ├── classifier.py
│   ├── prompts.py
│   ├── error_analysis.md
│   ├── requirements.txt
│   └── tests/
├── section_3/          Architecture & design documents
│   ├── architecture.md
│   ├── deprecation.md
│   └── fidelity_tradeoff.md
└── section_4/          SFS metric + hybrid OCR design
    ├── sfs_metric.py
    ├── sfs_writeup.md
    ├── hybrid_ocr.md
    └── requirements.txt
```
