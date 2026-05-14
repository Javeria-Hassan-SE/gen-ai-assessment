"""Section 1 — Handwriting OCR Pipeline · Streamlit Dashboard
Black & Shocking-Pink theme · Tasks 1.1 / 1.2 / 1.3
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# torchvision must be fully initialized before EasyOCR imports it.
# Importing it here loads torchvision._C which registers torchvision::nms.
try:
    import torchvision          # noqa: F401
except ImportError:
    pass

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from evaluation import evaluate_all
from ocr_pipeline import build_reader, metadata_y_threshold, process_page, OCR_ENGINES, _ENGINE_HF_MODEL
from preprocessing import _preprocess_array, to_grayscale, compute_deskew_angle
from utils import (
    compute_blur_score,
    default_pdf_path,
    detect_duplicate_pages,
    draw_ocr_boxes,
    is_too_blurry,
    pdf_to_images,
)

st.set_page_config(
    page_title="Section 1 — Handwriting OCR Pipeline",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
/* ---- global ---- */
html, body, [class*="css"] { background-color:#0a0a0a; color:#ffffff; }
.stApp { background-color:#0a0a0a; }
[data-testid="stSidebar"] { background-color:#111111; border-right:2px solid #FF1493; }
/* ---- headings ---- */
h1,h2,h3,h4 { color:#FF1493 !important; }
/* ---- tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    background:#1a1a1a; border-radius:10px; gap:4px; padding:6px;
}
.stTabs [data-baseweb="tab"] {
    background:#222; color:#aaa; border-radius:8px; padding:10px 20px;
    font-weight:700; font-size:0.95rem;
}
.stTabs [aria-selected="true"] {
    background:#FF1493 !important; color:#fff !important;
}
/* ---- buttons ---- */
.stButton>button {
    background:#FF1493; color:#fff; border:none; border-radius:8px;
    font-weight:700; padding:8px 20px;
}
.stButton>button:hover { background:#cc0077; }
/* ---- metric boxes ---- */
[data-testid="metric-container"] {
    background:#1a1a1a; border:1px solid #FF1493;
    border-radius:10px; padding:14px;
}
[data-testid="stMetricValue"] { color:#FF1493 !important; font-size:1.8rem; }
/* ---- expanders ---- */
.streamlit-expanderHeader {
    background:#1a1a1a !important; color:#FF1493 !important;
    border:1px solid #333; border-radius:6px;
}
/* ---- dataframes ---- */
.dataframe { background:#1a1a1a; color:#fff; }
thead th { background:#FF1493 !important; color:#fff !important; }
/* ---- file uploader ---- */
[data-testid="stFileUploadDropzone"] {
    background:#1a1a1a; border:2px dashed #FF1493; border-radius:10px;
}
/* ---- scrollbar ---- */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:#111; }
::-webkit-scrollbar-thumb { background:#FF1493; border-radius:4px; }
/* ---- custom cards ---- */
.card {
    background:#1a1a1a; border:1px solid #FF1493;
    border-radius:12px; padding:18px; margin:8px 0;
}
.step-label {
    color:#FF1493; font-size:0.8rem; font-weight:700;
    text-align:center; margin-bottom:4px;
}
.step-desc {
    color:#888; font-size:0.72rem; text-align:center; margin-top:2px;
}
.badge-pink {
    background:#FF1493; color:#fff; padding:2px 8px;
    border-radius:12px; font-size:0.75rem; font-weight:600;
}
.badge-green {
    background:#1a7a4a; color:#fff; padding:2px 8px;
    border-radius:12px; font-size:0.75rem; font-weight:600;
}
.badge-gold {
    background:#b8860b; color:#fff; padding:2px 8px;
    border-radius:12px; font-size:0.75rem; font-weight:600;
}
.composite-score {
    font-size:3.5rem; font-weight:900; color:#FF1493;
    text-align:center; line-height:1;
}
.divider { border-top:1px solid #333; margin:16px 0; }
</style>
""",
    unsafe_allow_html=True,
)

def _init_state():
    defaults = {
        "pdf_images": [],
        "duplicate_indices": [],
        "preprocessed": {},   # page_idx -> (final, steps_dict)
        "ocr_results": {},    # page_idx -> result dict
        "gt_data": [],
        "ocr_engine": "easyocr",
        "use_gpu": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

@st.cache_resource(show_spinner="Loading OCR model (first run only)…")
def get_ocr_reader(engine: str = "easyocr", gpu: bool = False):
    return build_reader(engine=engine, gpu=gpu)


def _load_ground_truth() -> list[dict]:
    gt_path = Path(__file__).parent / "ground_truth.json"
    if gt_path.exists():
        return json.loads(gt_path.read_text(encoding="utf-8"))
    return []


def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _is_hf_model_cached(model_id: str) -> bool:
    """Return True if the HuggingFace model is already in the local disk cache."""
    try:
        from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
        result = try_to_load_from_cache(model_id, "config.json")
        return result is not None and result is not _CACHED_NO_EXIST
    except Exception:
        # huggingface_hub not available or unexpected API — fall back to path check
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        slug = "models--" + model_id.replace("/", "--")
        return (cache_dir / slug).exists()


def _img_to_pil(arr: np.ndarray) -> Image.Image:
    if len(arr.shape) == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr)


def _section_header(title: str, subtitle: str = ""):
    st.markdown(f"<h2 style='margin-bottom:2px'>{title}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p style='color:#888;margin-top:0'>{subtitle}</p>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown(
            "<h1 style='color:#FF1493;font-size:1.3rem'>⚙ Pipeline Config</h1>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

        dpi_val = st.slider("PDF render DPI", 150, 400, cfg.DPI, 50,
                            help="Higher DPI = better quality but slower")
        conf_threshold = st.slider("OCR confidence threshold", 0.1, 0.9, cfg.CONFIDENCE_THRESHOLD, 0.05,
                                   help="Words below this are flagged as low-confidence")

        st.markdown("<p style='color:#FF1493;font-weight:700;margin-bottom:4px'>OCR Engine</p>", unsafe_allow_html=True)
        engine_key = st.selectbox(
            "OCR Engine",
            options=list(OCR_ENGINES.keys()),
            format_func=lambda k: OCR_ENGINES[k],
            index=list(OCR_ENGINES.keys()).index(st.session_state.get("ocr_engine", "easyocr")),
            label_visibility="collapsed",
        )
        if engine_key != st.session_state.get("ocr_engine"):
            st.session_state.ocr_results = {}   # clear stale results on engine switch
        st.session_state["ocr_engine"] = engine_key

        gpu_present = _gpu_available()
        gpu_hint = "✓ CUDA detected" if gpu_present else "CUDA not detected via torch (torch may not be installed yet)"
        use_gpu = st.checkbox(
            f"Use GPU (CUDA) — {gpu_hint}",
            value=st.session_state.get("use_gpu", False),
        )
        if use_gpu != st.session_state.get("use_gpu"):
            st.session_state.ocr_results = {}   # stale results if device changes
        st.session_state["use_gpu"] = use_gpu

        hf_model = _ENGINE_HF_MODEL.get(engine_key)
        if hf_model:
            cached = _is_hf_model_cached(hf_model)
            if cached:
                st.success(f"✓ Model cached locally — loads in seconds")
            else:
                st.warning(
                    f"First use downloads **{OCR_ENGINES[engine_key].split('~')[1].split(' ')[0] if '~' in OCR_ENGINES[engine_key] else '?'}**  \n"
                    f"Saved to `~/.cache/huggingface/` — downloaded only once."
                )
            if st.button("⬇ Pre-load model now", use_container_width=True, key="preload_btn"):
                with st.spinner("Downloading / loading model — please wait…"):
                    try:
                        get_ocr_reader(engine_key, use_gpu)
                        st.success("Model ready!")
                    except ImportError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Load failed: {e}")

        st.markdown("---")

        st.markdown("<p style='color:#FF1493;font-weight:700'>Load PDF</p>", unsafe_allow_html=True)
        auto_path = default_pdf_path()
        if auto_path:
            st.success(f"✓ Found Testing.pdf")
            use_default = st.checkbox("Use Testing.pdf", value=True)
        else:
            use_default = False

        uploaded = None
        if not use_default or not auto_path:
            uploaded = st.file_uploader("Upload PDF", type=["pdf"])

        load_btn = st.button("Load PDF & Preprocess", use_container_width=True)

        if load_btn:
            _do_load(auto_path if use_default and auto_path else None,
                     uploaded, dpi_val, conf_threshold)

        if st.session_state.pdf_images:
            n = len(st.session_state.pdf_images)
            dups = st.session_state.duplicate_indices
            st.markdown(
                f"<div class='card'>"
                f"<b style='color:#FF1493'>Pages loaded:</b> {n}<br>"
                f"<b style='color:#FF1493'>Duplicates:</b> {len(dups)} "
                f"(pages {[i+1 for i in dups]})</div>",
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown(
            "<p style='color:#555;font-size:0.75rem'>"
            "Section 1<br>"
            "EasyOCR + spaCy sentencizer (no LLM)</p>",
            unsafe_allow_html=True,
        )
    return conf_threshold


def _do_load(pdf_path, uploaded_file, dpi, conf_thresh):
    cfg.CONFIDENCE_THRESHOLD = conf_thresh  # update global for this run
    with st.spinner("Converting PDF → images…"):
        try:
            if pdf_path:
                images = pdf_to_images(pdf_path=pdf_path, dpi=dpi)
            elif uploaded_file:
                images = pdf_to_images(pdf_bytes=uploaded_file.read(), dpi=dpi)
            else:
                st.error("No PDF source provided.")
                return
        except Exception as exc:
            st.error(f"PDF loading failed: {exc}")
            return

    st.session_state.pdf_images = images
    st.session_state.duplicate_indices = detect_duplicate_pages(images)
    st.session_state.preprocessed = {}
    st.session_state.ocr_results = {}

    with st.spinner("Preprocessing all pages…"):
        for idx, img in enumerate(images):
            gray = to_grayscale(img, "RGB")
            angle = compute_deskew_angle(gray)
            blur = compute_blur_score(gray)
            final, ocr_input, steps = _preprocess_array(img, "RGB")
            st.session_state.preprocessed[idx] = {
                "final": final,
                "ocr_input": ocr_input,
                "steps": steps,
                "deskew_angle": angle,
                "blur_score": blur,
            }


def render_tab_preprocessing():
    _section_header(
        "Task 1.1 — Image Preprocessing Pipeline",
        "Fixed sequential pipeline: Grayscale → Deskew → Denoise → CLAHE → Binarize → Line removal → Stroke reconnect",
    )

    if not st.session_state.pdf_images:
        st.info("Load a PDF from the sidebar to begin.")
        _render_pipeline_explainer()
        return

    images = st.session_state.pdf_images
    page_labels = [
        f"Page {i+1}" + (" ⚠ DUPLICATE" if i in st.session_state.duplicate_indices else "")
        for i in range(len(images))
    ]
    sel = st.radio("Select page to inspect", page_labels, horizontal=True)
    page_idx = page_labels.index(sel)

    pre = st.session_state.preprocessed.get(page_idx)
    if not pre:
        st.warning("Preprocessing data not found. Reload the PDF.")
        return

    steps: dict = pre["steps"]
    angle: float = pre["deskew_angle"]
    blur: float = pre["blur_score"]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Detected Skew", f"{angle:+.2f}°")
    col_b.metric("Blur Score (Laplacian σ²)", f"{blur:.1f}",
                 delta="OK" if blur > cfg.BLUR_LAPLACIAN_THRESHOLD else "BLURRY",
                 delta_color="normal" if blur > cfg.BLUR_LAPLACIAN_THRESHOLD else "inverse")
    col_c.metric("Duplicate page?",
                 "YES ⚠" if page_idx in st.session_state.duplicate_indices else "No")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    step_items = list(steps.items())
    cols_per_row = 4
    for row_start in range(0, len(step_items), cols_per_row):
        row_steps = step_items[row_start: row_start + cols_per_row]
        cols = st.columns(len(row_steps))
        for col, (label, (img_arr, desc)) in zip(cols, row_steps):
            with col:
                st.markdown(f"<div class='step-label'>{label}</div>", unsafe_allow_html=True)
                st.image(_img_to_pil(img_arr), use_container_width=True)
                st.markdown(f"<div class='step-desc'>{desc}</div>", unsafe_allow_html=True)

    with st.expander("Why these pipeline steps? (design rationale)"):
        st.markdown(
            """
**Stroke normalisation excluded intentionally.**
Thinning strokes to a uniform width (common in printed-digit datasets) hurts natural child
handwriting — the variable stroke width *is* part of the OCR signal. Aggressive thinning
disconnects letters and merges adjacent characters.

**Vertical margin line removal** (Step 7) is an addition to the base approach.
The red vertical margin line visible on page 3 passes through the binarization stage intact
and appears as a long vertical black stripe that confuses EasyOCR's line-detection. The
vertical morphological opening only examines the left 20 % of the image so it never touches
any actual text character.

**Duplicate detection** uses perceptual hash (pixel-level correlation on 32×32 thumbnail).
Pages 1 and 3 of the test PDF are the same story photographed from different angles —
flagged here so downstream tasks can skip redundant processing.
            """
        )


def _render_pipeline_explainer():
    """Show a static explainer card when no PDF is loaded."""
    st.markdown(
        """
<div class='card'>
<b style='color:#FF1493'>Pipeline steps (in order):</b>
<ol style='color:#ccc;line-height:2'>
<li><b>Grayscale</b> — removes color channels irrelevant to handwriting</li>
<li><b>Deskew</b> — projection profile analysis finds and corrects page tilt</li>
<li><b>CLAHE</b> — local contrast enhancement recovers faint pencil strokes</li>
<li><b>Noise removal</b> — median blur preserves edge sharpness</li>
<li><b>Binarize</b> — adaptive Gaussian threshold handles uneven illumination</li>
<li><b>Horizontal line removal</b> — morphological opening erases ruled lines</li>
<li><b>Vertical margin removal</b> — removes red notebook margin line (left 20 % only)</li>
<li><b>Reconnect strokes</b> — morphological closing bridges binarization gaps</li>
</ol>
</div>""",
        unsafe_allow_html=True,
    )


def render_tab_ocr():
    _section_header(
        "Task 1.2 — OCR Extraction & Sentence Segmentation",
        "EasyOCR (CNN+LSTM) · per-word confidence · spaCy rule-based sentencizer · metadata excluded",
    )

    if not st.session_state.pdf_images:
        st.info("Load a PDF from the sidebar to begin.")
        return

    images = st.session_state.pdf_images
    page_labels = [
        f"Page {i+1}" + (" ⚠ DUP" if i in st.session_state.duplicate_indices else "")
        for i in range(len(images))
    ]
    sel = st.radio("Select page", page_labels, horizontal=True, key="ocr_page_radio")
    page_idx = page_labels.index(sel)

    run_btn = st.button(f"▶ Run OCR on Page {page_idx+1}", use_container_width=False)

    if run_btn:
        pre = st.session_state.preprocessed.get(page_idx)
        if not pre:
            st.error("Preprocess data missing — reload the PDF.")
            return
        engine = st.session_state.get("ocr_engine", "easyocr")
        use_gpu = st.session_state.get("use_gpu", False)
        try:
            reader = get_ocr_reader(engine, use_gpu)
        except ImportError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"Failed to load OCR engine: {e}")
            return

        gt_data = st.session_state.gt_data or _load_ground_truth()
        sample_id = gt_data[page_idx]["sample_id"] if page_idx < len(gt_data) else f"page{page_idx+1}"
        with st.spinner("Running EasyOCR…"):
            result = process_page(
                raw_image=images[page_idx],
                ocr_input=pre["ocr_input"],
                reader=reader,
                sample_id=sample_id,
                page_num=page_idx,
                is_duplicate=page_idx in st.session_state.duplicate_indices,
                blur_score=pre["blur_score"],
                deskew_angle=pre["deskew_angle"],
            )
        st.session_state.ocr_results[page_idx] = result
        st.success("OCR complete!")

    result = st.session_state.ocr_results.get(page_idx)
    if not result:
        st.caption("Click 'Run OCR' above to process this page.")
        return

    _render_ocr_output(images[page_idx], result)


def _render_ocr_output(raw_image: np.ndarray, result: dict):
    # All visualization data lives under _debug; public fields match reference spec
    dbg = result["_debug"]
    meta = dbg["page_meta"]
    stats = dbg["stats"]
    metadata_words = dbg["metadata_words"]
    body_words = dbg["body_words"]
    full_text = dbg["full_text"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Words", stats["total_words"])
    c2.metric("Low-Confidence Words", stats["low_confidence_words"])
    c3.metric("Sentences", stats["total_sentences"])
    c4.metric("Flagged Sentences", stats["flagged_sentences"])

    # Page-level flag
    if result["page_low_confidence"]:
        st.markdown("<span class='badge-pink'>PAGE LOW CONFIDENCE</span>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.markdown("<h4>Annotated Image</h4>", unsafe_allow_html=True)
        annotated = draw_ocr_boxes(
            raw_image, dbg["raw_ocr"], meta["metadata_y_threshold"]
        )
        st.image(annotated, use_container_width=True)
        st.markdown(
            "<div style='font-size:0.8rem;color:#888'>"
            "<span class='badge-green'>■</span> High confidence &nbsp;&nbsp;"
            "<span class='badge-pink'>■</span> Low confidence &nbsp;&nbsp;"
            "<span class='badge-gold'>■</span> Metadata (excluded)"
            "</div>",
            unsafe_allow_html=True,
        )

        meta_fields = result["metadata_excluded"]
        with st.expander(f"metadata_excluded — {len(metadata_words)} words filtered"):
            if meta_fields:
                for k, v in meta_fields.items():
                    st.markdown(
                        f"<div style='font-size:0.85rem'>"
                        f"<b style='color:#FF1493'>{k}:</b> {v}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.caption("No metadata header detected on this page.")

    with right_col:
        st.markdown("<h4>Sentences</h4>", unsafe_allow_html=True)
        for sent in result["sentences"]:
            flag_html = (
                "<span class='badge-pink'>LOW CONF</span>"
                if sent["low_confidence"]
                else "<span class='badge-green'>OK</span>"
            )
            lc_words_html = ""
            if sent["low_confidence_words"]:
                lc_words_html = (
                    "<br><small style='color:#FF1493'>Uncertain: "
                    + ", ".join(f"<i>{w}</i>" for w in sent["low_confidence_words"])
                    + "</small>"
                )
            st.markdown(
                f"<div class='card' style='margin:6px 0'>"
                f"<b style='color:#888'>id:{sent['id']}</b> {flag_html}<br>"
                f"<span style='font-size:0.9rem'>{sent['text']}</span>"
                f"{lc_words_html}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<h4>Full Extracted Body Text</h4>", unsafe_allow_html=True)
    st.code(full_text, language=None)

    with st.expander("Reference JSON output (process_page() structure)"):
        display_result = {
            "sample_id": result["sample_id"],
            "metadata_excluded": result["metadata_excluded"],
            "sentences": result["sentences"],
            "page_low_confidence": result["page_low_confidence"],
        }
        st.json(display_result)

    with st.expander("Word-level OCR table (all body words)"):
        if body_words:
            word_df = pd.DataFrame(
                [
                    {
                        "word": w["text"],
                        "confidence": w["confidence"],
                        "low_conf": w["low_confidence"],
                        "x1": w["x1"],
                        "y1": w["y1"],
                    }
                    for w in body_words
                ]
            )
            st.dataframe(
                word_df.style.apply(
                    lambda col: [
                        "color: #FF1493" if v else "" for v in word_df["low_conf"]
                    ],
                    subset=["word"],
                ),
                use_container_width=True,
                hide_index=True,
            )

    if st.button("→ Copy this output to Evaluation tab"):
        page_idx = meta["page_num"]
        gt_data = st.session_state.gt_data
        if gt_data and page_idx < len(gt_data):
            gt_data[page_idx]["predicted"] = full_text
            st.session_state.gt_data = gt_data
            st.success(f"Copied predicted text for page {page_idx+1} → Evaluation tab.")
        else:
            st.info("Ground truth not loaded yet — check the Evaluation tab.")


def render_tab_evaluation():
    _section_header(
        "Task 1.3 — Evaluation Metrics",
        "CER · WER · Sentence F1 · Composite Score · NSFPR (Non-Standard Form Preservation)",
    )

    if not st.session_state.gt_data:
        st.session_state.gt_data = _load_ground_truth()

    gt_data = st.session_state.gt_data

    st.markdown("<h4>Ground Truth & Predictions</h4>", unsafe_allow_html=True)
    st.caption("Predicted text auto-fills when you click '→ Copy to Evaluation' in the OCR tab, or paste manually below.")

    edited_samples = []
    for i, sample in enumerate(gt_data):
        with st.expander(f"Sample {i+1}: {sample.get('description', sample['sample_id'])}"):
            col_gt, col_pred = st.columns(2)
            with col_gt:
                st.markdown("<b style='color:#FF1493'>Ground Truth</b>", unsafe_allow_html=True)
                gt_text = st.text_area("", value=sample.get("ground_truth", ""),
                                       height=150, key=f"gt_{i}", label_visibility="collapsed")
            with col_pred:
                st.markdown("<b style='color:#FF1493'>Predicted (OCR output)</b>", unsafe_allow_html=True)
                pred_text = st.text_area("", value=sample.get("predicted", ""),
                                         height=150, key=f"pred_{i}", label_visibility="collapsed")

            edited_samples.append({
                **sample,
                "ground_truth": gt_text,
                "predicted": pred_text,
            })

    st.session_state.gt_data = edited_samples

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    run_eval = st.button("▶ Run Evaluation", use_container_width=False)

    if run_eval:
        active = [s for s in edited_samples if s.get("predicted", "").strip()]
        if not active:
            st.warning("No predicted text found. Run OCR first or paste predictions manually.")
            return

        with st.spinner("Computing metrics…"):
            per_sample, aggregate = evaluate_all(active)

        _render_evaluation_results(per_sample, aggregate)


def _render_evaluation_results(per_sample: list[dict], aggregate: dict):
    composite = aggregate.get("composite_score", 0.0)
    st.markdown(
        f"<div class='card' style='text-align:center'>"
        f"<div style='color:#888;font-size:0.9rem;font-weight:700'>COMPOSITE SCORE</div>"
        f"<div class='composite-score'>{composite:.4f}</div>"
        f"<div style='color:#888;font-size:0.8rem'>"
        f"= (1−CER)×0.50 + (1−WER)×0.30 + SentF1×0.20"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    has_nsfpr = "nsfpr" in aggregate
    cols = st.columns(5 if has_nsfpr else 4)
    cols[0].metric("CER", f"{aggregate['cer']:.4f}", help="Character Error Rate — lower is better")
    cols[1].metric("WER", f"{aggregate['wer']:.4f}", help="Word Error Rate — lower is better")
    cols[2].metric("Sent-F1", f"{aggregate['sent_f1']:.4f}", help="Sentence F1 — higher is better")
    cols[3].metric("Sent-Precision / Recall",
                   f"{aggregate['sent_precision']:.2f} / {aggregate['sent_recall']:.2f}")
    if has_nsfpr:
        cols[4].metric("NSFPR", f"{aggregate['nsfpr']:.4f}",
                       help="Non-Standard Form Preservation Rate — fraction of intentional misspellings kept verbatim")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("<h4>Per-Sample Results</h4>", unsafe_allow_html=True)
    all_rows = per_sample + [aggregate]
    df = pd.DataFrame(all_rows)

    # Style: highlight aggregate row and colour composite score column
    def _style(row):
        if row["sample_id"] == "AGGREGATE (mean)":
            return ["background-color:#2a0a1a; color:#FF1493; font-weight:bold"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_style, axis=1).format(
        {k: "{:.4f}" for k in df.columns if k != "sample_id"}
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("<h4>Metric Comparison (per sample)</h4>", unsafe_allow_html=True)
    if _PLOTLY_AVAILABLE:
        fig = go.Figure()
        sample_ids = [r["sample_id"] for r in per_sample]
        colors = {"cer": "#FF1493", "wer": "#ff69b4", "sent_f1": "#00d4aa", "composite_score": "#ffffff"}
        for metric, color in colors.items():
            vals = [r.get(metric, 0) for r in per_sample]
            fig.add_trace(go.Bar(name=metric.upper(), x=sample_ids, y=vals, marker_color=color))

        fig.update_layout(
            barmode="group",
            plot_bgcolor="#0a0a0a",
            paper_bgcolor="#0a0a0a",
            font_color="#ffffff",
            legend=dict(bgcolor="#1a1a1a", bordercolor="#FF1493", borderwidth=1),
            xaxis=dict(gridcolor="#222"),
            yaxis=dict(gridcolor="#222", range=[0, 1]),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Install plotly for chart visualisation: pip install plotly")

    with st.expander("What is NSFPR and why does standard CER miss it?"):
        st.markdown(
            """
**Non-Standard Form Preservation Rate (NSFPR)**

Standard CER/WER measure edit distance against the ground truth — but they treat
*spelling corrections* as improvements.

Consider this ground-truth fragment: `he got scord he scrxmd AAA A!`

If the OCR pipeline (or a post-processing step) outputs `he got scared he screamed AAA A!`,
the CER is **lower** (fewer edits) yet the transcription is **less faithful** — it silently
normalised intentional child misspellings.

NSFPR catches this by:
1. Identifying all words in the ground truth that are not in a standard dictionary (pyspellchecker).
2. Checking what fraction of those words appear verbatim in the prediction.
3. A perfect verbatim transcription scores NSFPR = 1.0; a pipeline that auto-corrects scores < 1.0.

**Cross-reference:** This is the core problem described in Section 3.3 of the assessment.
            """
        )

    results_json = json.dumps({"per_sample": per_sample, "aggregate": aggregate}, indent=2)
    st.download_button(
        "⬇ Download results JSON",
        data=results_json,
        file_name="section1_evaluation_results.json",
        mime="application/json",
    )


def main():
    st.markdown(
        "<h1 style='text-align:center;margin-bottom:0'>Section 1 — Handwriting OCR Pipeline</h1>"
        "<p style='text-align:center;color:#888;margin-top:4px'>"
        "GenAI Engineer Assessment &nbsp;·&nbsp; Traditional CV + EasyOCR &nbsp;·&nbsp; No LLM"
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    render_sidebar()

    tab1, tab2, tab3 = st.tabs(
        ["1.1  Preprocessing", "1.2  OCR & Segmentation", "1.3  Evaluation Metrics"]
    )

    with tab1:
        render_tab_preprocessing()
    with tab2:
        render_tab_ocr()
    with tab3:
        render_tab_evaluation()


if __name__ == "__main__":
    main()
