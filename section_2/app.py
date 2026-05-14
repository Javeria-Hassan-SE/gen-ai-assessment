"""Section 2 — Streamlit UI for the two-agent sentence classification pipeline."""

from __future__ import annotations

import os
import sys
import textwrap
from collections import Counter

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from classifier import (
    classify_sentences,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_MODEL,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
)
from prompts import TRICKY_EXAMPLES, INCOMPLETE_PIPELINE_EXAMPLES

st.set_page_config(
    page_title="Sentence Classifier",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0d0d0d;
        color: #f0f0f0;
    }
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #ff2d78;
    }
    [data-testid="stSidebar"] * { color: #f0f0f0 !important; }

    h1, h2, h3, h4 { color: #ff2d78 !important; }

    .stButton > button {
        background-color: #ff2d78; color: #ffffff;
        border: none; border-radius: 6px;
        font-weight: 600; padding: 0.45rem 1.2rem;
        transition: background 0.2s;
    }
    .stButton > button:hover { background-color: #e0005e; color: #ffffff; }

    .stTextArea textarea, .stTextInput input {
        background-color: #1a1a1a !important;
        color: #f0f0f0 !important;
        border: 1px solid #ff2d78 !important;
        border-radius: 6px;
    }
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #1a1a1a !important;
        color: #f0f0f0 !important;
        border: 1px solid #ff2d78 !important;
    }
    .stRadio label { color: #f0f0f0 !important; }

    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1a;
        border-bottom: 2px solid #ff2d78;
    }
    .stTabs [data-baseweb="tab"] { color: #888; background-color: transparent; }
    .stTabs [aria-selected="true"] {
        color: #ff2d78 !important;
        border-bottom: 2px solid #ff2d78;
    }

    [data-testid="metric-container"] {
        background-color: #1a1a1a;
        border: 1px solid #ff2d78;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    [data-testid="stMetricValue"] { color: #ff2d78 !important; }

    .streamlit-expanderHeader {
        background-color: #1a1a1a !important;
        color: #ff2d78 !important;
        border: 1px solid #333;
        border-radius: 6px;
    }

    .badge-simple         { background:#1e4d2b; color:#5dde8e; padding:2px 8px; border-radius:4px; font-weight:600; }
    .badge-compound       { background:#1a3a5c; color:#5ab4f7; padding:2px 8px; border-radius:4px; font-weight:600; }
    .badge-complex        { background:#4a2a00; color:#f7a935; padding:2px 8px; border-radius:4px; font-weight:600; }
    .badge-compound-complex { background:#3a1a4a; color:#c07ef7; padding:2px 8px; border-radius:4px; font-weight:600; }
    .badge-incomplete     { background:#4a1a1a; color:#f75d5d; padding:2px 8px; border-radius:4px; font-weight:600; }

    .result-card {
        background-color: #1a1a1a;
        border: 1px solid #333;
        border-left: 4px solid #ff2d78;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
    }
    .result-card:hover { border-left-color: #ff80aa; }

    .provider-badge-anthropic {
        background: #1a2a4a; color: #7ab4f7;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.82em; font-weight: 600;
    }
    .provider-badge-openai {
        background: #1a3a1a; color: #5dde8e;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.82em; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


_BADGE_CLASS = {
    "Simple": "badge-simple",
    "Compound": "badge-compound",
    "Complex": "badge-complex",
    "Compound-Complex": "badge-compound-complex",
    "Incomplete": "badge-incomplete",
}


def badge(label: str) -> str:
    cls = _BADGE_CLASS.get(label, "badge-incomplete")
    return f'<span class="{cls}">{label}</span>'


def render_result_card(idx: int, r: dict):
    path_str = " → ".join(r["agent_path"])
    emb_html = ""
    if r.get("embedded_sentence"):
        emb_html = (
            f'<div style="margin-top:4px;font-size:0.82em;color:#aaa;">'
            f'Embedded: <em>"{r["embedded_sentence"]}"</em>'
            f' &nbsp;|&nbsp; Originally: {badge(r["original_flag"])}'
            f"</div>"
        )
    reasoning_html = ""
    if r.get("reasoning"):
        reasoning_html = (
            f'<div style="margin-top:4px;font-size:0.80em;color:#888;">'
            f'Reasoning: {r["reasoning"]}'
            f"</div>"
        )
    st.markdown(
        f"""
        <div class="result-card">
          <div style="font-size:0.78em;color:#888;margin-bottom:2px;">#{idx + 1}</div>
          <div style="font-size:0.97em;margin-bottom:6px;">
            <strong style="color:#f0f0f0;">{r['sentence']}</strong>
          </div>
          <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">
            {badge(r['classification'])}
            <span style="font-size:0.80em;color:#666;">via: {path_str}</span>
          </div>
          {emb_html}
          {reasoning_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_metrics(results: list[dict]):
    counts = Counter(r["classification"] for r in results)
    cols = st.columns(5)
    for col, label in zip(
        cols, ["Simple", "Compound", "Complex", "Compound-Complex", "Incomplete"]
    ):
        col.metric(label=label, value=counts.get(label, 0))


with st.sidebar:
    st.markdown("## ⚙️ Settings")

    provider = st.radio(
        "AI Provider",
        options=[PROVIDER_ANTHROPIC, PROVIDER_OPENAI],
        format_func=lambda p: "🟣 Anthropic (Claude)" if p == PROVIDER_ANTHROPIC else "🟢 OpenAI (GPT)",
        horizontal=True,
    )
    st.session_state["provider"] = provider

    st.markdown("---")

    if provider == PROVIDER_ANTHROPIC:
        st.markdown("**Anthropic API Key**")
        anthropic_key = st.text_input(
            label="anthropic_key",
            type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            placeholder="sk-ant-...",
            label_visibility="collapsed",
        )
        if anthropic_key:
            os.environ["ANTHROPIC_API_KEY"] = anthropic_key

        model_choice = st.selectbox(
            "Model",
            options=[
                "claude-haiku-4-5-20251001",
                "claude-sonnet-4-6",
                "claude-opus-4-7",
            ],
            index=0,
        )
        active_key = anthropic_key or os.environ.get("ANTHROPIC_API_KEY", "")

    else:  # OpenAI
        st.markdown("**OpenAI API Key**")
        openai_key = st.text_input(
            label="openai_key",
            type="password",
            value=os.environ.get("OPENAI_API_KEY", ""),
            placeholder="sk-...",
            label_visibility="collapsed",
        )
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key

        model_choice = st.selectbox(
            "Model",
            options=[
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-3.5-turbo",
            ],
            index=0,
        )
        active_key = openai_key or os.environ.get("OPENAI_API_KEY", "")

    if active_key:
        st.success("API key set ✓")
    else:
        st.warning("No API key — enter one above")

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        textwrap.dedent("""
        **Two-agent pipeline:**
        1. **Classifier Agent** — batch-classifies all sentences.
        2. **Embedded Sentence Agent** — runs only on *Incomplete* fragments to
           recover hidden complete clauses.

        Routing is enforced in Python, not prompts.
        """)
    )
    st.markdown(
        "<span style='color:#ff2d78;font-size:0.85em;'>Section 2</span>",
        unsafe_allow_html=True,
    )


def _provider_badge(p: str) -> str:
    if p == PROVIDER_OPENAI:
        return '<span class="provider-badge-openai">OpenAI</span>'
    return '<span class="provider-badge-anthropic">Anthropic</span>'


def _run_classify(sentences: list[str]) -> list[dict] | None:
    """Run classification with current sidebar settings, return results or None on error."""
    if not sentences:
        st.warning("Please enter at least one sentence.")
        return None
    if not active_key:
        st.error("No API key found. Enter it in the sidebar.")
        return None
    try:
        with st.spinner(f"Classifying with {model_choice}…"):
            return classify_sentences(
                sentences,
                model=model_choice,
                provider=provider,
                api_key=active_key or None,
            )
    except Exception as exc:
        st.error(f"Classification failed: {exc}")
        return None


st.markdown(
    "<h1 style='margin-bottom:0;'>🧠 Sentence Classifier</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<p style='color:#888;margin-top:4px;'>Two-agent pipeline · {_provider_badge(provider)} · {model_choice}</p>",
    unsafe_allow_html=True,
)

tab_classify, tab_examples, tab_pipeline = st.tabs([
    "Classify Sentences", "Tricky Examples", "Pipeline Examples"
])

with tab_classify:
    st.markdown("### Enter Sentences")
    st.markdown(
        "<p style='color:#888;font-size:0.88em;'>One sentence per line. "
        "Non-standard spelling is classified as-written.</p>",
        unsafe_allow_html=True,
    )

    default_text = "\n".join([
        "he ran home",
        "she wents to scool and he staid home",
        "becaus he was scord he runed home",
        "she wents to scool and he staid home becaus it was raing",
        "becaus I go Home and",
        "and then the big",
        "he wuz scord but he staid becaus his frend wuz there",
        "and when the lite came on",
    ])

    raw_input = st.text_area(
        label="sentences",
        value=default_text,
        height=220,
        label_visibility="collapsed",
    )

    col_run, col_clear, _ = st.columns([1, 1, 5])
    run_clicked = col_run.button("▶ Classify", use_container_width=True)
    if col_clear.button("✕ Clear", use_container_width=True):
        st.session_state.pop("classify_results", None)
        st.rerun()

    if run_clicked:
        sentences = [s.strip() for s in raw_input.splitlines() if s.strip()]
        results = _run_classify(sentences)
        if results is not None:
            st.session_state["classify_results"] = results

    if "classify_results" in st.session_state:
        results = st.session_state["classify_results"]
        st.markdown("---")
        st.markdown("### Results")
        render_summary_metrics(results)
        st.markdown("<br>", unsafe_allow_html=True)
        for i, r in enumerate(results):
            render_result_card(i, r)
        with st.expander("Raw JSON"):
            st.json(results)


with tab_examples:
    st.markdown("### Tricky Examples (Task 2.1)")
    st.markdown(
        "<p style='color:#888;font-size:0.88em;'>"
        "Five examples likely to cause mis-classification — one per category."
        "</p>",
        unsafe_allow_html=True,
    )

    for ex in TRICKY_EXAMPLES:
        with st.expander(
            f"{ex['category']} — {ex['sentence'][:60]}{'…' if len(ex['sentence']) > 60 else ''}"
        ):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Sentence:** `{ex['sentence']}`")
                st.markdown(f"**Why tricky:** {ex['why_tricky']}")
            with col2:
                st.markdown(
                    f"**Expected:**<br>{badge(ex['expected'])}",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    if st.button("▶ Run all tricky examples", key="run_tricky"):
        sentences = [ex["sentence"] for ex in TRICKY_EXAMPLES]
        expected = [ex["expected"] for ex in TRICKY_EXAMPLES]
        results = _run_classify(sentences)
        if results:
            correct = sum(
                1 for r, exp in zip(results, expected) if r["classification"] == exp
            )
            st.metric(
                f"Accuracy on tricky set ({_provider_badge(provider)} {model_choice})",
                f"{correct}/{len(results)}",
            )
            for r, exp in zip(results, expected):
                match = r["classification"] == exp
                icon = "✅" if match else "❌"
                note = f" *(expected {exp})*" if not match else ""
                st.markdown(
                    f"{icon} **{r['sentence'][:70]}** → "
                    f"{badge(r['classification'])}{note}",
                    unsafe_allow_html=True,
                )


with tab_pipeline:
    st.markdown("### Pipeline Examples (Task 2.1)")
    st.markdown(
        "<p style='color:#888;font-size:0.88em;'>"
        "Two Incomplete examples showing full agent interaction."
        "</p>",
        unsafe_allow_html=True,
    )

    for ex in INCOMPLETE_PIPELINE_EXAMPLES:
        st.markdown(f"#### {ex['label']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Input**")
            st.code(ex["input"])
            st.markdown(
                f"**Classifier output:** {badge(ex['classifier_output'])}",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown("**Embedded Agent output**")
            st.json(ex["embedded_agent_output"])
        with col3:
            st.markdown("**Final output**")
            st.json(ex["final_output"])
        st.markdown(f"*{ex['explanation']}*")
        st.markdown("---")

    if st.button("▶ Run pipeline examples live", key="run_pipeline"):
        sentences = [ex["input"] for ex in INCOMPLETE_PIPELINE_EXAMPLES]
        results = _run_classify(sentences)
        if results:
            st.markdown("### Live results")
            for i, r in enumerate(results):
                render_result_card(i, r)
