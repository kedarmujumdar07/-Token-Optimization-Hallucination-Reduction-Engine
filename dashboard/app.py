"""
dashboard/app.py
----------------
TokenGuard Streamlit dashboard — 3 pages:

  Page 1 — Try It Live
      Interactive prompt optimizer with side-by-side comparison,
      metrics cards, LLM response, and hallucination highlights.

  Page 2 — Analytics
      Plotly charts: tokens saved over time, cache hit pie,
      strategy bar chart, and cost savings metric cards.

  Page 3 — Benchmark
      Run predefined test cases, compare strategies, export CSV.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Path fix — allow imports from project root when run from dashboard/
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv
load_dotenv(dotenv_path=_ROOT / ".env")

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TokenGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark glassmorphism theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Background ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Metric cards ── */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99,102,241,0.25);
    }
    [data-testid="stMetricLabel"] { color: rgba(255,255,255,0.6) !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"] { color: #a78bfa !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"] { color: #34d399 !important; }

    /* ── Text areas & inputs ── */
    .stTextArea textarea, .stTextInput input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.3) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.25s ease !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(99,102,241,0.45) !important;
    }

    /* ── Expanders ── */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
        color: #c4b5fd !important;
        font-weight: 500 !important;
    }

    /* ── Hallucination flags ── */
    .hall-flag {
        background: rgba(239,68,68,0.15);
        border-left: 4px solid #ef4444;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        color: #fca5a5;
        font-size: 0.9rem;
    }
    .hall-neutral {
        background: rgba(234,179,8,0.12);
        border-left: 4px solid #eab308;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        color: #fde68a;
        font-size: 0.9rem;
    }
    .hall-ok {
        background: rgba(52,211,153,0.10);
        border-left: 4px solid #34d399;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        color: #6ee7b7;
        font-size: 0.9rem;
    }

    /* ── Section headers ── */
    .section-header {
        color: #a78bfa;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 24px 0 12px 0;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(167,139,250,0.3);
    }

    /* ── Info boxes ── */
    .stAlert { border-radius: 10px !important; }

    /* ── Tables ── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* ── Select box ── */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }

    /* ── Dividers ── */
    hr { border-color: rgba(255,255,255,0.08) !important; }

    /* ── Checkbox ── */
    .stCheckbox label { color: #c4b5fd !important; }

    /* ── Plotly chart background ── */
    .js-plotly-plot .plotly { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state — persists across reruns in the same browser session
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "request_log": [],        # list of TokenGuardResponse dicts
        "guard": None,            # TokenGuard instance
        "guard_error": None,      # error string if init failed
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ---------------------------------------------------------------------------
# TokenGuard lazy initialisation
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading TokenGuard models…")
def _load_guard():
    """Load TokenGuard once per Streamlit session (cached across reruns)."""
    try:
        from gateway.llm_client import TokenGuard
        guard = TokenGuard(
            token_budget=int(os.getenv("TOKEN_BUDGET", "4000")),
            cache_threshold=float(os.getenv("CACHE_THRESHOLD", "0.92")),
            keep_ratio=float(os.getenv("KEEP_RATIO", "0.70")),
            cache_persist_dir=str(_ROOT / "chroma_db"),
        )
        return guard, None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Sidebar — navigation + config
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 16px 0 8px 0;'>
            <span style='font-size:3rem;'>🛡️</span>
            <h2 style='color:#a78bfa; margin:4px 0 0 0; font-weight:700;'>TokenGuard</h2>
            <p style='color:rgba(255,255,255,0.4); font-size:0.8rem; margin:0;'>
                Token Optimization & Hallucination Middleware
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    page = st.radio(
        "Navigation",
        ["🚀 Try It Live", "📊 Analytics", "🧪 Benchmark"],
        label_visibility="collapsed",
    )

    st.divider()

    with st.expander("⚙️ Configuration"):
        api_provider = st.selectbox("LLM Provider", ["Anthropic", "OpenAI"])
        model_options = {
            "Anthropic": [
                "claude-sonnet-4-6",
                "claude-sonnet-4-5",
                "claude-haiku-3-5",
                "claude-opus-4-5",
            ],
            "OpenAI": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
        }
        selected_model = st.selectbox("Model", model_options[api_provider])
        max_tokens = st.slider("Max Response Tokens", 256, 2048, 1000, 64)
        cache_threshold = st.slider("Cache Similarity Threshold", 0.80, 0.99, 0.92, 0.01)
        keep_ratio = st.slider("Context Keep Ratio", 0.30, 1.00, 0.70, 0.05)

    st.divider()

    # Live cache stats
    guard, guard_err = _load_guard()
    if guard:
        cs = guard.cache_stats()
        col_a, col_b = st.columns(2)
        col_a.metric("Cached", cs["total_cached"])
        col_b.metric("Hit Rate", f"{cs['hit_rate']*100:.1f}%")

    st.markdown(
        "<p style='color:rgba(255,255,255,0.2);font-size:0.7rem;text-align:center;"
        "margin-top:20px;'>v1.0.0 · TokenGuard</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Helper — render hallucination flags
# ---------------------------------------------------------------------------
def _render_flags(flags: list[dict]) -> None:
    if not flags:
        st.success("✅ No hallucinations detected — response is fully grounded.")
        return

    for f in flags:
        label = f.get("label", "NEUTRAL")
        sentence = f.get("sentence", "")
        score = f.get("score", 0.0)
        css_class = {
            "CONTRADICTION": "hall-flag",
            "NEUTRAL": "hall-neutral",
            "ENTAILMENT": "hall-ok",
        }.get(label, "hall-neutral")
        icon = {"CONTRADICTION": "🚨", "NEUTRAL": "⚠️", "ENTAILMENT": "✅"}.get(label, "⚠️")
        st.markdown(
            f'<div class="{css_class}">'
            f'{icon} <strong>[{label}]</strong> (score: {score:.2f})<br/>'
            f'<span style="opacity:0.85">{sentence}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ============================================================================
# PAGE 1 — TRY IT LIVE
# ============================================================================
if page == "🚀 Try It Live":
    st.markdown(
        "<h1 style='color:#a78bfa;font-weight:700;margin-bottom:4px;'>🚀 Try It Live</h1>"
        "<p style='color:rgba(255,255,255,0.5);margin-top:0;'>Send a prompt through the full TokenGuard pipeline.</p>",
        unsafe_allow_html=True,
    )

    if guard_err:
        st.error(f"⚠️ TokenGuard failed to load: {guard_err}")
        st.info("Check your Python environment and model downloads.")
        st.stop()

    # ── Input section ────────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-header">📝 Input</div>', unsafe_allow_html=True)
        user_prompt = st.text_area(
            "Your prompt",
            placeholder="Ask anything… The more verbose, the more we can compress!",
            height=180,
            key="prompt_input",
            label_visibility="collapsed",
        )
        context_doc = st.text_area(
            "Context document (optional)",
            placeholder="Paste a long context / RAG document here…",
            height=120,
            key="context_input",
        )
        source_doc = st.text_area(
            "Source document for hallucination check (optional)",
            placeholder="Paste the ground-truth source document here…",
            height=100,
            key="source_input",
        )
        enable_hallucination = st.checkbox("Enable hallucination detection", value=True)

        run_btn = st.button("⚡ Optimize + Send", use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">📊 Compression Preview</div>', unsafe_allow_html=True)
        if user_prompt:
            compress_result = guard.compress_only(user_prompt)
            orig_col, comp_col = st.columns(2)
            with orig_col:
                st.caption("🔴 Original prompt")
                st.code(user_prompt, language=None)
                st.caption(f"{compress_result['original_tokens']} tokens")
            with comp_col:
                st.caption("🟢 Compressed prompt")
                st.code(compress_result["compressed_text"], language=None)
                st.caption(f"{compress_result['compressed_tokens']} tokens")

            removed = compress_result["removed_by_strategy"]
            st.markdown(
                f"**Removed:** "
                f"🗑 {removed['filler']} filler · "
                f"📋 {removed['duplicate']} duplicates · "
                f"📉 {removed['low_info']} low-info",
            )
        else:
            st.info("Enter a prompt on the left to see the compression preview.")

    st.divider()

    # ── Run pipeline ─────────────────────────────────────────────────────
    if run_btn:
        if not user_prompt.strip():
            st.warning("Please enter a prompt first.")
        else:
            with st.spinner("🔄 Running TokenGuard pipeline…"):
                source_docs = [source_doc] if source_doc.strip() else []
                try:
                    resp = guard.complete(
                        prompt=user_prompt,
                        context=context_doc if context_doc.strip() else None,
                        source_docs=source_docs if enable_hallucination else [],
                        model=selected_model,
                        max_tokens=max_tokens,
                    )
                    # Log for analytics
                    st.session_state["request_log"].append(resp.to_dict())
                except Exception as exc:
                    st.error(f"❌ Pipeline error: {exc}")
                    st.stop()

            # ── Metrics row ──────────────────────────────────────────────
            st.markdown('<div class="section-header">📈 Optimization Metrics</div>', unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Tokens Saved",      resp.tokens_saved,           delta=f"-{resp.compression_ratio*100:.1f}%")
            m2.metric("Original Tokens",   resp.original_tokens)
            m3.metric("Final Tokens",      resp.final_tokens)
            m4.metric("Cache Hit",         "✅ Yes" if resp.cache_hit else "❌ No")
            m5.metric("Latency",           f"{resp.latency_ms:.0f} ms")

            c1, c2, c3 = st.columns(3)
            c1.metric("Compression Ratio", f"{resp.compression_ratio*100:.1f}%")
            c2.metric("Hallucination Rate", f"{resp.hallucination_rate*100:.1f}%")
            c3.metric("Cost Saved (est.)", f"${resp.estimated_cost_saved_usd:.5f}")

            # Optimizations applied
            if resp.optimizations_applied:
                tags_html = " ".join(
                    f'<span style="background:rgba(99,102,241,0.25);color:#c4b5fd;'
                    f'padding:3px 10px;border-radius:20px;font-size:0.8rem;'
                    f'margin:3px;display:inline-block;">{o}</span>'
                    for o in resp.optimizations_applied
                )
                st.markdown(
                    f'<div style="margin:8px 0;">Optimizations applied: {tags_html}</div>',
                    unsafe_allow_html=True,
                )

            st.divider()

            # ── LLM Response ─────────────────────────────────────────────
            st.markdown('<div class="section-header">💬 LLM Response</div>', unsafe_allow_html=True)
            if resp.cache_hit:
                st.info("⚡ Served from semantic cache — no LLM call was made.")
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.05);border:1px solid '
                f'rgba(255,255,255,0.1);border-radius:12px;padding:20px;'
                f'color:#e2e8f0;line-height:1.7;font-size:0.95rem;">'
                f'{resp.text.replace(chr(10), "<br/>")}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ── Hallucination flags ───────────────────────────────────────
            if source_docs and enable_hallucination:
                st.markdown('<div class="section-header">🔍 Hallucination Analysis</div>', unsafe_allow_html=True)
                _render_flags(resp.hallucination_flags)


# ============================================================================
# PAGE 2 — ANALYTICS
# ============================================================================
elif page == "📊 Analytics":
    st.markdown(
        "<h1 style='color:#a78bfa;font-weight:700;margin-bottom:4px;'>📊 Analytics</h1>"
        "<p style='color:rgba(255,255,255,0.5);margin-top:0;'>Session-level optimization statistics.</p>",
        unsafe_allow_html=True,
    )

    log = st.session_state["request_log"]

    if not log:
        st.info("No requests yet — head to **Try It Live** and send some prompts!")
    else:
        # ── Metric cards ─────────────────────────────────────────────────
        total_saved   = sum(r.get("tokens_saved", 0) for r in log)
        total_cost    = sum(r.get("estimated_cost_saved_usd", 0.0) for r in log)
        cache_hits    = sum(1 for r in log if r.get("cache_hit"))
        avg_ratio     = sum(r.get("compression_ratio", 0) for r in log) / len(log)
        avg_latency   = sum(r.get("latency_ms", 0) for r in log) / len(log)

        st.markdown('<div class="section-header">🏆 Session Summary</div>', unsafe_allow_html=True)
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total Requests",      len(log))
        s2.metric("Tokens Saved",        f"{total_saved:,}")
        s3.metric("Cost Saved (est.)",   f"${total_cost:.4f}")
        s4.metric("Cache Hits",          f"{cache_hits}/{len(log)}")
        s5.metric("Avg Compression",     f"{avg_ratio*100:.1f}%")

        st.divider()

        # ── Chart row 1: tokens saved over time + cache pie ──────────────
        ch1, ch2 = st.columns([2, 1], gap="large")

        with ch1:
            st.markdown('<div class="section-header">📉 Tokens Saved Per Request</div>', unsafe_allow_html=True)
            tokens_data = [r.get("tokens_saved", 0) for r in log]
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=list(range(1, len(tokens_data) + 1)),
                y=tokens_data,
                mode="lines+markers",
                name="Tokens Saved",
                line=dict(color="#6366f1", width=2.5),
                marker=dict(color="#a78bfa", size=8, symbol="circle"),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.12)",
            ))
            fig_line.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="#c4b5fd", family="Inter"),
                xaxis=dict(title="Request #", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title="Tokens Saved", gridcolor="rgba(255,255,255,0.06)"),
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
            )
            st.plotly_chart(fig_line, use_container_width=True)

        with ch2:
            st.markdown('<div class="section-header">🎯 Cache Performance</div>', unsafe_allow_html=True)
            misses = len(log) - cache_hits
            fig_pie = go.Figure(go.Pie(
                labels=["Cache Hits", "Cache Misses"],
                values=[cache_hits, misses],
                marker=dict(colors=["#6366f1", "#374151"]),
                hole=0.55,
                textinfo="label+percent",
                textfont=dict(color="#e2e8f0"),
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#c4b5fd", family="Inter"),
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                height=280,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # ── Chart row 2: compression ratio over time ──────────────────────
        ch3, ch4 = st.columns([1, 1], gap="large")

        with ch3:
            st.markdown('<div class="section-header">📐 Compression Ratio Over Time</div>', unsafe_allow_html=True)
            ratios = [r.get("compression_ratio", 0) * 100 for r in log]
            fig_ratio = px.bar(
                x=list(range(1, len(ratios) + 1)),
                y=ratios,
                labels={"x": "Request #", "y": "Compression %"},
                color=ratios,
                color_continuous_scale=["#312e81", "#6366f1", "#a78bfa"],
            )
            fig_ratio.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="#c4b5fd", family="Inter"),
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=10, b=0),
                height=250,
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(fig_ratio, use_container_width=True)

        with ch4:
            st.markdown('<div class="section-header">⚡ Latency Distribution</div>', unsafe_allow_html=True)
            latencies = [r.get("latency_ms", 0) for r in log]
            fig_lat = px.histogram(
                x=latencies,
                nbins=min(len(latencies), 20),
                labels={"x": "Latency (ms)", "y": "Count"},
                color_discrete_sequence=["#8b5cf6"],
            )
            fig_lat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="#c4b5fd", family="Inter"),
                margin=dict(l=0, r=0, t=10, b=0),
                height=250,
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        # ── Savings breakdown bar chart ───────────────────────────────────
        st.markdown('<div class="section-header">🏗️ Savings by Strategy</div>', unsafe_allow_html=True)
        # Aggregate optimization counts
        strategy_counts: dict[str, int] = {}
        for r in log:
            for opt in r.get("optimizations_applied", []):
                strategy_counts[opt] = strategy_counts.get(opt, 0) + 1

        if strategy_counts:
            fig_bar = px.bar(
                x=list(strategy_counts.keys()),
                y=list(strategy_counts.values()),
                labels={"x": "Strategy", "y": "Times Applied"},
                color=list(strategy_counts.keys()),
                color_discrete_sequence=["#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd", "#ddd6fe"],
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.03)",
                font=dict(color="#c4b5fd", family="Inter"),
                showlegend=False,
                margin=dict(l=0, r=0, t=10, b=0),
                height=260,
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Raw log table ─────────────────────────────────────────────────
        with st.expander("📋 Raw Request Log"):
            import pandas as pd
            df = pd.DataFrame([
                {
                    "Request #": i + 1,
                    "Original Tokens": r.get("original_tokens", 0),
                    "Final Tokens": r.get("final_tokens", 0),
                    "Tokens Saved": r.get("tokens_saved", 0),
                    "Compression %": f"{r.get('compression_ratio',0)*100:.1f}%",
                    "Cache Hit": "✅" if r.get("cache_hit") else "❌",
                    "Hallucination Rate": f"{r.get('hallucination_rate',0)*100:.1f}%",
                    "Latency (ms)": f"{r.get('latency_ms',0):.0f}",
                    "Model": r.get("model_used", ""),
                }
                for i, r in enumerate(log)
            ])
            st.dataframe(df, use_container_width=True)


# ============================================================================
# PAGE 3 — BENCHMARK
# ============================================================================
elif page == "🧪 Benchmark":
    st.markdown(
        "<h1 style='color:#a78bfa;font-weight:700;margin-bottom:4px;'>🧪 Benchmark</h1>"
        "<p style='color:rgba(255,255,255,0.5);margin-top:0;'>Run predefined test cases and measure strategy performance.</p>",
        unsafe_allow_html=True,
    )

    # ── Load test cases ───────────────────────────────────────────────────
    test_cases_dir = _ROOT / "tests" / "test_cases"
    test_files = sorted(test_cases_dir.glob("*.txt")) if test_cases_dir.exists() else []

    if not test_files:
        st.warning(
            "No test cases found in `tests/test_cases/`.  "
            "Run the benchmark script first:  `python experiments/benchmark.py`"
        )
        st.stop()

    # ── Controls ──────────────────────────────────────────────────────────
    b_col1, b_col2, b_col3 = st.columns([2, 1, 1])
    with b_col1:
        selected_file = st.selectbox(
            "Select test case",
            options=[f.name for f in test_files],
        )
    with b_col2:
        bench_model = st.selectbox("Model for benchmark", ["claude-sonnet-4-6", "gpt-4o-mini"])
    with b_col3:
        st.markdown("<br/>", unsafe_allow_html=True)
        run_bench = st.button("▶ Run Benchmark", use_container_width=True)

    # ── Preview selected test case ────────────────────────────────────────
    if selected_file:
        selected_path = test_cases_dir / selected_file
        with st.expander(f"📄 Preview: {selected_file}"):
            content = selected_path.read_text(encoding="utf-8")
            st.text(content[:2000] + ("…" if len(content) > 2000 else ""))

    # ── Run benchmark ─────────────────────────────────────────────────────
    if run_bench and selected_file:
        content = (test_cases_dir / selected_file).read_text(encoding="utf-8")

        # Parse query from first line if format is "Query: ..."
        lines = content.splitlines()
        query = "Summarize the key points."
        for line in lines:
            if line.lower().startswith("query:"):
                query = line.split(":", 1)[1].strip()
                content = "\n".join(
                    l for l in lines if not l.lower().startswith("query:")
                )
                break

        results: list[dict] = []

        with st.spinner("🔄 Running benchmark strategies…"):
            strategies = [
                ("No Optimization",     False, False),
                ("Compression Only",    True,  False),
                ("Pruning + Compress",  True,  True),
            ]

            for strategy_name, do_compress, do_prune in strategies:
                t0 = time.perf_counter()

                if do_compress:
                    comp = guard.compress_only(query)
                    used_prompt = comp["compressed_text"]
                    orig_tok = comp["original_tokens"]
                    comp_tok = comp["compressed_tokens"]
                else:
                    used_prompt = query
                    orig_tok = int(len(query.split()) * 1.3)
                    comp_tok = orig_tok

                if do_prune:
                    from core.pruner import ContextPruner
                    pruner = ContextPruner()
                    prune_res = pruner.prune(used_prompt, content)
                    used_context = prune_res["pruned_context"]
                    pruned_tok = prune_res["tokens_saved"]
                else:
                    used_context = content
                    pruned_tok = 0

                latency = (time.perf_counter() - t0) * 1000
                context_tok = int(len(used_context.split()) * 1.3)
                total_saved = (orig_tok - comp_tok) + pruned_tok

                results.append({
                    "Strategy": strategy_name,
                    "Original Tokens": orig_tok + int(len(content.split()) * 1.3),
                    "Optimized Tokens": comp_tok + context_tok,
                    "Tokens Saved": total_saved,
                    "Savings %": f"{total_saved / max(orig_tok + int(len(content.split())*1.3), 1) * 100:.1f}%",
                    "Latency (ms)": f"{latency:.1f}",
                })

        # ── Results table ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Benchmark Results</div>', unsafe_allow_html=True)
        import pandas as pd
        df_bench = pd.DataFrame(results)
        st.dataframe(df_bench, use_container_width=True)

        # ── Tokens saved bar chart ─────────────────────────────────────────
        fig_bench = px.bar(
            df_bench,
            x="Strategy",
            y="Tokens Saved",
            color="Strategy",
            color_discrete_sequence=["#374151", "#6366f1", "#a78bfa"],
            text="Tokens Saved",
        )
        fig_bench.update_traces(textposition="outside", textfont=dict(color="#e2e8f0"))
        fig_bench.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,0.03)",
            font=dict(color="#c4b5fd", family="Inter"),
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0),
            height=300,
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(fig_bench, use_container_width=True)

        # ── CSV export ────────────────────────────────────────────────────
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=df_bench.columns.tolist())
        writer.writeheader()
        writer.writerows(results)

        st.download_button(
            label="⬇️ Export Results as CSV",
            data=csv_buf.getvalue().encode("utf-8"),
            file_name=f"tokenguard_benchmark_{selected_file.replace('.txt','')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
