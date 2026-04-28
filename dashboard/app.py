"""Streamlit dashboard for MLX vs GGUF benchmark results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_bench.aggregate import aggregate_means, load_raw  # noqa: E402

RAW_DIR = ROOT / "results" / "raw"
SUMMARY_CSV = ROOT / "results" / "summary.csv"
QUALITY_GLOB = "quality_*.json"


st.set_page_config(page_title="MLX vs GGUF — Gemma 4 Bench", layout="wide")

FORMAT_COLORS = {"mlx": "#0a84ff", "gguf": "#34c759"}


@st.cache_data(ttl=10)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = load_raw(RAW_DIR)
    means = aggregate_means(raw) if not raw.empty else pd.DataFrame()
    return raw, means


@st.cache_data(ttl=10)
def load_quality() -> list[dict]:
    rows: list[dict] = []
    for p in sorted((ROOT / "results").glob(QUALITY_GLOB)):
        try:
            data = json.loads(p.read_text())
            if isinstance(data, list):
                rows.extend(data)
            else:
                rows.append(data)
        except Exception:
            continue
    return rows


def page_overview(raw: pd.DataFrame, means: pd.DataFrame) -> None:
    st.header("Overview")
    if means.empty:
        st.info("No measurements yet. Run `uv run python scripts/run_bench.py ...` first.")
        return

    models = sorted(means["model_id"].unique().tolist())
    sel_model = st.selectbox("Model", models, index=0)
    df = means[means["model_id"] == sel_model].copy()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Generation speed (tok/s)")
        fig = px.bar(
            df, x="scenario", y="tg_tps_mean", color="fmt", barmode="group",
            error_y="tg_tps_std",
            color_discrete_map=FORMAT_COLORS,
            labels={"tg_tps_mean": "TG tok/s", "scenario": "scenario"},
        )
        fig.update_layout(height=400, legend_title_text="format")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Prompt processing speed (tok/s)")
        fig = px.bar(
            df, x="scenario", y="pp_tps_mean", color="fmt", barmode="group",
            error_y="pp_tps_std",
            color_discrete_map=FORMAT_COLORS,
            labels={"pp_tps_mean": "PP tok/s"},
        )
        fig.update_layout(height=400, legend_title_text="format")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Speed vs Memory (Pareto view)")
    fig = px.scatter(
        df, x="peak_mem_gb_mean", y="tg_tps_mean", color="fmt",
        symbol="scenario", size_max=20, size=[12]*len(df),
        color_discrete_map=FORMAT_COLORS,
        hover_data=["scenario", "pp_tps_mean", "tg_tps_mean"],
        labels={"peak_mem_gb_mean": "peak memory (GB)", "tg_tps_mean": "TG tok/s"},
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Headline numbers")
    head = df.pivot_table(
        index="scenario", columns="fmt",
        values=["pp_tps_mean", "tg_tps_mean", "peak_mem_gb_mean"],
    ).round(1)
    st.dataframe(head, use_container_width=True)


def page_scaling(means: pd.DataFrame) -> None:
    st.header("Context-length scaling")
    if means.empty:
        st.info("No measurements yet.")
        return
    sel_metric = st.radio("Metric", ["TG tok/s", "PP tok/s"], horizontal=True)
    metric_col = "tg_tps_mean" if sel_metric == "TG tok/s" else "pp_tps_mean"

    df = means.copy()
    df["x_label"] = df["scenario"]
    fig = px.line(
        df.sort_values("n_prompt"),
        x="n_prompt", y=metric_col, color="fmt",
        line_dash="n_gen", markers=True,
        color_discrete_map=FORMAT_COLORS,
        labels={"n_prompt": "prefill length (tokens)", metric_col: sel_metric},
    )
    fig.update_xaxes(type="log")
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)


def page_quality() -> None:
    st.header("Output divergence (quality)")
    rows = load_quality()
    if not rows:
        st.info("No quality data. Run `uv run python scripts/compare_quality.py`.")
        return
    qdf = pd.DataFrame(rows)

    if "cos_sim" in qdf.columns:
        st.subheader("Embedding cosine similarity (MLX response vs GGUF response)")
        fig = go.Figure()
        fig.add_trace(go.Box(y=qdf["cos_sim"], name="all", boxpoints="all"))
        fig.update_layout(yaxis_title="cosine similarity", height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Higher = more similar. ~1.0 = nearly identical, <0.85 = noticeable divergence.")

    st.subheader("Side-by-side responses")
    for r in rows:
        with st.expander(f"[{r.get('id','?')}] {r.get('prompt','')[:80]}…"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**MLX**")
                st.code(r.get("mlx_response", ""))
            with c2:
                st.markdown("**GGUF**")
                st.code(r.get("gguf_response", ""))
            if "cos_sim" in r:
                st.metric("cos similarity", f"{r['cos_sim']:.3f}")


def page_raw(raw: pd.DataFrame) -> None:
    st.header("Raw measurements")
    if raw.empty:
        st.info("No measurements yet.")
        return
    st.dataframe(raw.sort_values("ts", ascending=False), use_container_width=True, height=600)
    st.download_button(
        "Download summary.csv",
        data=raw.to_csv(index=False).encode(),
        file_name="summary.csv",
    )


def main():
    st.title("MLX vs GGUF — Gemma 4 Benchmark")
    st.caption(f"Data dir: `{RAW_DIR}` (refresh with R)")
    raw, means = load_data()
    page = st.sidebar.radio("Page", ["Overview", "Scaling", "Quality", "Raw"])
    if page == "Overview":
        page_overview(raw, means)
    elif page == "Scaling":
        page_scaling(means)
    elif page == "Quality":
        page_quality()
    else:
        page_raw(raw)


if __name__ == "__main__":
    main()
