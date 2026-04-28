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
from llm_bench.evals.aggregate import (  # noqa: E402
    load_eval_results,
    primary_metric_view,
)
from llm_bench.index import build_index  # noqa: E402
from llm_bench.registry import get_registry  # noqa: E402

RAW_DIR = ROOT / "results" / "raw"
EVAL_DIR = ROOT / "results" / "eval_scores"
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
def load_evals() -> tuple[pd.DataFrame, pd.DataFrame]:
    full = load_eval_results(EVAL_DIR) if EVAL_DIR.exists() else pd.DataFrame()
    primary = primary_metric_view(full) if not full.empty else pd.DataFrame()
    return full, primary


@st.cache_data(ttl=10)
def load_index() -> dict:
    return build_index()


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


# ---------------- Eval pages ----------------

VARIANT_ORDER = [
    "26B-MoE-mlx-8bit", "26B-MoE-gguf-q8",
    "26B-MoE-mlx-4bit", "26B-MoE-gguf-q4",
    "31B-Dense-mlx-8bit", "31B-Dense-gguf-q8",
]


def page_evals_overview(primary: pd.DataFrame) -> None:
    st.header("Eval scores — overview heatmap")
    if primary.empty:
        st.info("No eval results yet. Run `bash scripts/run_evals_overnight.sh`.")
        return

    pivot = primary.pivot_table(index="task", columns="variant", values="value")
    # Reorder columns to a friendly variant ordering
    cols = [c for c in VARIANT_ORDER if c in pivot.columns] + \
           [c for c in pivot.columns if c not in VARIANT_ORDER]
    pivot = pivot[cols]

    fig = px.imshow(
        pivot, text_auto=".3f", aspect="auto", color_continuous_scale="viridis",
        labels={"x": "variant", "y": "task", "color": "score"},
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Primary metric per task"):
        st.dataframe(
            primary[["task", "variant", "metric", "value", "stderr"]]
            .sort_values(["task", "variant"]),
            use_container_width=True, height=400,
        )


def page_evals_compare(primary: pd.DataFrame) -> None:
    st.header("MLX vs GGUF — accuracy delta")
    if primary.empty:
        st.info("No eval results yet."); return

    # join on (model_id, tier, task)
    df = primary.copy()
    # Bring back model_id, fmt, tier from VARIANT_META via variant key
    from llm_bench.evals.aggregate import VARIANT_META, TIER_MAP
    meta = pd.DataFrame(
        [(k, *v, TIER_MAP.get(v[2], "")) for k, v in VARIANT_META.items()],
        columns=["variant", "model_id", "fmt", "quant", "tier"],
    )
    df = df.merge(meta, on="variant")
    pivot = df.pivot_table(
        index=["model_id", "tier", "task"], columns="fmt", values="value",
    ).dropna(how="any").reset_index()
    if pivot.empty:
        st.info("Need both MLX and GGUF runs for the same model_id+tier.")
        return
    pivot["delta_mlx_minus_gguf"] = pivot["mlx"] - pivot["gguf"]
    pivot["pct_delta"] = (pivot["delta_mlx_minus_gguf"] / pivot["gguf"]).round(3)

    sel_tier = st.radio("Tier", sorted(pivot["tier"].unique()), horizontal=True)
    view = pivot[pivot["tier"] == sel_tier].sort_values("delta_mlx_minus_gguf")
    fig = px.bar(
        view, x="delta_mlx_minus_gguf", y="task", color="model_id",
        orientation="h", text="delta_mlx_minus_gguf",
        labels={"delta_mlx_minus_gguf": "MLX − GGUF score"},
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=400 + 30 * len(view))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Positive = MLX wins, negative = GGUF wins. Same model + same tier.")
    st.dataframe(view, use_container_width=True)


def page_evals_quantization(primary: pd.DataFrame) -> None:
    st.header("Quantization — 8bit vs 4bit accuracy")
    if primary.empty:
        st.info("No eval results yet."); return
    from llm_bench.evals.aggregate import VARIANT_META, TIER_MAP
    meta = pd.DataFrame(
        [(k, *v, TIER_MAP.get(v[2], "")) for k, v in VARIANT_META.items()],
        columns=["variant", "model_id", "fmt", "quant", "tier"],
    )
    df = primary.merge(meta, on="variant")
    pivot = df.pivot_table(
        index=["model_id", "fmt", "task"], columns="tier", values="value",
    ).dropna(how="any").reset_index()
    if pivot.empty:
        st.info("Need both 8bit and 4bit runs for the same model_id+fmt.")
        return
    pivot["delta_4bit_minus_8bit"] = pivot["4bit"] - pivot["8bit"]
    fig = px.bar(
        pivot.sort_values("delta_4bit_minus_8bit"),
        x="delta_4bit_minus_8bit", y="task", color="fmt", facet_col="model_id",
        orientation="h", labels={"delta_4bit_minus_8bit": "4bit − 8bit score"},
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Negative = quantization hurt. Bars near 0 = 4bit is essentially free.")
    st.dataframe(pivot, use_container_width=True)


def page_evals_dimension(full: pd.DataFrame, primary: pd.DataFrame) -> None:
    st.header("Per-dimension breakdown")
    if primary.empty:
        st.info("No eval results yet."); return
    dims = sorted(primary["dim"].unique()) if "dim" in primary.columns else []
    if not dims:
        # primary view dropped dim col when picking; reattach from full
        primary = primary.merge(
            full[["task", "dim"]].drop_duplicates(), on="task", how="left",
        )
        dims = sorted(primary["dim"].dropna().unique())
    sel_dim = st.selectbox("Dimension", dims)
    sub = primary[primary["dim"] == sel_dim] if "dim" in primary.columns else primary
    fig = px.bar(
        sub, x="task", y="value", color="variant", barmode="group",
        error_y="stderr", labels={"value": "score"},
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def page_evals_longbench(full: pd.DataFrame) -> None:
    st.header("LongBench — sub-task detail")
    if full.empty:
        st.info("No eval results yet."); return
    lb = full[full["task"] == "longbench"].copy()
    if lb.empty:
        st.info("LongBench not in the current results."); return
    # subtask is e.g. "longbench_narrativeqa" — strip prefix for axis
    lb["subtask_short"] = lb["subtask"].str.replace("longbench_", "", regex=False)
    # filter to leaf metrics only (skip group rows where value is 0 by structure)
    leaf = lb[lb["value"] > 0].copy()
    if leaf.empty:
        leaf = lb
    fig = px.bar(
        leaf, x="subtask_short", y="value", color="variant", barmode="group",
        labels={"value": "score", "subtask_short": "subtask"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=520)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("21 sub-tasks across QA, summarization, code, few-shot, retrieval. Filter view in Raw.")


def page_evals_raw(full: pd.DataFrame) -> None:
    st.header("Eval raw rows")
    if full.empty:
        st.info("No eval results yet."); return
    st.dataframe(full.sort_values(["task", "variant", "metric"]),
                 use_container_width=True, height=600)
    st.download_button(
        "Download eval_summary_full.csv",
        data=full.to_csv(index=False).encode(),
        file_name="eval_summary_full.csv",
    )


def page_catalog(index: dict) -> None:
    """Top-level model catalog: registry × measurement status."""
    st.header("Model Catalog")
    st.caption(
        f"Registry: `{index['registry_path']}` · "
        f"bench_version: `{index['bench_version']}` · "
        f"generated: {index['generated_at']}"
    )

    totals = index["totals"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Variants in registry", totals["variants"])
    c2.metric("Speed complete", totals["speed_complete"])
    c3.metric("Evals started", totals["evals_started"])
    c4.metric("Speed pending", totals["variants"] - totals["speed_complete"])

    rows = []
    for v in index["variants"]:
        s = v["speed"]; e = v["evals"]
        speed_pct = (s["scenarios_measured"] / s["scenarios_total"]
                     if s["scenarios_total"] else 0)
        eval_pct = (e["tasks_measured"] / e["tasks_supported"]
                    if e["tasks_supported"] else 0)
        rows.append({
            "Local": "✓" if v["exists_locally"] else "✗",
            "Variant": v["key"],
            "Model": v["model_id"],
            "Fmt": v["fmt"],
            "Tier": v["tier"],
            "Size GB": v["approx_size_gb"],
            "Speed": f"{s['scenarios_measured']}/{s['scenarios_total']}",
            "Speed %": speed_pct,
            "Speed last": (s["last_measured"] or "—")[:19],
            "Evals": f"{e['tasks_measured']}/{e['tasks_supported']}",
            "Evals %": eval_pct,
            "Evals last": (e["last_measured"] or "—")[:19],
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Speed %": st.column_config.ProgressColumn(
                "Speed %", min_value=0, max_value=1, format="%.0f%%"),
            "Evals %": st.column_config.ProgressColumn(
                "Evals %", min_value=0, max_value=1, format="%.0f%%"),
        },
        hide_index=True,
        height=320,
    )

    st.markdown("### Add a new model")
    st.code(
        "# 1. Edit models/registry.yaml — add a new entry under `models:`\n"
        "# 2. uv run python scripts/sync_models.py --all-missing\n"
        "# 3. uv run python scripts/run_bench.py --all-pending\n"
        "# 4. uv run python scripts/run_evals.py --all-variants --suite full --skip-existing\n"
        "# 5. Refresh this page (R) — new variant appears with progress bars at 0%",
        language="bash",
    )


def main():
    st.title("llm-bench — MLX vs GGUF on Apple Silicon")
    st.caption(f"Speed dir: `{RAW_DIR}`  ·  Eval dir: `{EVAL_DIR}` (refresh with R)")
    raw, means = load_data()
    full_evals, primary_evals = load_evals()
    index = load_index()

    pages = {
        # Catalog (top-level status)
        "Catalog": lambda: page_catalog(index),
        # Speed/memory
        "Speed Overview": lambda: page_overview(raw, means),
        "Speed Scaling": lambda: page_scaling(means),
        "Output Quality (cos sim)": lambda: page_quality(),
        "Speed Raw": lambda: page_raw(raw),
        # Evals
        "Evals Heatmap": lambda: page_evals_overview(primary_evals),
        "Evals · MLX vs GGUF": lambda: page_evals_compare(primary_evals),
        "Evals · Quantization": lambda: page_evals_quantization(primary_evals),
        "Evals · Dimension": lambda: page_evals_dimension(full_evals, primary_evals),
        "Evals · LongBench Detail": lambda: page_evals_longbench(full_evals),
        "Evals Raw": lambda: page_evals_raw(full_evals),
    }
    page = st.sidebar.radio("Page", list(pages.keys()))
    pages[page]()


if __name__ == "__main__":
    main()
