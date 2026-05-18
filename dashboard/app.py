"""Streamlit dashboard for llm-bench results."""

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
from llm_bench.reporting import ordered_variants, runtime_column  # noqa: E402

RAW_DIR = ROOT / "results" / "raw"
EVAL_DIR = ROOT / "results" / "eval_scores"
SUMMARY_CSV = ROOT / "results" / "summary.csv"
QUALITY_GLOB = "quality_*.json"


st.set_page_config(page_title="llm-bench", layout="wide")

RUNTIME_COLORS = {
    "mlx": "#0a84ff",
    "gguf": "#34c759",
    "openai-compatible": "#ff9f0a",
}

STATUS_LABELS = {
    "measured": "OK - completed",
    "directional": "WEAK - directional score",
    "diagnostic": "CHECK - diagnostic only",
    "missing": "MISSING - run needed",
    "optional": "TODO - optional",
    "speed_only": "SPEED - no eval target",
    "unsupported": "N/A - unsupported",
}

STATUS_SHORT = {
    "measured": "OK",
    "directional": "WEAK",
    "diagnostic": "CHECK",
    "missing": "MISSING",
    "optional": "TODO",
    "speed_only": "SPEED",
    "unsupported": "N/A",
}

STATUS_HELP = {
    "measured": "Measured and suitable for normal comparison.",
    "directional": "Measured, but answer extraction or task design can undercount quality.",
    "diagnostic": "Useful for investigation, but not a headline benchmark score.",
    "missing": "Supported required task has not completed for this variant.",
    "optional": "Supported optional task is not required for the primary matrix.",
    "speed_only": "This variant is intentionally tracked for speed, not eval coverage.",
    "unsupported": "Task is not applicable to this variant's declared capabilities.",
}

STATUS_ACTION = {
    "measured": "No action.",
    "directional": "Use as a signal, not as a final ranking.",
    "diagnostic": "Review when debugging behavior or data quality.",
    "missing": "Run the required eval for this variant/task.",
    "optional": "Run later if this lane matters for the comparison.",
    "speed_only": "Use speed pages instead of eval pages.",
    "unsupported": "No action unless registry capabilities are wrong.",
}

STATUS_SEVERITY = {
    "missing": 0,
    "directional": 1,
    "diagnostic": 2,
    "optional": 3,
    "speed_only": 4,
    "unsupported": 5,
    "measured": 6,
}

STATUS_COMPLETENESS = {
    "missing": 0,
    "unsupported": 1,
    "speed_only": 2,
    "optional": 3,
    "diagnostic": 4,
    "directional": 5,
    "measured": 6,
}

ATTENTION_STATUSES = {"missing", "directional", "diagnostic"}
TASK_LANE_ORDER = {"primary": 0, "diagnostic": 1, "optional": 2, "mtplx_speedup": 3}


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


def _pct(done: int, total: int) -> float:
    return (done / total * 100) if total else 0.0


def _coverage_frame(index: dict) -> pd.DataFrame:
    rows = []
    for v in index["variants"]:
        speed = v["speed"]
        evals = v["evals"]
        summary = evals.get("coverage_summary", {})
        attention_count = sum(summary.get(s, 0) for s in ATTENTION_STATUSES)
        for row in evals.get("coverage", []):
            status = row["status"]
            rows.append({
                "Model": v["model_id"],
                "Variant": v["key"],
                "Backend": v.get("backend", v["fmt"]),
                "Tier": v["tier"],
                "Local": "yes" if v["exists_locally"] else "no",
                "Speed": f"{speed['scenarios_measured']}/{speed['scenarios_total']}",
                "Speed %": _pct(speed["scenarios_measured"], speed["scenarios_total"]),
                "Eval": f"{evals['tasks_measured']}/{evals['tasks_supported']}",
                "Eval %": _pct(evals["tasks_measured"], evals["tasks_supported"]),
                "Attention": attention_count,
                "Dim": row["dim"],
                "Task": row["task"],
                "Runner": row["runner"],
                "Lane": row["lane"],
                "Required": "yes" if row["required"] else "no",
                "Supported": "yes" if row["supported"] else "no",
                "Measured": "yes" if row["measured"] else "no",
                "Status": status,
                "Status label": STATUS_LABELS.get(status, status),
                "Meaning": STATUS_HELP.get(status, ""),
                "Next action": STATUS_ACTION.get(status, ""),
                "_severity": STATUS_SEVERITY.get(status, 9),
                "_lane_order": TASK_LANE_ORDER.get(row["lane"], 9),
            })
    return pd.DataFrame(rows)


def _variant_status_frame(index: dict) -> pd.DataFrame:
    rows = []
    for v in index["variants"]:
        speed = v["speed"]
        evals = v["evals"]
        summary = evals.get("coverage_summary", {})
        attention_count = sum(summary.get(s, 0) for s in ATTENTION_STATUSES)
        rows.append({
            "Local": "yes" if v["exists_locally"] else "no",
            "Model": v["model_id"],
            "Variant": v["key"],
            "Backend": v.get("backend", v["fmt"]),
            "Tier": v["tier"],
            "Speed": f"{speed['scenarios_measured']}/{speed['scenarios_total']}",
            "Speed %": _pct(speed["scenarios_measured"], speed["scenarios_total"]),
            "Eval": f"{evals['tasks_measured']}/{evals['tasks_supported']}",
            "Eval %": _pct(evals["tasks_measured"], evals["tasks_supported"]),
            "Missing": summary.get("missing", 0),
            "Weak": summary.get("directional", 0),
            "Diagnostic": summary.get("diagnostic", 0),
            "Optional open": summary.get("optional", 0),
            "Attention": attention_count,
            "Speed last": (speed["last_measured"] or "-")[:19],
            "Eval last": (evals["last_measured"] or "-")[:19],
        })
    return pd.DataFrame(rows).sort_values(
        ["Attention", "Missing", "Weak", "Model", "Variant"],
        ascending=[False, False, False, True, True],
    )


def _status_matrix_frame(
    coverage: pd.DataFrame,
    variant_order: list[str],
) -> pd.DataFrame:
    rows = []
    grouped = coverage.groupby(["Dim", "Task", "Lane"], sort=False)
    for (dim, task, lane), group in grouped:
        by_variant = group.set_index("Variant")["Status"].to_dict()
        row = {"Dim": dim, "Task": task, "Lane": lane}
        severities = []
        for variant in variant_order:
            status = by_variant.get(variant, "unsupported")
            row[variant] = STATUS_SHORT.get(status, status.upper())
            severities.append(STATUS_SEVERITY.get(status, 9))
        row["_sort"] = min(severities) if severities else 9
        row["_lane_order"] = TASK_LANE_ORDER.get(lane, 9)
        rows.append(row)
    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return matrix
    return matrix.sort_values(["_sort", "_lane_order", "Dim", "Task"]).drop(
        columns=["_sort", "_lane_order"],
    )


def _score_lookup(primary: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return pd.DataFrame(columns=["Variant", "Task", "Score", "Metric"])
    scores = primary[["variant", "task", "value", "metric"]].copy()
    scores = scores.rename(columns={
        "variant": "Variant",
        "task": "Task",
        "value": "Score",
        "metric": "Metric",
    })
    scores["Score"] = scores["Score"].round(4)
    return scores


def _task_metadata(coverage: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    meta = coverage[["Task", "Dim", "Lane", "Runner"]].drop_duplicates()
    if primary.empty:
        return meta
    scored = primary[["task", "dim"]].drop_duplicates().rename(
        columns={"task": "Task", "dim": "Dim"},
    )
    meta = meta.merge(scored, on=["Task", "Dim"], how="outer")
    meta["Lane"] = meta["Lane"].fillna("historical")
    meta["Runner"] = meta["Runner"].fillna("results")
    return meta


def _variant_label(row: pd.Series) -> str:
    pieces = [
        str(row.get("backend", "")),
        str(row.get("quant", "")),
        str(row.get("tier", "")),
    ]
    return " · ".join(p for p in pieces if p and p != "nan")


def _speed_compare_rows(means: pd.DataFrame, models: list[str], scenario: str) -> pd.DataFrame:
    if means.empty or not models:
        return pd.DataFrame()
    df = means[(means["model_id"].isin(models)) & (means["scenario"] == scenario)].copy()
    if df.empty:
        return df
    df["Variant used"] = df.apply(_variant_label, axis=1)
    df = df.sort_values(["model_id", "tg_tps_mean"], ascending=[True, False])
    best = df.groupby("model_id", as_index=False).head(1).copy()
    best = best.rename(columns={
        "model_id": "Model",
        "tg_tps_mean": "TG tok/s",
        "pp_tps_mean": "PP tok/s",
        "peak_mem_gb_mean": "Peak GB",
        "wall_s_mean": "Wall s",
        "n_runs": "Runs",
    })
    return best[[
        "Model",
        "Variant used",
        "scenario",
        "TG tok/s",
        "PP tok/s",
        "Peak GB",
        "Wall s",
        "Runs",
    ]]


def _best_score_rows(
    primary: pd.DataFrame,
    coverage: pd.DataFrame,
    models: list[str],
    lanes: list[str],
) -> pd.DataFrame:
    if primary.empty or not models:
        return pd.DataFrame()
    meta = _task_metadata(coverage, primary)
    scores = primary.copy()
    scores = scores.rename(columns={
        "model_id": "Model",
        "variant": "Variant used",
        "task": "Task",
        "value": "Score",
        "metric": "Metric",
    })
    scores = scores[scores["Model"].isin(models)]
    scores = scores.merge(meta, on="Task", how="left", suffixes=("", "_catalog"))
    scores["Dim"] = scores["Dim"].fillna(scores.get("dim", ""))
    scores["Lane"] = scores["Lane"].fillna("historical")
    scores = scores[scores["Lane"].isin(lanes)]
    if scores.empty:
        return scores
    scores = scores.sort_values(["Model", "Task", "Score"], ascending=[True, True, False])
    best = scores.groupby(["Model", "Task"], as_index=False).head(1).copy()
    best["Score"] = best["Score"].round(4)
    return best[[
        "Model",
        "Variant used",
        "Dim",
        "Task",
        "Lane",
        "Metric",
        "Score",
    ]]


def _score_matrix(best_scores: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    if best_scores.empty:
        return pd.DataFrame()
    matrix = best_scores.pivot_table(
        index=["Dim", "Task", "Lane"],
        columns="Model",
        values="Score",
        aggfunc="max",
    ).reset_index()
    cols = ["Dim", "Task", "Lane"] + [m for m in models if m in matrix.columns]
    return matrix[cols].sort_values(["Lane", "Dim", "Task"])


def _model_gap_matrix(
    coverage: pd.DataFrame,
    models: list[str],
    lanes: list[str],
) -> pd.DataFrame:
    if coverage.empty or not models:
        return pd.DataFrame()
    rows = []
    filtered = coverage[
        coverage["Model"].isin(models)
        & coverage["Lane"].isin(lanes)
        & (coverage["Status"] != "unsupported")
    ].copy()
    for (dim, task, lane), group in filtered.groupby(["Dim", "Task", "Lane"], sort=False):
        row = {"Dim": dim, "Task": task, "Lane": lane}
        sort_values = []
        for model in models:
            statuses = group[group["Model"] == model]["Status"].tolist()
            if statuses:
                best_status = max(statuses, key=lambda s: STATUS_COMPLETENESS.get(s, -1))
            else:
                best_status = "unsupported"
            row[model] = STATUS_SHORT.get(best_status, best_status.upper())
            sort_values.append(STATUS_SEVERITY.get(best_status, 9))
        row["_sort"] = min(sort_values) if sort_values else 9
        row["_lane_order"] = TASK_LANE_ORDER.get(lane, 9)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    matrix = pd.DataFrame(rows)
    return matrix.sort_values(["_sort", "_lane_order", "Dim", "Task"]).drop(
        columns=["_sort", "_lane_order"],
    )


def _model_compare_summary(
    models: list[str],
    variants: pd.DataFrame,
    speed_rows: pd.DataFrame,
    score_rows: pd.DataFrame,
    gap_matrix: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model in models:
        v = variants[variants["Model"] == model]
        speed = speed_rows[speed_rows["Model"] == model] if not speed_rows.empty else pd.DataFrame()
        scores = score_rows[score_rows["Model"] == model] if not score_rows.empty else pd.DataFrame()
        gaps = gap_matrix[["Task", model]] if not gap_matrix.empty and model in gap_matrix else pd.DataFrame()
        missing = int((gaps[model] == "MISSING").sum()) if not gaps.empty else 0
        weak = int((gaps[model] == "WEAK").sum()) if not gaps.empty else 0
        check = int((gaps[model] == "CHECK").sum()) if not gaps.empty else 0
        rows.append({
            "Model": model,
            "Variants": len(v),
            "Best speed variant": speed["Variant used"].iloc[0] if not speed.empty else "-",
            "TG tok/s": speed["TG tok/s"].iloc[0] if not speed.empty else None,
            "Peak GB": speed["Peak GB"].iloc[0] if not speed.empty else None,
            "Scored tasks": scores["Task"].nunique() if not scores.empty else 0,
            "Missing": missing,
            "Weak": weak,
            "Check": check,
            "Best eval %": v["Eval %"].max() if not v.empty else 0,
            "Best speed %": v["Speed %"].max() if not v.empty else 0,
        })
    return pd.DataFrame(rows)


def page_model_compare(index: dict, means: pd.DataFrame, primary: pd.DataFrame) -> None:
    """Compare models side-by-side across speed, eval scores, and coverage debt."""
    st.header("Model Compare")
    st.caption(
        "모델을 여러 개 선택해 속도, 평가 점수, 누락/주의 태스크를 나란히 봅니다. "
        "점수와 속도는 모델별 최고 variant를 대표값으로 사용합니다."
    )

    variants = _variant_status_frame(index)
    coverage = _coverage_frame(index)
    if variants.empty:
        st.info("No model registry data yet.")
        return

    model_rank = variants.groupby("Model", as_index=False).agg(
        Variants=("Variant", "count"),
        best_eval=("Eval %", "max"),
        best_speed=("Speed %", "max"),
        attention=("Attention", "sum"),
    )
    model_rank = model_rank.sort_values(
        ["best_eval", "best_speed", "attention", "Model"],
        ascending=[False, False, True, True],
    )
    model_options = model_rank["Model"].tolist()
    default_models = model_options[: min(5, len(model_options))]

    selected_models = st.multiselect(
        "Models to compare",
        model_options,
        default=default_models,
    )
    if not selected_models:
        st.info("Select at least one model.")
        return

    scenario_options = sorted(means[means["model_id"].isin(selected_models)]["scenario"].unique())
    if not scenario_options:
        scenario_options = sorted(means["scenario"].unique()) if not means.empty else []
    default_scenario = scenario_options.index("p256_g128") if "p256_g128" in scenario_options else 0

    lane_options = [
        lane for lane in ["primary", "diagnostic", "optional", "historical", "mtplx_speedup"]
        if lane == "historical" or lane in set(coverage["Lane"])
    ]
    default_lanes = [
        lane for lane in ["primary", "diagnostic", "optional", "historical"]
        if lane in lane_options
    ]

    c1, c2, c3 = st.columns([1.1, 1, 1])
    with c1:
        speed_scenario = st.selectbox(
            "Speed scenario",
            scenario_options,
            index=default_scenario if scenario_options else None,
            disabled=not scenario_options,
        )
    with c2:
        speed_metric = st.radio(
            "Speed metric",
            ["TG tok/s", "PP tok/s"],
            horizontal=True,
        )
    with c3:
        selected_lanes = st.multiselect(
            "Eval lanes",
            lane_options,
            default=default_lanes,
        )
    if not selected_lanes:
        st.info("Select at least one eval lane.")
        return

    speed_rows = _speed_compare_rows(means, selected_models, speed_scenario) if scenario_options else pd.DataFrame()
    score_rows = _best_score_rows(primary, coverage, selected_models, selected_lanes)

    if not score_rows.empty:
        task_counts = score_rows.groupby("Task")["Model"].nunique().sort_values(ascending=False)
        default_tasks = task_counts[task_counts >= min(2, len(selected_models))].index.tolist()
        if not default_tasks:
            default_tasks = task_counts.index.tolist()[:12]
        selected_tasks = st.multiselect(
            "Score tasks",
            task_counts.index.tolist(),
            default=default_tasks,
        )
        score_rows = score_rows[score_rows["Task"].isin(selected_tasks)]

    gap_matrix = _model_gap_matrix(coverage, selected_models, selected_lanes)
    summary = _model_compare_summary(selected_models, variants, speed_rows, score_rows, gap_matrix)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Models", len(selected_models))
    k2.metric("Scored task cells", len(score_rows))
    k3.metric("Missing cells", int(summary["Missing"].sum()) if not summary.empty else 0)
    k4.metric("Weak/check cells", int(summary[["Weak", "Check"]].sum().sum()) if not summary.empty else 0)

    st.subheader("Comparison scorecard")
    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
        column_config={
            "TG tok/s": st.column_config.NumberColumn("TG tok/s", format="%.2f"),
            "Peak GB": st.column_config.NumberColumn("Peak GB", format="%.2f"),
            "Best eval %": st.column_config.ProgressColumn(
                "Best eval %", min_value=0, max_value=100, format="%.0f%%"),
            "Best speed %": st.column_config.ProgressColumn(
                "Best speed %", min_value=0, max_value=100, format="%.0f%%"),
        },
    )

    chart_col, debt_col = st.columns(2)
    with chart_col:
        st.subheader("Speed winner by model")
        if speed_rows.empty:
            st.info("No speed rows for the current model/scenario selection.")
        else:
            fig = px.bar(
                speed_rows.sort_values(speed_metric, ascending=False),
                x="Model",
                y=speed_metric,
                color="Variant used",
                text=speed_metric,
                hover_data=["PP tok/s", "TG tok/s", "Peak GB", "Wall s", "Runs"],
                labels={speed_metric: speed_metric},
            )
            fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig.update_layout(height=430, showlegend=True, xaxis_tickangle=-25)
            st.plotly_chart(fig, width="stretch")
    with debt_col:
        st.subheader("Coverage debt")
        if summary.empty:
            st.info("No coverage rows for selected models.")
        else:
            debt = summary[["Model", "Missing", "Weak", "Check"]].copy()
            fig = px.bar(
                debt,
                x="Model",
                y=["Missing", "Weak", "Check"],
                barmode="stack",
                labels={"value": "task cells", "variable": "status"},
            )
            fig.update_layout(height=430, xaxis_tickangle=-25)
            st.plotly_chart(fig, width="stretch")

    st.subheader("Eval score matrix")
    score_matrix = _score_matrix(score_rows, selected_models)
    if score_matrix.empty:
        st.info("No eval scores match the selected models and lanes.")
    else:
        heatmap_source = score_matrix.copy()
        heatmap_source["Task label"] = (
            heatmap_source["Lane"] + " / "
            + heatmap_source["Dim"] + " / "
            + heatmap_source["Task"]
        )
        heatmap_cols = [model for model in selected_models if model in heatmap_source.columns]
        heatmap = heatmap_source.set_index("Task label")[heatmap_cols]
        fig = px.imshow(
            heatmap,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale="viridis",
            labels={"x": "model", "y": "task", "color": "best score"},
        )
        fig.update_layout(height=min(720, 180 + 30 * len(heatmap)))
        st.plotly_chart(fig, width="stretch")
        with st.expander("Winning variants behind each score", expanded=False):
            st.dataframe(
                score_rows.sort_values(["Task", "Score"], ascending=[True, False]),
                width="stretch",
                hide_index=True,
                column_config={
                    "Score": st.column_config.NumberColumn("Score", format="%.3f"),
                },
                height=min(520, 96 + 35 * len(score_rows)),
            )

    st.subheader("Task status matrix")
    st.caption(
        "모델별 최고 상태를 비교합니다. MISSING은 아직 대표 variant 기준으로도 "
        "완료된 결과가 없다는 뜻입니다."
    )
    if gap_matrix.empty:
        st.info("No task status rows match the selected lanes.")
    else:
        st.dataframe(
            gap_matrix,
            width="stretch",
            hide_index=True,
            height=min(720, 72 + 35 * len(gap_matrix)),
        )


def page_model_status(index: dict, primary: pd.DataFrame) -> None:
    """Model-first benchmark coverage page."""
    st.header("Model Benchmark Status")
    st.caption(
        "모델별로 어떤 벤치마크가 완료됐고, 어떤 태스크가 누락/주의 상태인지 "
        "먼저 볼 수 있는 운영 화면입니다."
    )

    coverage = _coverage_frame(index)
    variants = _variant_status_frame(index)
    if coverage.empty or variants.empty:
        st.info("No registry coverage data yet.")
        return

    total_missing = int(variants["Missing"].sum())
    total_weak = int(variants["Weak"].sum())
    total_diagnostic = int(variants["Diagnostic"].sum())
    total_attention = int(variants["Attention"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model variants", len(variants))
    c2.metric("Required missing", total_missing)
    c3.metric("Weak scores", total_weak)
    c4.metric("Needs attention", total_attention)
    st.caption(
        f"Diagnostic rows: {total_diagnostic}. "
        "Missing은 꼭 채워야 하는 태스크, Weak는 점수 해석에 주의가 필요한 태스크입니다."
    )

    with st.expander("Status legend", expanded=False):
        legend = pd.DataFrame([
            {"Code": STATUS_SHORT[k], "Status": STATUS_LABELS[k], "Meaning": STATUS_HELP[k]}
            for k in [
                "measured",
                "missing",
                "directional",
                "diagnostic",
                "optional",
                "speed_only",
                "unsupported",
            ]
        ])
        st.dataframe(legend, width="stretch", hide_index=True)

    st.subheader("All model variants")
    only_attention = st.toggle("Show only variants with missing/weak/diagnostic tasks", value=False)
    table = variants.copy()
    if only_attention:
        table = table[table["Attention"] > 0]
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Speed %": st.column_config.ProgressColumn(
                "Speed %", min_value=0, max_value=100, format="%.0f%%"),
            "Eval %": st.column_config.ProgressColumn(
                "Eval %", min_value=0, max_value=100, format="%.0f%%"),
            "Missing": st.column_config.NumberColumn(
                "Missing", help="Supported required tasks that have not completed."),
            "Weak": st.column_config.NumberColumn(
                "Weak", help="Measured directional rows; useful but not final ranking quality."),
            "Diagnostic": st.column_config.NumberColumn(
                "Diagnostic", help="Diagnostic rows to review separately from headline scores."),
            "Attention": st.column_config.NumberColumn(
                "Attention", help="Missing + Weak + Diagnostic."),
        },
        height=360,
    )

    st.subheader("Model detail")
    model_counts = variants.groupby("Model").agg(
        variants=("Variant", "count"),
        missing=("Missing", "sum"),
        attention=("Attention", "sum"),
    )
    model_options = sorted(
        variants["Model"].unique().tolist(),
        key=lambda m: (-model_counts.loc[m, "attention"], m),
    )

    def model_label(model: str) -> str:
        row = model_counts.loc[model]
        return (
            f"{model} | {int(row['variants'])} variants, "
            f"{int(row['missing'])} missing, {int(row['attention'])} attention"
        )

    selected_model = st.selectbox(
        "Model",
        model_options,
        format_func=model_label,
    )
    model_variants = variants[variants["Model"] == selected_model].copy()
    variant_options = model_variants["Variant"].tolist()
    selected_variants = st.multiselect(
        "Variants",
        variant_options,
        default=variant_options,
    )
    if not selected_variants:
        st.info("Select at least one variant.")
        return

    lane_options = [lane for lane in ["primary", "diagnostic", "optional", "mtplx_speedup"]
                    if lane in set(coverage["Lane"])]
    default_lanes = [lane for lane in ["primary", "diagnostic", "optional"]
                     if lane in lane_options]
    selected_lanes = st.multiselect(
        "Task lanes",
        lane_options,
        default=default_lanes or lane_options,
    )
    show_unsupported = st.checkbox("Show unsupported tasks in matrix", value=False)

    detail = model_variants[model_variants["Variant"].isin(selected_variants)]
    st.dataframe(
        detail,
        width="stretch",
        hide_index=True,
        column_config={
            "Speed %": st.column_config.ProgressColumn(
                "Speed %", min_value=0, max_value=100, format="%.0f%%"),
            "Eval %": st.column_config.ProgressColumn(
                "Eval %", min_value=0, max_value=100, format="%.0f%%"),
        },
    )

    model_coverage = coverage[
        (coverage["Model"] == selected_model)
        & (coverage["Variant"].isin(selected_variants))
        & (coverage["Lane"].isin(selected_lanes))
    ].copy()
    if not show_unsupported:
        model_coverage = model_coverage[model_coverage["Status"] != "unsupported"]

    st.markdown("### Task matrix")
    st.caption(
        "각 행은 태스크, 각 열은 variant입니다. MISSING/WEAK/CHECK가 있으면 "
        "아래 부실/주의 태스크 표에서 바로 원인을 확인하세요."
    )
    matrix = _status_matrix_frame(model_coverage, selected_variants)
    if matrix.empty:
        st.info("No tasks match the current filters.")
    else:
        st.dataframe(
            matrix,
            width="stretch",
            hide_index=True,
            height=min(720, 72 + 35 * len(matrix)),
        )

    st.markdown("### Weak / Missing Tasks")
    st.caption("부실/주의 태스크만 모아 봅니다. Optional TODO는 필요할 때만 포함하세요.")
    include_optional = st.checkbox("Include optional TODO tasks", value=False)
    issue_statuses = set(ATTENTION_STATUSES)
    if include_optional:
        issue_statuses.add("optional")

    issues = coverage[
        (coverage["Model"] == selected_model)
        & (coverage["Variant"].isin(selected_variants))
        & (coverage["Status"].isin(issue_statuses))
    ].copy()
    scores = _score_lookup(primary)
    issues = issues.merge(scores, how="left", on=["Variant", "Task"])
    issues = issues.sort_values(
        ["_severity", "_lane_order", "Variant", "Dim", "Task"],
        ascending=[True, True, True, True, True],
    )

    if issues.empty:
        st.success("No missing, weak, or diagnostic tasks for the current selection.")
    else:
        st.dataframe(
            issues[[
                "Variant",
                "Backend",
                "Tier",
                "Dim",
                "Task",
                "Lane",
                "Status label",
                "Score",
                "Metric",
                "Meaning",
                "Next action",
            ]],
            width="stretch",
            hide_index=True,
            column_config={
                "Score": st.column_config.NumberColumn("Score", format="%.3f"),
                "Status label": st.column_config.TextColumn(
                    "Status", help="Why this row needs attention."),
            },
            height=min(620, 96 + 35 * len(issues)),
        )


def page_overview(raw: pd.DataFrame, means: pd.DataFrame) -> None:
    st.header("Overview")
    if means.empty:
        st.info("No measurements yet. Run `uv run python scripts/run_bench.py ...` first.")
        return

    models = sorted(means["model_id"].unique().tolist())
    sel_model = st.selectbox("Model", models, index=0)
    df = means[means["model_id"] == sel_model].copy()
    runtime = runtime_column(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Generation speed (tok/s)")
        fig = px.bar(
            df, x="scenario", y="tg_tps_mean", color=runtime, barmode="group",
            error_y="tg_tps_std",
            color_discrete_map=RUNTIME_COLORS,
            labels={"tg_tps_mean": "TG tok/s", "scenario": "scenario"},
        )
        fig.update_layout(height=400, legend_title_text=runtime)
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.subheader("Prompt processing speed (tok/s)")
        fig = px.bar(
            df, x="scenario", y="pp_tps_mean", color=runtime, barmode="group",
            error_y="pp_tps_std",
            color_discrete_map=RUNTIME_COLORS,
            labels={"pp_tps_mean": "PP tok/s"},
        )
        fig.update_layout(height=400, legend_title_text=runtime)
        st.plotly_chart(fig, width="stretch")

    st.subheader("Speed vs Memory (Pareto view)")
    fig = px.scatter(
        df, x="peak_mem_gb_mean", y="tg_tps_mean", color=runtime,
        symbol="scenario", size_max=20, size=[12]*len(df),
        color_discrete_map=RUNTIME_COLORS,
        hover_data=["scenario", "pp_tps_mean", "tg_tps_mean"],
        labels={"peak_mem_gb_mean": "peak memory (GB)", "tg_tps_mean": "TG tok/s"},
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Headline numbers")
    head = df.pivot_table(
        index="scenario", columns=runtime,
        values=["pp_tps_mean", "tg_tps_mean", "peak_mem_gb_mean"],
    ).round(1)
    st.dataframe(head, width="stretch")


def page_scaling(means: pd.DataFrame) -> None:
    st.header("Context-length scaling")
    if means.empty:
        st.info("No measurements yet.")
        return
    sel_metric = st.radio("Metric", ["TG tok/s", "PP tok/s"], horizontal=True)
    metric_col = "tg_tps_mean" if sel_metric == "TG tok/s" else "pp_tps_mean"

    df = means.copy()
    runtime = runtime_column(df)
    df["x_label"] = df["scenario"]
    fig = px.line(
        df.sort_values("n_prompt"),
        x="n_prompt", y=metric_col, color=runtime,
        line_dash="n_gen", markers=True,
        color_discrete_map=RUNTIME_COLORS,
        labels={"n_prompt": "prefill length (tokens)", metric_col: sel_metric},
    )
    fig.update_xaxes(type="log")
    fig.update_layout(height=480)
    st.plotly_chart(fig, width="stretch")


def page_quality() -> None:
    st.header("Output divergence (quality)")
    rows = load_quality()
    if not rows:
        st.info("No quality data. Run `uv run python scripts/compare_quality.py`.")
        return
    qdf = pd.DataFrame(rows)

    if "cos_sim" in qdf.columns:
        st.subheader("Embedding cosine similarity")
        fig = go.Figure()
        fig.add_trace(go.Box(y=qdf["cos_sim"], name="all", boxpoints="all"))
        fig.update_layout(yaxis_title="cosine similarity", height=380)
        st.plotly_chart(fig, width="stretch")
        st.caption("Higher = more similar. ~1.0 = nearly identical, <0.85 = noticeable divergence.")

    st.subheader("Side-by-side responses")
    for r in rows:
        with st.expander(f"[{r.get('id','?')}] {r.get('prompt','')[:80]}…"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**{r.get('left_label', 'A')}**")
                st.code(r.get("left_response", r.get("mlx_response", "")))
            with c2:
                st.markdown(f"**{r.get('right_label', 'B')}**")
                st.code(r.get("right_response", r.get("gguf_response", "")))
            if "cos_sim" in r:
                st.metric("cos similarity", f"{r['cos_sim']:.3f}")


def page_raw(raw: pd.DataFrame) -> None:
    st.header("Raw measurements")
    if raw.empty:
        st.info("No measurements yet.")
        return
    st.dataframe(raw.sort_values("ts", ascending=False), width="stretch", height=600)
    st.download_button(
        "Download summary.csv",
        data=raw.to_csv(index=False).encode(),
        file_name="summary.csv",
    )


# ---------------- Eval pages ----------------

def page_evals_overview(primary: pd.DataFrame) -> None:
    st.header("Eval scores — overview heatmap")
    if primary.empty:
        st.info("No eval results yet. Run `bash scripts/run_evals_overnight.sh`.")
        return

    pivot = primary.pivot_table(index="task", columns="variant", values="value")
    cols = [c for c in ordered_variants(primary) if c in pivot.columns]
    cols += [c for c in pivot.columns if c not in cols]
    pivot = pivot[cols]

    fig = px.imshow(
        pivot, text_auto=".3f", aspect="auto", color_continuous_scale="viridis",
        labels={"x": "variant", "y": "task", "color": "score"},
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, width="stretch")

    with st.expander("Primary metric per task"):
        st.dataframe(
            primary[["task", "variant", "metric", "value", "stderr"]]
            .sort_values(["task", "variant"]),
            width="stretch", height=400,
        )


def page_evals_compare(primary: pd.DataFrame) -> None:
    st.header("Runtime accuracy delta")
    if primary.empty:
        st.info("No eval results yet.")
        return

    group_options = [c for c in ["backend", "fmt", "artifact_type"] if c in primary.columns]
    group_col = st.selectbox("Compare by", group_options, index=0) if group_options else "fmt"
    groups = sorted(g for g in primary[group_col].dropna().unique().tolist() if g)
    if len(groups) < 2:
        st.info(f"Need at least two {group_col} groups for the same model_id+tier.")
        return
    c1, c2 = st.columns(2)
    with c1:
        baseline = st.selectbox("Baseline", groups, index=0)
    with c2:
        challenger = st.selectbox("Challenger", groups, index=min(1, len(groups) - 1))
    if baseline == challenger:
        st.info("Choose two different groups.")
        return

    pivot = primary.pivot_table(
        index=["model_id", "tier", "task"], columns=group_col, values="value",
    ).reset_index()
    if pivot.empty or baseline not in pivot.columns or challenger not in pivot.columns:
        st.info(f"Need both {baseline} and {challenger} runs for the same model_id+tier.")
        return
    pivot = pivot.dropna(subset=[baseline, challenger])
    if pivot.empty:
        st.info(f"Need both {baseline} and {challenger} runs for the same model_id+tier.")
        return
    delta_col = f"delta_{challenger}_minus_{baseline}"
    pivot[delta_col] = pivot[challenger] - pivot[baseline]
    pivot["pct_delta"] = (pivot[delta_col] / pivot[baseline]).round(3)

    sel_tier = st.radio("Tier", sorted(pivot["tier"].unique()), horizontal=True)
    view = pivot[pivot["tier"] == sel_tier].sort_values(delta_col)
    fig = px.bar(
        view, x=delta_col, y="task", color="model_id",
        orientation="h", text=delta_col,
        labels={delta_col: f"{challenger} − {baseline} score"},
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=400 + 30 * len(view))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Positive = {challenger} wins, negative = {baseline} wins. Same model + tier.")
    st.dataframe(view, width="stretch")


def page_evals_quantization(primary: pd.DataFrame) -> None:
    st.header("Quantization — 8bit vs 4bit accuracy")
    if primary.empty:
        st.info("No eval results yet.")
        return
    runtime = runtime_column(primary)
    pivot = primary.pivot_table(
        index=["model_id", runtime, "task"], columns="tier", values="value",
    ).dropna(how="any").reset_index()
    if pivot.empty:
        st.info(f"Need both 8bit and 4bit runs for the same model_id+{runtime}.")
        return
    pivot["delta_4bit_minus_8bit"] = pivot["4bit"] - pivot["8bit"]
    fig = px.bar(
        pivot.sort_values("delta_4bit_minus_8bit"),
        x="delta_4bit_minus_8bit", y="task", color=runtime, facet_col="model_id",
        orientation="h", labels={"delta_4bit_minus_8bit": "4bit − 8bit score"},
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, width="stretch")
    st.caption("Negative = quantization hurt. Bars near 0 = 4bit is essentially free.")
    st.dataframe(pivot, width="stretch")


def page_evals_dimension(full: pd.DataFrame, primary: pd.DataFrame) -> None:
    st.header("Per-dimension breakdown")
    if primary.empty:
        st.info("No eval results yet.")
        return
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
    st.plotly_chart(fig, width="stretch")


def page_evals_longbench(full: pd.DataFrame) -> None:
    st.header("LongBench — sub-task detail")
    if full.empty:
        st.info("No eval results yet.")
        return
    lb = full[full["task"] == "longbench"].copy()
    if lb.empty:
        st.info("LongBench not in the current results.")
        return
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
    st.plotly_chart(fig, width="stretch")
    st.caption("21 sub-tasks across QA, summarization, code, few-shot, retrieval. Filter view in Raw.")


def page_evals_raw(full: pd.DataFrame) -> None:
    st.header("Eval raw rows")
    if full.empty:
        st.info("No eval results yet.")
        return
    st.dataframe(full.sort_values(["task", "variant", "metric"]),
                 width="stretch", height=600)
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
        s = v["speed"]
        e = v["evals"]
        speed_pct = (s["scenarios_measured"] / s["scenarios_total"] * 100
                     if s["scenarios_total"] else 0)
        eval_pct = (e["tasks_measured"] / e["tasks_supported"] * 100
                    if e["tasks_supported"] else 0)
        rows.append({
            "Local": "✓" if v["exists_locally"] else "✗",
            "Variant": v["key"],
            "Model": v["model_id"],
            "Fmt": v["fmt"],
            "Backend": v.get("backend", v["fmt"]),
            "Artifact": v.get("artifact_type", ""),
            "Tier": v["tier"],
            "Capabilities": ", ".join(v.get("capabilities", [])),
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
        width="stretch",
        column_config={
            "Speed %": st.column_config.ProgressColumn(
                "Speed %", min_value=0, max_value=100, format="%.0f%%"),
            "Evals %": st.column_config.ProgressColumn(
                "Evals %", min_value=0, max_value=100, format="%.0f%%"),
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
    st.title("llm-bench")
    st.caption(f"Speed dir: `{RAW_DIR}`  ·  Eval dir: `{EVAL_DIR}` (refresh with R)")
    raw, means = load_data()
    full_evals, primary_evals = load_evals()
    index = load_index()

    pages = {
        # Status
        "Model Status": lambda: page_model_status(index, primary_evals),
        "Model Compare": lambda: page_model_compare(index, means, primary_evals),
        "Catalog": lambda: page_catalog(index),
        # Speed/memory
        "Speed Overview": lambda: page_overview(raw, means),
        "Speed Scaling": lambda: page_scaling(means),
        "Output Quality (cos sim)": lambda: page_quality(),
        "Speed Raw": lambda: page_raw(raw),
        # Evals
        "Evals Heatmap": lambda: page_evals_overview(primary_evals),
        "Evals · Runtime Compare": lambda: page_evals_compare(primary_evals),
        "Evals · Quantization": lambda: page_evals_quantization(primary_evals),
        "Evals · Dimension": lambda: page_evals_dimension(full_evals, primary_evals),
        "Evals · LongBench Detail": lambda: page_evals_longbench(full_evals),
        "Evals Raw": lambda: page_evals_raw(full_evals),
    }
    page = st.sidebar.radio("Page", list(pages.keys()))
    pages[page]()


if __name__ == "__main__":
    main()
