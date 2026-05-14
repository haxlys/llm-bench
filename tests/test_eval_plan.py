"""Tests for ordered eval catch-up planning."""

from __future__ import annotations

from llm_bench.eval_plan import build_eval_catchup_plan, render_markdown_plan


def _variant(key: str, coverage: list[dict], measured: int = 8, total: int = 8) -> dict:
    return {
        "key": key,
        "speed": {
            "scenarios_measured": measured,
            "scenarios_total": total,
        },
        "evals": {"coverage": coverage},
    }


def _row(task: str, status: str, lane: str = "primary") -> dict:
    return {
        "dim": "test",
        "task": task,
        "runner": "runner",
        "lane": lane,
        "required": lane == "primary",
        "status": status,
    }


def test_build_eval_catchup_plan_orders_primary_speed_and_optional_work():
    plan = build_eval_catchup_plan({
        "variants": [
            _variant(
                "model-a",
                [
                    _row("kmmlu_pro", "missing"),
                    _row("bfcl", "optional", lane="optional"),
                ],
                measured=4,
                total=8,
            ),
            _variant(
                "model-b",
                [
                    _row("sourceqa", "diagnostic", lane="diagnostic"),
                    _row("mbpp", "missing"),
                    _row("programbench", "optional", lane="optional"),
                ],
            ),
            _variant(
                "mtplx-mtp",
                [_row("sourceqa", "speed_only", lane="mtplx_speedup")],
                measured=0,
                total=8,
            ),
        ]
    })

    assert plan["summary"] == {
        "primary_missing": 2,
        "optional_pending": 2,
        "speed_incomplete": 2,
    }
    assert plan["primary"][0]["id"] == "kmmlu_pro"
    assert plan["primary"][0]["tasks"] == ["kmmlu_pro"]
    assert "TASKS=\"kmmlu_pro\"" in plan["primary"][0]["command"]
    assert plan["primary"][1]["id"] == "primary_code"
    assert plan["speed"]["variants"][0]["variant"] == "model-a"
    assert plan["optional"][0]["id"] == "optional_bigcodebench_bfcl_livebench"
    assert "LLM_BENCH_INCLUDE_BFCL=1" in plan["optional"][0]["command"]
    assert plan["optional"][1]["id"] == "optional_programbench"
    assert "scripts/import_programbench.py" in plan["optional"][1]["command"]


def test_render_markdown_plan_includes_commands():
    markdown = render_markdown_plan({
        "summary": {
            "primary_missing": 0,
            "optional_pending": 0,
            "speed_incomplete": 0,
        },
        "primary": [],
        "speed": {
            "variants": [],
            "title": "Complete speed.",
            "command": "",
            "followup_command": "",
        },
        "optional": [],
    })

    assert "Primary missing rows: 0" in markdown
    assert "No pending rows." in markdown
