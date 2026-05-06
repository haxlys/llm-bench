"""Tests for eval trace JSONL output."""

from __future__ import annotations

import json

from llm_bench.evals.trace import append_trace


def test_append_trace_writes_one_json_record_per_line(tmp_path):
    trace_path = tmp_path / "eval_traces" / "run.jsonl"

    append_trace(
        trace_path,
        {
            "variant": "26B-MoE-gguf-q8",
            "task": "sourceqa",
            "runner": "sourceqa",
            "wall_s": 1.2,
            "status": "ok",
            "results_file": "results.json",
            "error": None,
            "samples_file": "samples.json",
        },
    )
    append_trace(
        trace_path,
        {
            "variant": "26B-MoE-gguf-q8",
            "task": "bfcl",
            "runner": "bfcl",
            "wall_s": 2.3,
            "status": "error",
            "results_file": None,
            "error": "boom",
        },
    )

    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]

    assert [row["task"] for row in rows] == ["sourceqa", "bfcl"]
    assert rows[0]["status"] == "ok"
    assert rows[1]["error"] == "boom"
