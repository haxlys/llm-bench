"""Tests for the synthetic memory-stability diagnostic runner."""

from __future__ import annotations

import json

import pytest

from llm_bench.evals import memory_stability as ms


def test_parse_output_grid_accepts_fenced_json_object():
    assert ms.parse_output_grid('```json\n{"output": [[1, 2], [3, 4]]}\n```') == [
        [1, 2],
        [3, 4],
    ]


def test_apply_family_oracles_are_deterministic():
    grid = [
        [1, 2, 6],
        [4, 9, 0],
    ]

    assert ms.apply_family("f17", grid) == [[1, 2, 0], [4, 9, 0]]
    assert ms.apply_family("f23", grid) == [[1, 7, 6], [4, 9, 0]]
    assert ms.apply_family("f41", grid) == [[0, 0, 0], [4, 0, 0]]
    assert ms.apply_family("f58", grid) == [[9, 2, 6], [4, 1, 0]]


def test_solve_prompt_marks_memory_as_retrieved_strategy():
    case = ms.build_eval_cases(limit=1)[0]

    prompt = ms.build_solve_prompt(case, "General lesson: keep the largest frame.")

    assert "Final retrieved long-term memory for this task" in prompt
    assert "Use this memory as optional prior experience" in prompt
    assert "MEMORY OVERRIDES CURRENT TASK EXAMPLES" not in prompt
    assert "Current task examples" in prompt
    assert prompt.index("Current task examples") < prompt.index("Final retrieved")


def test_solve_prompt_can_mark_memory_as_override():
    case = ms.build_eval_cases(limit=1)[0]

    prompt = ms.build_solve_prompt(
        case,
        "General lesson: keep the largest frame.",
        memory_priority="override",
    )

    assert "MEMORY OVERRIDES CURRENT TASK EXAMPLES" in prompt
    assert "base the high-level transformation" in prompt


def test_solve_prompt_can_withhold_current_examples_for_forced_mode():
    case = ms.build_eval_cases(limit=1)[0]

    prompt = ms.build_solve_prompt(
        case,
        "General lesson: keep the largest frame.",
        include_examples=False,
    )

    assert "withheld in stress-replacement mode" in prompt
    assert case.examples[0].id not in prompt


def test_run_memory_stability_detects_forced_abstraction_collapse(tmp_path):
    def fake_consolidate(prompt: str, _max_tokens: int) -> str:
        if "family-specific memory note" in prompt:
            for family in ms.FAMILIES:
                if f"Task code: {family.id}" in prompt:
                    return f"{family.id}: {family.description}."
            raise AssertionError("gated prompt did not include a known task code")
        return "General lesson: erase color 6 from all grids."

    def fake_answer(
        case: ms.EvalCase,
        _prompt: str,
        _policy: str,
        memory_context: str,
    ) -> str:
        if f"Task code: {case.rule_id}" in memory_context:
            output = case.expected_output
        elif f"{case.rule_id}:" in memory_context:
            output = case.expected_output
        elif "erase color 6" in memory_context and (
            "withheld in stress-replacement mode" in _prompt
            or "MEMORY OVERRIDES CURRENT TASK EXAMPLES" in _prompt
        ):
            output = ms.apply_family("f17", case.input_grid)
        else:
            output = case.expected_output
        return json.dumps({"output": output})

    result = ms.run_memory_stability(
        base_url="http://localhost:9090/v1",
        model_label="fake-model",
        output_dir=tmp_path,
        answer_fn=fake_answer,
        consolidate_fn=fake_consolidate,
    )

    metrics = result["results"]["memory_stability"]
    assert metrics["episodic_only_acc,none"] == pytest.approx(1.0)
    assert metrics["gated_abstraction_acc,none"] == pytest.approx(1.0)
    assert metrics["forced_abstraction_acc,none"] < metrics["episodic_only_acc,none"]
    assert metrics["forced_abstraction_acc,none"] < metrics["no_memory_acc,none"]
    assert metrics["stress_replacement_forced_abstraction_acc,none"] < 1.0
    assert metrics["fair_same_evidence_forced_abstraction_acc,none"] == pytest.approx(1.0)
    assert metrics["fair_same_evidence_gap,none"] == pytest.approx(0.0)
    assert metrics["forced_memory_collapse_rate,none"] == pytest.approx(1.0)

    results_file = tmp_path / result["results_file"].split("/")[-1]
    parsed = json.loads(results_file.read_text())
    assert parsed["results"]["memory_stability"]["acc,none"] == metrics["acc,none"]
    assert {sample["policy"] for sample in parsed["samples"]} == {
        "no_memory",
        "episodic_only",
        "forced_abstraction",
        "gated_abstraction",
    }
    assert {sample["mode"] for sample in parsed["samples"]} == {
        "stress_replacement",
        "fair_same_evidence",
    }
