"""Tests for deterministic source-grounded QA evaluation."""

from __future__ import annotations

import json

import pytest

from llm_bench.evals.aggregate import load_eval_results, primary_metric_view
from llm_bench.evals.sourceqa import SourceQATask, evaluate_answer, run_sourceqa


def test_evaluate_answer_checks_required_evidence_and_forbidden_terms():
    task = SourceQATask(
        id="demo",
        repo="https://example.invalid/repo.git",
        commit="abc123",
        question="Which helper should be used?",
        required_any=[
            ["use:enhance", "enhance"],
            ["$app/forms"],
        ],
        forbidden=["enhanceForm is documented"],
        evidence_paths=["docs/forms.md"],
    )

    result = evaluate_answer(
        task,
        "The documented helper is use:enhance from $app/forms. Evidence: docs/forms.md.",
    )

    assert result.score == pytest.approx(1.0)
    assert result.required_hits == 2
    assert result.forbidden_hits == []
    assert result.evidence_hits == ["docs/forms.md"]


def test_evaluate_answer_penalizes_missing_required_and_forbidden_terms():
    task = SourceQATask(
        id="demo",
        repo="https://example.invalid/repo.git",
        commit="abc123",
        question="Which helper should be used?",
        required_any=[["use:enhance"], ["$app/forms"]],
        forbidden=["enhanceForm is documented"],
        evidence_paths=["docs/forms.md"],
    )

    result = evaluate_answer(
        task,
        "enhanceForm is documented. I found it somewhere else.",
    )

    assert result.score < 0.5
    assert result.required_hits == 0
    assert result.forbidden_hits == ["enhanceForm is documented"]
    assert result.evidence_hits == []


def test_run_sourceqa_writes_synthetic_results_json(tmp_path):
    tasks_path = tmp_path / "tasks.json"
    tasks_path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "id": "docs-helper",
                        "repo": "local-fixture",
                        "commit": "fixture-commit",
                        "question": "Which helper should be used?",
                        "required_any": [["use:enhance"], ["$app/forms"]],
                        "forbidden": ["enhanceForm is documented"],
                        "evidence_paths": ["docs/forms.md"],
                    }
                ]
            }
        )
    )

    def fake_answer(task, _prompt):
        assert task.id == "docs-helper"
        return "Use use:enhance from $app/forms. Evidence: docs/forms.md."

    res = run_sourceqa(
        base_url="http://localhost:9090/v1",
        model_label="local-model",
        output_dir=tmp_path / "out",
        tasks_path=tasks_path,
        answer_fn=fake_answer,
    )

    assert res["task"] == "sourceqa"
    assert "error" not in res
    assert res["results"]["sourceqa"]["acc,none"] == pytest.approx(1.0)

    results_file = tmp_path / "out" / res["results_file"].split("/")[-1]
    parsed = json.loads(results_file.read_text())
    assert parsed["results"]["sourceqa"]["acc,none"] == pytest.approx(1.0)
    assert parsed["samples"][0]["answer"].startswith("Use use:enhance")
    assert parsed["samples"][0]["deterministic"]["score"] == pytest.approx(1.0)


def test_sourceqa_results_load_with_source_grounding_dimension(tmp_path):
    task_dir = tmp_path / "20260101T000000Z_vA_full" / "sourceqa" / "snapshot"
    task_dir.mkdir(parents=True)
    (task_dir / "results_2026-01-01_sourceqa.json").write_text(
        json.dumps({"results": {"sourceqa": {"acc,none": 0.75, "required_recall,none": 0.8}}})
    )

    full = load_eval_results(tmp_path)
    primary = primary_metric_view(full)

    assert full.iloc[0]["dim"] == "source_grounding"
    assert primary.iloc[0]["metric"] == "acc,none"
    assert primary.iloc[0]["value"] == pytest.approx(0.75)


def test_chat_completion_sends_bearer_token(monkeypatch):
    from llm_bench.evals.sourceqa import chat_completion

    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": "answer"}}]}

    def fake_post(url, *, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    answer = chat_completion(
        "https://models.example.test/v1",
        "provider/model",
        "question",
        api_key="secret-token",
    )

    assert answer == "answer"
    assert calls[0]["headers"]["Authorization"] == "Bearer secret-token"
