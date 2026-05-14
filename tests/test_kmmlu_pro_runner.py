"""Tests for the KMMLU-Pro external runner."""

from __future__ import annotations

import json

import pytest

from llm_bench.evals import kmmlu_pro_runner as kmmlu


class _Message:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, answers: list[str]):
        self.answers = answers
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Response(self.answers.pop(0))


class _Chat:
    def __init__(self, answers: list[str]):
        self.completions = _Completions(answers)


class _Client:
    def __init__(self, answers: list[str]):
        self.chat = _Chat(answers)


def test_run_kmmlu_pro_scores_weighted_accuracy(tmp_path, monkeypatch):
    rows = [
        {
            "question": "1+1?",
            "options": ["2", "3", "4", "5"],
            "solution": 1,
            "score": 2,
            "license_name": "law",
            "subject": "math",
        },
        {
            "question": "2+2?",
            "options": ["2", "3", "4", "5"],
            "solution": "C",
            "score": 1,
            "license_name": "law",
            "subject": "math",
        },
    ]
    client = _Client(["풀이...\n정답: A", "정답: B"])
    monkeypatch.setattr(kmmlu, "_load_rows", lambda limit: rows[:limit])
    monkeypatch.setattr(kmmlu, "_openai_client", lambda base_url, api_key: client)

    res = kmmlu.run_kmmlu_pro(
        base_url="http://localhost:9090/v1",
        model_label="m",
        output_dir=tmp_path,
        limit=2,
    )

    assert res["task"] == "kmmlu_pro"
    assert "error" not in res
    assert res["results"]["kmmlu_pro"]["acc,none"] == pytest.approx(2 / 3)
    assert res["results"]["kmmlu_pro"]["questions,none"] == pytest.approx(2.0)
    assert res["results"]["kmmlu_pro"]["law_acc,none"] == pytest.approx(2 / 3)
    assert client.chat.completions.calls[0]["model"] == "m"
    assert client.chat.completions.calls[0]["messages"][0]["role"] == "system"
    assert "exactly one uppercase letter" in client.chat.completions.calls[0]["messages"][0]["content"]
    assert client.chat.completions.calls[0]["max_tokens"] == kmmlu.DEFAULT_MAX_TOKENS

    synth = list(tmp_path.glob("results_*_kmmlu_pro.json"))
    assert len(synth) == 1
    parsed = json.loads(synth[0].read_text())
    assert parsed["results"]["kmmlu_pro"]["acc,none"] == pytest.approx(2 / 3)
    assert (tmp_path / "samples_kmmlu_pro.jsonl").exists()


def test_extract_choice_prefers_final_answer_marker():
    assert kmmlu._extract_choice("A도 가능하지만 최종 정답: C") == "C"


def test_build_prompt_requires_answer_letter_only():
    prompt = kmmlu._build_prompt({"question": "1+1?", "options": ["2", "3", "4", "5", "6"]})

    assert "문자 하나만" in prompt
    assert "`정답:` 접두사는 출력하지 마세요" in prompt


def test_target_choice_accepts_one_indexed_solution():
    assert kmmlu._target_choice({"solution": 1}) == "A"
    assert kmmlu._target_choice({"solution": "5"}) == "E"
