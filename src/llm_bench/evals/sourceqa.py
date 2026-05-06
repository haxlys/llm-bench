"""Source-grounded open-ended QA runner with deterministic scoring.

The runner gives the model curated evidence files from a pinned repository and
scores the answer with local string checks. Optional judge output is recorded as
metadata only; the primary score remains deterministic.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import yaml

DEFAULT_TASKS_PATH = Path(__file__).with_name("sourceqa_tasks.yaml")
DEFAULT_TIMEOUT_S = 10 * 60


@dataclass(frozen=True)
class SourceQATask:
    id: str
    repo: str
    commit: str
    question: str
    required_any: list[list[str]]
    forbidden: list[str] = field(default_factory=list)
    evidence_paths: list[str] = field(default_factory=list)
    search_path: str | None = None
    max_evidence_chars: int = 60_000


@dataclass(frozen=True)
class SourceQAScore:
    score: float
    required_hits: int
    required_total: int
    evidence_hits: list[str]
    evidence_total: int
    forbidden_hits: list[str]
    missed_required: list[list[str]]
    missed_evidence: list[str]

    @property
    def required_recall(self) -> float:
        return self.required_hits / self.required_total if self.required_total else 1.0

    @property
    def evidence_recall(self) -> float:
        return len(self.evidence_hits) / self.evidence_total if self.evidence_total else 1.0


AnswerFn = Callable[[SourceQATask, str], str]
EvidenceFn = Callable[[SourceQATask], str]


def load_tasks(path: Path = DEFAULT_TASKS_PATH) -> list[SourceQATask]:
    """Load source QA tasks from JSON or YAML."""
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    items = data.get("tasks", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(f"Expected a list of sourceqa tasks in {path}")
    return [_task_from_dict(item) for item in items]


def evaluate_answer(task: SourceQATask, answer: str) -> SourceQAScore:
    """Deterministically score required facts, forbidden phrases, and evidence citations."""
    normalized = _norm(answer)
    required_hits = 0
    missed_required: list[list[str]] = []
    for group in task.required_any:
        if any(_norm(term) in normalized for term in group):
            required_hits += 1
        else:
            missed_required.append(group)

    evidence_hits = [path for path in task.evidence_paths if _norm(path) in normalized]
    missed_evidence = [path for path in task.evidence_paths if path not in evidence_hits]
    forbidden_hits = [term for term in task.forbidden if _norm(term) in normalized]

    required_total = len(task.required_any)
    evidence_total = len(task.evidence_paths)
    required_recall = required_hits / required_total if required_total else 1.0
    evidence_recall = len(evidence_hits) / evidence_total if evidence_total else 1.0
    forbidden_ok = 0.0 if forbidden_hits else 1.0
    penalty = min(1.0, 0.25 * len(forbidden_hits))
    score = max(
        0.0,
        (0.70 * required_recall) + (0.20 * evidence_recall) + (0.10 * forbidden_ok) - penalty,
    )

    return SourceQAScore(
        score=round(score, 4),
        required_hits=required_hits,
        required_total=required_total,
        evidence_hits=evidence_hits,
        evidence_total=evidence_total,
        forbidden_hits=forbidden_hits,
        missed_required=missed_required,
        missed_evidence=missed_evidence,
    )


def build_prompt(task: SourceQATask, evidence: str) -> str:
    """Build the user prompt sent to the local OpenAI-compatible server."""
    return "\n\n".join(
        [
            "Answer the question using only the pinned source evidence below.",
            f"Repository: {task.repo}",
            f"Commit: {task.commit}",
            f"Question:\n{task.question}",
            "Instructions:\n"
            "- Cite the exact evidence file path(s) you used.\n"
            "- Include exact API names, commands, defaults, or symbols when relevant.\n"
            "- If the source does not document something, say that directly.\n"
            "- Be concise.",
            f"Evidence:\n{evidence}",
        ]
    )


def run_sourceqa(
    base_url: str,
    model_label: str,
    output_dir: Path,
    tasks_path: Path = DEFAULT_TASKS_PATH,
    cache_dir: Path | None = None,
    limit: int | None = None,
    judge_model: str | None = None,
    answer_fn: AnswerFn | None = None,
    evidence_fn: EvidenceFn | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    api_key: str | None = None,
) -> dict:
    """Run source-grounded QA tasks and write a synthetic results_*.json.

    `answer_fn` and `evidence_fn` exist to keep tests deterministic; production
    runs use the local OpenAI-compatible server and pinned repo evidence.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(tasks_path)
    if limit is not None:
        tasks = tasks[:limit]
    cache_root = cache_dir or output_dir.parent / "_sourceqa_cache"
    samples = []

    for task in tasks:
        try:
            if evidence_fn is not None:
                evidence = evidence_fn(task)
            elif answer_fn is not None:
                evidence = ""
            else:
                repo_path = ensure_repo(task, cache_root, timeout_s=timeout_s)
                evidence = read_evidence(task, repo_path)
            prompt = build_prompt(task, evidence)
            answer = (
                answer_fn(task, prompt)
                if answer_fn is not None
                else chat_completion(
                    base_url, model_label, prompt, timeout_s=timeout_s, api_key=api_key
                )
            )
            deterministic = evaluate_answer(task, answer)
            sample = {
                "id": task.id,
                "repo": task.repo,
                "commit": task.commit,
                "question": task.question,
                "evidence_paths": task.evidence_paths,
                "answer": answer,
                "deterministic": asdict(deterministic),
                "error": None,
            }
            if judge_model:
                sample["judge"] = run_optional_judge(
                    base_url=base_url,
                    judge_model=judge_model,
                    task=task,
                    answer=answer,
                    deterministic=deterministic,
                    timeout_s=timeout_s,
                    api_key=api_key,
                )
            samples.append(sample)
        except Exception as exc:
            samples.append(
                {
                    "id": task.id,
                    "repo": task.repo,
                    "commit": task.commit,
                    "question": task.question,
                    "evidence_paths": task.evidence_paths,
                    "answer": "",
                    "deterministic": asdict(
                        SourceQAScore(
                            score=0.0,
                            required_hits=0,
                            required_total=len(task.required_any),
                            evidence_hits=[],
                            evidence_total=len(task.evidence_paths),
                            forbidden_hits=[],
                            missed_required=task.required_any,
                            missed_evidence=task.evidence_paths,
                        )
                    ),
                    "error": str(exc),
                }
            )

    metrics = _aggregate_samples(samples)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    results_path = output_dir / f"results_{ts}_sourceqa.json"
    payload = {"results": {"sourceqa": metrics}, "samples": samples}
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    out: dict = {
        "task": "sourceqa",
        "results_file": str(results_path),
        "samples_file": str(results_path),
        "results": {"sourceqa": metrics},
    }
    if samples and all(sample.get("error") for sample in samples):
        out["error"] = "all sourceqa samples failed"
    return out


def ensure_repo(task: SourceQATask, cache_root: Path, timeout_s: int = DEFAULT_TIMEOUT_S) -> Path:
    """Clone/fetch a repo into cache and check out the pinned commit."""
    local = Path(task.repo).expanduser()
    if local.exists():
        return local.resolve()

    cache_root.mkdir(parents=True, exist_ok=True)
    target = cache_root / _safe_name(f"{task.repo}-{task.commit[:12]}")
    if not target.exists():
        _git(["clone", task.repo, str(target)], timeout_s=timeout_s)
    else:
        _git(["-C", str(target), "fetch", "--all", "--tags", "--prune"], timeout_s=timeout_s)
    _git(["-C", str(target), "checkout", "--force", task.commit], timeout_s=timeout_s)
    return target


def read_evidence(task: SourceQATask, repo_path: Path) -> str:
    """Read configured evidence files from repo_path with path traversal checks."""
    root = repo_path.resolve()
    remaining = task.max_evidence_chars
    chunks: list[str] = []
    for rel in task.evidence_paths:
        if remaining <= 0:
            break
        target = (root / rel).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Evidence path escapes repo: {rel}")
        text = target.read_text(encoding="utf-8", errors="replace")
        clipped = text[:remaining]
        remaining -= len(clipped)
        chunks.append(f"--- {rel} ---\n{clipped}")
    return "\n\n".join(chunks)


def chat_completion(
    base_url: str,
    model_label: str,
    prompt: str,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    max_tokens: int = 2048,
    api_key: str | None = None,
) -> str:
    """Call a local OpenAI-compatible chat completion endpoint."""
    import requests

    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        json={
            "model": model_label,
            "messages": [
                {"role": "system", "content": "You are a source-grounded evaluator target."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": max_tokens,
        },
        headers=_auth_headers(api_key),
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def run_optional_judge(
    base_url: str,
    judge_model: str,
    task: SourceQATask,
    answer: str,
    deterministic: SourceQAScore,
    timeout_s: int,
    api_key: str | None = None,
) -> dict:
    """Record optional judge metadata without changing deterministic score."""
    prompt = "\n\n".join(
        [
            "Score this source-grounded answer from 0 to 4. Output JSON only.",
            f"Question:\n{task.question}",
            f"Required signals:\n{json.dumps(task.required_any, ensure_ascii=False)}",
            f"Forbidden terms:\n{json.dumps(task.forbidden, ensure_ascii=False)}",
            f"Deterministic score:\n{json.dumps(asdict(deterministic), ensure_ascii=False)}",
            f"Answer:\n{answer}",
            'Return exactly: {"score": number, "notes": string}',
        ]
    )
    raw = chat_completion(
        base_url,
        judge_model,
        prompt,
        timeout_s=timeout_s,
        max_tokens=512,
        api_key=api_key,
    )
    parsed = _safe_json_object(raw)
    return {"model": judge_model, "raw": raw, "parsed": parsed}


def _aggregate_samples(samples: list[dict]) -> dict:
    if not samples:
        return {
            "acc,none": 0.0,
            "required_recall,none": 0.0,
            "evidence_recall,none": 0.0,
            "forbidden_violation_rate,none": 0.0,
            "n_samples,none": 0,
        }

    scores = [sample["deterministic"] for sample in samples]
    n = len(scores)

    def avg(values: list[float]) -> float:
        return round(sum(values) / n, 4)

    return {
        "acc,none": avg([score["score"] for score in scores]),
        "required_recall,none": avg(
            [
                score["required_hits"] / score["required_total"] if score["required_total"] else 1.0
                for score in scores
            ]
        ),
        "evidence_recall,none": avg(
            [
                len(score["evidence_hits"]) / score["evidence_total"]
                if score["evidence_total"]
                else 1.0
                for score in scores
            ]
        ),
        "forbidden_violation_rate,none": avg(
            [1.0 if score["forbidden_hits"] else 0.0 for score in scores]
        ),
        "n_samples,none": n,
    }


def _task_from_dict(data: dict) -> SourceQATask:
    required_any = data.get("required_any", [])
    if not isinstance(required_any, list):
        raise ValueError("required_any must be a list")
    groups = [group if isinstance(group, list) else [str(group)] for group in required_any]
    return SourceQATask(
        id=str(data["id"]),
        repo=str(data["repo"]),
        commit=str(data["commit"]),
        question=str(data["question"]),
        required_any=[[str(term) for term in group] for group in groups],
        forbidden=[str(term) for term in data.get("forbidden", [])],
        evidence_paths=[str(path) for path in data.get("evidence_paths", [])],
        search_path=data.get("search_path"),
        max_evidence_chars=int(data.get("max_evidence_chars", 60_000)),
    )


def _git(args: list[str], timeout_s: int) -> None:
    proc = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr[-1000:]}")


def _norm(value: str) -> str:
    return value.lower()


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")[:120]


def _auth_headers(api_key: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _safe_json_object(raw: str) -> dict | None:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
