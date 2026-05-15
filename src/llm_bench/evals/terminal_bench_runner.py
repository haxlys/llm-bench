"""Terminal-Bench wrapper.

Terminal-Bench is a Docker-backed agentic terminal benchmark. This wrapper
keeps the integration small: run ``tb run`` against an OpenAI-compatible model
endpoint, then convert Terminal-Bench's ``results.json`` into the synthetic
``results_*.json`` shape consumed by the normal llm-bench aggregation pipeline.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

TASK_NAME = "terminal_bench"
DEFAULT_DATASET = "terminal-bench-core==0.1.1"
DEFAULT_AGENT = "terminus"
DEFAULT_BIN = "uvx --from terminal-bench tb"


def terminal_bench_available(terminal_bench_bin: str = DEFAULT_BIN) -> bool:
    parts = shlex.split(terminal_bench_bin)
    return bool(parts and shutil.which(parts[0]))


def run_terminal_bench(
    *,
    base_url: str,
    model_label: str,
    output_dir: Path,
    dataset: str = DEFAULT_DATASET,
    agent: str = DEFAULT_AGENT,
    task_ids: list[str] | None = None,
    n_tasks: int | None = 1,
    terminal_bench_bin: str = DEFAULT_BIN,
    docker_host: str | None = None,
    api_key: str | None = None,
    run_id: str | None = None,
    global_agent_timeout_sec: int | None = None,
    global_test_timeout_sec: int | None = None,
    cleanup: bool = True,
) -> dict:
    """Run Terminal-Bench and write a llm-bench compatible result file."""
    if not terminal_bench_available(terminal_bench_bin):
        return {
            "task": TASK_NAME,
            "error": f"Terminal-Bench command not found: {terminal_bench_bin}",
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    tb_output_root = output_dir / "_tb_runs"
    tb_run_id = _safe_run_id(run_id)
    cmd = _terminal_bench_command(
        terminal_bench_bin=terminal_bench_bin,
        dataset=dataset,
        output_path=tb_output_root,
        run_id=tb_run_id,
        agent=agent,
        model_label=model_label,
        task_ids=task_ids or [],
        n_tasks=n_tasks,
        global_agent_timeout_sec=global_agent_timeout_sec,
        global_test_timeout_sec=global_test_timeout_sec,
        cleanup=cleanup,
    )
    env = _terminal_bench_env(
        base_url=base_url,
        api_key=api_key,
        docker_host=docker_host,
    )
    log_path = output_dir / "terminal_bench.log"
    with log_path.open("w") as log:
        log.write("=== cmd ===\n")
        log.write(shlex.join(cmd) + "\n")
        log.write("=== stdout/stderr ===\n")
        log.flush()
        completed = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=env,
        )

    raw_results = tb_output_root / tb_run_id / "results.json"
    if not raw_results.is_file():
        return {
            "task": TASK_NAME,
            "error": f"rc={completed.returncode}; missing Terminal-Bench results.json",
            "log": str(log_path),
            "raw_run_dir": str(tb_output_root / tb_run_id),
        }

    try:
        result = import_terminal_bench_results(
            raw_results=raw_results,
            output_dir=output_dir,
            metadata={
                "dataset": dataset,
                "agent": agent,
                "model_label": model_label,
                "base_url": base_url,
                "task_ids": task_ids or [],
                "n_tasks": n_tasks,
                "terminal_bench_bin": terminal_bench_bin,
                "raw_run_dir": str(tb_output_root / tb_run_id),
            },
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as exc:
        return {
            "task": TASK_NAME,
            "error": f"could not import Terminal-Bench results: {exc}",
            "log": str(log_path),
            "results_file": str(raw_results),
            "raw_run_dir": str(tb_output_root / tb_run_id),
        }

    result["log"] = str(log_path)
    result["raw_run_dir"] = str(tb_output_root / tb_run_id)
    if completed.returncode != 0:
        result["error"] = f"rc={completed.returncode}"
    return result


def import_terminal_bench_results(
    *,
    raw_results: Path,
    output_dir: Path,
    metadata: dict | None = None,
) -> dict:
    """Convert an existing Terminal-Bench ``results.json`` into llm-bench shape."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(raw_results.read_text())
    samples = _samples_from_results(data)
    metrics = aggregate_terminal_bench_scores(samples)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    results_path = output_dir / f"results_{ts}_{TASK_NAME}.json"
    payload = {
        "results": {TASK_NAME: metrics},
        "samples": samples,
        "metadata": metadata or {},
        "raw_results": str(raw_results),
    }
    results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return {
        "task": TASK_NAME,
        "results_file": str(results_path),
        "samples_file": str(results_path),
        "results": {TASK_NAME: metrics},
        "raw_results": str(raw_results),
    }


def aggregate_terminal_bench_scores(samples: list[dict]) -> dict:
    if not samples:
        return {
            "resolved_rate,none": 0.0,
            "n_tasks,none": 0,
            "n_resolved,none": 0,
            "n_unresolved,none": 0,
            "avg_input_tokens,none": 0.0,
            "avg_output_tokens,none": 0.0,
            "avg_wall_s,none": 0.0,
        }

    n = len(samples)
    n_resolved = sum(1 for sample in samples if sample["resolved"])
    input_tokens = [int(sample.get("input_tokens") or 0) for sample in samples]
    output_tokens = [int(sample.get("output_tokens") or 0) for sample in samples]
    wall_s = [float(sample.get("wall_s") or 0.0) for sample in samples]
    return {
        "resolved_rate,none": round(n_resolved / n, 4),
        "n_tasks,none": n,
        "n_resolved,none": n_resolved,
        "n_unresolved,none": n - n_resolved,
        "avg_input_tokens,none": round(sum(input_tokens) / n, 1),
        "avg_output_tokens,none": round(sum(output_tokens) / n, 1),
        "avg_wall_s,none": round(sum(wall_s) / n, 1),
    }


def _terminal_bench_command(
    *,
    terminal_bench_bin: str,
    dataset: str,
    output_path: Path,
    run_id: str,
    agent: str,
    model_label: str,
    task_ids: list[str],
    n_tasks: int | None,
    global_agent_timeout_sec: int | None,
    global_test_timeout_sec: int | None,
    cleanup: bool,
) -> list[str]:
    cmd = shlex.split(terminal_bench_bin) + [
        "run",
        "--dataset",
        dataset,
        "--output-path",
        str(output_path),
        "--run-id",
        run_id,
        "--agent",
        agent,
        "--model",
        model_label,
        "--n-concurrent",
        "1",
        "--no-upload-results",
    ]
    for task_id in task_ids:
        cmd.extend(["--task-id", task_id])
    if n_tasks is not None:
        cmd.extend(["--n-tasks", str(n_tasks)])
    if global_agent_timeout_sec is not None:
        cmd.extend(["--global-agent-timeout-sec", str(global_agent_timeout_sec)])
    if global_test_timeout_sec is not None:
        cmd.extend(["--global-test-timeout-sec", str(global_test_timeout_sec)])
    cmd.append("--cleanup" if cleanup else "--no-cleanup")
    return cmd


def _terminal_bench_env(
    *,
    base_url: str,
    api_key: str | None,
    docker_host: str | None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["OPENAI_API_BASE"] = base_url.rstrip("/")
    env["OPENAI_API_KEY"] = api_key or env.get("OPENAI_API_KEY") or "sk-local"
    if docker_host:
        env["DOCKER_HOST"] = docker_host
    elif "DOCKER_HOST" not in env:
        inferred = _infer_docker_host()
        if inferred:
            env["DOCKER_HOST"] = inferred
    return env


def _infer_docker_host() -> str:
    colima = Path.home() / ".colima" / "default" / "docker.sock"
    if colima.exists():
        return f"unix://{colima}"
    return ""


def _samples_from_results(data: dict) -> list[dict]:
    rows = data.get("results", [])
    if not isinstance(rows, list):
        rows = []
    return [_sample_from_row(row) for row in rows if isinstance(row, dict)]


def _sample_from_row(row: dict) -> dict:
    return {
        "task_id": str(row.get("task_id", "")),
        "resolved": bool(row.get("is_resolved")),
        "failure_mode": str(row.get("failure_mode") or ""),
        "input_tokens": int(row.get("total_input_tokens") or 0),
        "output_tokens": int(row.get("total_output_tokens") or 0),
        "wall_s": _elapsed_seconds(
            str(row.get("trial_started_at") or ""),
            str(row.get("trial_ended_at") or ""),
        ),
        "parser_results": row.get("parser_results") or {},
    }


def _elapsed_seconds(start: str, end: str) -> float:
    try:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
    except ValueError:
        return 0.0
    return round((end_dt - start_dt).total_seconds(), 1)


def _safe_run_id(run_id: str | None = None) -> str:
    value = run_id or datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    safe = re.sub(r"[^a-z0-9_-]+", "-", value.lower()).strip("-_")
    return safe or "terminal-bench-run"
