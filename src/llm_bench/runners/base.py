"""Common types and helpers for benchmark runners."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    name: str
    n_prompt: int
    n_gen: int


@dataclass
class BenchResult:
    model_id: str
    fmt: str  # "mlx" | "gguf"
    quant: str
    scenario: str
    n_prompt: int
    n_gen: int
    pp_tps: float
    tg_tps: float
    peak_mem_gb: float
    wall_s: float
    run_idx: int
    ts: str
    bench_version: str = ""    # stamped from llm_bench.BENCH_VERSION
    variant_key: str = ""      # registry key (e.g. "26B-MoE-mlx-8bit")
    backend: str = ""
    artifact_type: str = ""
    generation_mode: str = ""
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_TIME_REAL_RE = re.compile(r"^\s*([\d.]+)\s+real", re.MULTILINE)
_TIME_MAXRSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.MULTILINE)

# Hard ceiling per benchmark invocation. The longest single scenario
# (8K prefill + 512 gen on 31B GGUF) tops out around ~6 minutes; 30 minutes
# is generous but bounded enough to keep an overnight matrix from being held
# hostage by a single hung process.
DEFAULT_TIMEOUT_S = 30 * 60


def run_with_time(
    cmd: list[str],
    env: dict | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    check: bool = True,
) -> tuple[str, str, float, float]:
    """Run cmd wrapped with /usr/bin/time -l. Returns (stdout, stderr, wall_s, peak_mem_gb).

    /usr/bin/time -l is BSD time on macOS — emits 'maximum resident set size' in bytes.
    Raises subprocess.TimeoutExpired (parent kills the process group) if the wrapped
    command exceeds timeout_s.
    """
    wrapped = ["/usr/bin/time", "-l", *cmd]
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        wrapped,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        # Kill the whole process group. `/usr/bin/time` is only the wrapper;
        # the actual model process is its child and can otherwise survive.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        raise subprocess.TimeoutExpired(
            cmd=wrapped,
            timeout=timeout_s,
            output=stdout,
            stderr=stderr,
        ) from e
    wall = time.perf_counter() - t0
    if proc.returncode != 0 and check:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd[:3])}...\n"
            f"stdout tail:\n{stdout[-2000:]}\n"
            f"stderr tail:\n{stderr[-2000:]}"
        )

    real_match = _TIME_REAL_RE.search(stderr)
    if real_match:
        wall = float(real_match.group(1))
    rss_match = _TIME_MAXRSS_RE.search(stderr)
    peak_mem_gb = (int(rss_match.group(1)) / (1024 ** 3)) if rss_match else 0.0
    return stdout, stderr, wall, peak_mem_gb


def write_raw(result: BenchResult, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = result.ts.replace(":", "").replace("-", "")
    fname = f"{safe_ts}_{result.model_id}_{result.fmt}_{result.quant}_{result.scenario}_r{result.run_idx}.json"
    path = raw_dir / fname
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return path
