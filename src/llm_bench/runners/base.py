"""Common types and helpers for benchmark runners."""

from __future__ import annotations

import json
import re
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
    raw: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_TIME_REAL_RE = re.compile(r"^\s*([\d.]+)\s+real", re.MULTILINE)
_TIME_MAXRSS_RE = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.MULTILINE)


def run_with_time(cmd: list[str], env: dict | None = None) -> tuple[str, str, float, float]:
    """Run cmd wrapped with /usr/bin/time -l. Returns (stdout, stderr, wall_s, peak_mem_gb).

    /usr/bin/time -l is BSD time on macOS — emits 'maximum resident set size' in bytes.
    """
    wrapped = ["/usr/bin/time", "-l", *cmd]
    t0 = time.perf_counter()
    proc = subprocess.run(wrapped, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    stderr = proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed (rc={proc.returncode}): {' '.join(cmd[:3])}...\n"
            f"stderr tail:\n{stderr[-2000:]}"
        )

    real_match = _TIME_REAL_RE.search(stderr)
    if real_match:
        wall = float(real_match.group(1))
    rss_match = _TIME_MAXRSS_RE.search(stderr)
    peak_mem_gb = (int(rss_match.group(1)) / (1024 ** 3)) if rss_match else 0.0
    return proc.stdout, stderr, wall, peak_mem_gb


def write_raw(result: BenchResult, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = result.ts.replace(":", "").replace("-", "")
    fname = f"{safe_ts}_{result.model_id}_{result.fmt}_{result.quant}_{result.scenario}_r{result.run_idx}.json"
    path = raw_dir / fname
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return path
