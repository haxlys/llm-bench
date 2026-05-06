"""Tests for benchmark subprocess helpers."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from llm_bench.runners.base import run_with_time


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_run_with_time_timeout_kills_child_process_group(tmp_path):
    child_pid_file = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        f"open({str(child_pid_file)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(30)\n"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_with_time([sys.executable, "-c", script], timeout_s=0.5)

    child_pid = int(child_pid_file.read_text())
    deadline = time.time() + 2
    while time.time() < deadline and _pid_alive(child_pid):
        time.sleep(0.05)

    if _pid_alive(child_pid):
        os.kill(child_pid, signal.SIGKILL)
        pytest.fail("timed-out benchmark left a child process running")


def test_run_with_time_returns_captured_stdout_text():
    stdout, stderr, _wall_s, _peak_mem_gb = run_with_time(
        [sys.executable, "-c", "print('{\"ok\": true}')"],
        timeout_s=5,
    )

    assert isinstance(stdout, str)
    assert '{"ok": true}' in stdout
    assert isinstance(stderr, str)
