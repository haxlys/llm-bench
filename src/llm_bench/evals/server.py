"""Spawn a per-model OpenAI-compatible inference server (mlx_lm or llama.cpp)."""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests


class ModelServer:
    """Context manager: boots model server, waits ready, terminates on exit.

    Usage:
        with ModelServer(fmt='mlx', model_path='lmstudio-community/...') as base_url:
            # base_url = 'http://127.0.0.1:9090/v1'
            ...
    """

    def __init__(
        self,
        fmt: str,                     # "mlx" | "gguf"
        model_path: str,              # HF id or local file
        port: int = 9090,
        host: str = "127.0.0.1",
        n_gpu_layers: int = 999,
        boot_timeout_s: int = 240,
        log_file: Path | None = None,
    ):
        if fmt not in ("mlx", "gguf"):
            raise ValueError(f"unknown fmt: {fmt}")
        self.fmt = fmt
        self.model_path = model_path
        self.port = port
        self.host = host
        self.n_gpu_layers = n_gpu_layers
        self.boot_timeout_s = boot_timeout_s
        self.log_file = log_file
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _build_cmd(self) -> list[str]:
        if self.fmt == "mlx":
            return [
                sys.executable, "-m", "mlx_lm", "server",
                "--model", self.model_path,
                "--host", self.host, "--port", str(self.port),
                "--temp", "0.0",
            ]
        if not shutil.which("llama-server"):
            raise RuntimeError("llama-server not found (brew install llama.cpp)")
        return [
            "llama-server",
            "-m", self.model_path,
            "--host", self.host, "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers),
            "--jinja",                        # required for Gemma 4 chat template
            "-c", "16384",                    # context size; raise for long evals
        ]

    def _is_ready(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/models", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def __enter__(self) -> str:
        cmd = self._build_cmd()
        log_target = open(self.log_file, "w") if self.log_file else subprocess.DEVNULL
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=log_target, stderr=subprocess.STDOUT,
            )
        finally:
            # Popen dup'd the fd; the parent's handle is no longer needed.
            if self.log_file:
                log_target.close()

        deadline = time.time() + self.boot_timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"server died during boot (rc={self.proc.returncode}); "
                    f"check log: {self.log_file}"
                )
            if self._is_ready():
                return self.base_url
            time.sleep(2)
        self._terminate()
        raise TimeoutError(f"server not ready after {self.boot_timeout_s}s")

    def __exit__(self, exc_type, exc, tb):
        self._terminate()

    def _terminate(self) -> None:
        if not self.proc:
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)
        self.proc = None
