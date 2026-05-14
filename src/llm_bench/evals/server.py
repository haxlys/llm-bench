"""OpenAI-compatible eval server lifecycle adapters."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

# mlx_lm.server defaults --max-tokens to 512, which is too low for our eval
# matrix: when generation hits that ceiling without emitting EOS, the
# response comes back with finish_reason="length" and content="" because the
# chat-template terminator never appeared. Setting a high default here lets
# typical task outputs (gsm8k CoT, MBPP code) finish naturally with
# finish_reason="stop" so content is preserved. Reproducer 2026-04-29.
MLX_DEFAULT_MAX_TOKENS = 8192


class ModelServer:
    """Context manager for local model servers or existing endpoints.

    Usage:
        with ModelServer(fmt='mlx', model_path='lmstudio-community/...') as base_url:
            # base_url = 'http://127.0.0.1:9090/v1'
            ...

        with ModelServer(fmt='api', backend='openai-compatible',
                         artifact_type='endpoint',
                         model_path='https://host/v1') as base_url:
            # base_url = 'https://host/v1' and no subprocess is spawned
            ...
    """

    def __init__(
        self,
        fmt: str,                     # legacy label: "mlx" | "gguf" | "api"
        model_path: str,              # HF id or local file
        backend: str | None = None,
        artifact_type: str | None = None,
        port: int = 9090,
        host: str = "127.0.0.1",
        n_gpu_layers: int = 999,
        context_size: int = 16384,
        boot_timeout_s: int = 240,
        log_file: Path | None = None,
        chat_template_args: dict[str, Any] | None = None,
        server_args: list[str] | None = None,
        runtime_root: str = "",
    ):
        self.fmt = fmt
        self.backend = backend or fmt
        self.artifact_type = artifact_type or _default_artifact_type(fmt)
        if self.backend not in ("mlx", "gguf", "ds4", "openai-compatible"):
            raise ValueError(f"unknown backend: {self.backend}")
        self.model_path = model_path
        self.port = port
        self.host = host
        self.n_gpu_layers = n_gpu_layers
        self.context_size = context_size
        self.boot_timeout_s = boot_timeout_s
        self.log_file = log_file
        self.chat_template_args = chat_template_args
        self.server_args = server_args or []
        self.runtime_root = runtime_root
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        if self.artifact_type == "endpoint":
            return _normalize_endpoint_base_url(self.model_path)
        return f"http://{self.host}:{self.port}/v1"

    def _build_cmd(self) -> list[str]:
        if self.backend == "mlx":
            cmd = [
                sys.executable, "-m", "mlx_lm", "server",
                "--model", self.model_path,
                "--host", self.host, "--port", str(self.port),
                "--temp", "0.0",
                "--max-tokens", str(MLX_DEFAULT_MAX_TOKENS),
            ]
            # Qwen thinking templates can otherwise put the answer in the
            # OpenAI `reasoning` field and leave `content` empty/truncated.
            # Passing this as a server default keeps deterministic eval
            # extraction stable while callers may still override it.
            chat_template_args = (
                {"enable_thinking": False}
                if self.chat_template_args is None
                else self.chat_template_args
            )
            if chat_template_args:
                cmd.extend([
                    "--chat-template-args",
                    json.dumps(chat_template_args, separators=(",", ":")),
                ])
            if self.server_args:
                cmd.extend(self.server_args)
            return cmd
        if self.backend == "ds4":
            ds4_server = _find_ds4_binary("ds4-server", self.model_path, self.runtime_root)
            cmd = [
                ds4_server,
                "-m", self.model_path,
                "--host", self.host, "--port", str(self.port),
                "-c", str(self.context_size),
                "-n", str(MLX_DEFAULT_MAX_TOKENS),
            ]
            if self.server_args:
                cmd.extend(self.server_args)
            return cmd
        if self.backend != "gguf":
            raise RuntimeError(
                f"backend '{self.backend}' does not have a local server command"
            )
        if not shutil.which("llama-server"):
            raise RuntimeError("llama-server not found (brew install llama.cpp)")
        cmd = [
            "llama-server",
            "-m", self.model_path,
            "--host", self.host, "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers),
            "--jinja",                        # required for Gemma 4 chat template
            "--reasoning", "off",
            "--chat-template-kwargs",
            json.dumps({"enable_thinking": False}, separators=(",", ":")),
            "-c", str(self.context_size),
        ]
        if self.server_args:
            cmd.extend(self.server_args)
        return cmd

    def _subprocess_cwd(self) -> str | None:
        if self.backend == "ds4":
            return str(_resolve_ds4_root(self.model_path, self.runtime_root))
        return None

    def _is_ready(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/models", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def __enter__(self) -> str:
        if self.artifact_type == "endpoint":
            return self.base_url
        cmd = self._build_cmd()
        log_target = open(self.log_file, "w") if self.log_file else subprocess.DEVNULL
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=log_target, stderr=subprocess.STDOUT,
                cwd=self._subprocess_cwd(),
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


def _default_artifact_type(fmt: str) -> str:
    if fmt == "api":
        return "endpoint"
    if fmt == "gguf":
        return "gguf_file"
    return "hf_repo"


def _find_ds4_binary(name: str, model_path: str, runtime_root: str = "") -> str:
    ds4_root = _resolve_ds4_root(model_path, runtime_root)
    local = ds4_root / name
    if local.is_file():
        return str(local)
    found = shutil.which(name)
    if found:
        return found
    raise RuntimeError(f"{name} not found. Build ds4 or put {name} on PATH.")


def _resolve_ds4_root(model_path: str, runtime_root: str = "") -> Path:
    if runtime_root:
        return Path(runtime_root).expanduser().resolve()
    return Path(model_path).expanduser().resolve().parent.parent


def _normalize_endpoint_base_url(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"
