"""Collect responses from MLX and GGUF for each standard prompt and compute divergence.

Output: results/quality_<ts>.json — list of dicts with cos_sim, surface stats.
Optional: pip install -e '.[quality]' for sentence-transformers; otherwise only
length/char-overlap surface metrics are computed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_bench.prompts import QUALITY_PROMPTS  # noqa: E402

MAX_TOKENS = 512


def _gen_mlx(model_path: str, prompt: str) -> str:
    cmd = [
        sys.executable, "-m", "mlx_lm", "generate",
        "--model", model_path, "--prompt", prompt,
        "--max-tokens", str(MAX_TOKENS), "--temp", "0.0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"mlx_lm failed: {proc.stderr[-1000:]}")
    # mlx_lm prints prompt then "==========" then generation then "==========" then stats
    parts = proc.stdout.split("==========")
    return parts[1].strip() if len(parts) >= 3 else proc.stdout.strip()


def _gen_gguf(gguf_path: str, prompt: str, n_gpu_layers: int = 999) -> str:
    if not shutil.which("llama-cli"):
        raise RuntimeError("llama-cli not on PATH (brew install llama.cpp)")
    cmd = [
        "llama-cli", "-m", gguf_path, "-p", prompt,
        "-n", str(MAX_TOKENS), "-ngl", str(n_gpu_layers),
        "--temp", "0.0", "--no-cnv", "-no-display-prompt",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"llama-cli failed: {proc.stderr[-1000:]}")
    return proc.stdout.strip()


def _surface_stats(a: str, b: str) -> dict:
    sa, sb = set(a.split()), set(b.split())
    jacc = len(sa & sb) / max(1, len(sa | sb))
    return {
        "len_mlx": len(a),
        "len_gguf": len(b),
        "len_ratio": (len(a) / len(b)) if b else float("inf"),
        "word_jaccard": round(jacc, 3),
    }


def _embed_cos(texts_a: list[str], texts_b: list[str]) -> list[float]:
    try:
        from sentence_transformers import SentenceTransformer, util
    except ImportError:
        return [float("nan")] * len(texts_a)
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    ea = model.encode(texts_a, convert_to_tensor=True, normalize_embeddings=True)
    eb = model.encode(texts_b, convert_to_tensor=True, normalize_embeddings=True)
    return [float(util.cos_sim(ea[i], eb[i])) for i in range(len(texts_a))]


@click.command()
@click.option("--mlx-model", default="lmstudio-community/gemma-4-26B-A4B-it-MLX-8bit")
@click.option("--gguf-model", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--limit", type=int, default=None, help="Limit number of prompts for quick test")
def main(mlx_model: str, gguf_model: str, limit: int | None):
    prompts = QUALITY_PROMPTS[:limit] if limit else QUALITY_PROMPTS
    rows: list[dict] = []
    for i, p in enumerate(prompts, 1):
        click.echo(f"[{i}/{len(prompts)}] {p['id']} ", nl=False)
        t0 = time.perf_counter()
        try:
            mlx_resp = _gen_mlx(mlx_model, p["prompt"])
        except Exception as e:
            click.echo(f"MLX FAIL: {e}", err=True); continue
        try:
            gguf_resp = _gen_gguf(gguf_model, p["prompt"])
        except Exception as e:
            click.echo(f"GGUF FAIL: {e}", err=True); continue
        dt = time.perf_counter() - t0
        row = {
            "id": p["id"], "lang": p["lang"], "category": p["category"],
            "prompt": p["prompt"],
            "mlx_response": mlx_resp, "gguf_response": gguf_resp,
            **_surface_stats(mlx_resp, gguf_resp),
        }
        rows.append(row)
        click.echo(f"({dt:.1f}s, jacc={row['word_jaccard']:.2f})")

    click.echo("→ computing embedding similarities…")
    cos_sims = _embed_cos([r["mlx_response"] for r in rows],
                          [r["gguf_response"] for r in rows])
    for r, c in zip(rows, cos_sims):
        r["cos_sim"] = round(c, 4) if c == c else None  # NaN → None

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / f"quality_{ts}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    click.echo(f"→ wrote {out}")


if __name__ == "__main__":
    main()
