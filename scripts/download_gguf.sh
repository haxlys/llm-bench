#!/usr/bin/env bash
# Downloads Gemma 4 26B-A4B-it GGUF Q8_0 from unsloth.
# Default destination: ~/models/gguf/. Override with $LLM_BENCH_GGUF_DIR.

set -euo pipefail

DEST="${LLM_BENCH_GGUF_DIR:-$HOME/models/gguf}"
REPO="unsloth/gemma-4-26B-A4B-it-GGUF"
PATTERN="*Q8_0*.gguf"

mkdir -p "$DEST"
echo "→ Downloading $REPO ($PATTERN) to $DEST"

if command -v hf >/dev/null 2>&1; then
    hf download "$REPO" --include "$PATTERN" --local-dir "$DEST"
elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$REPO" --include "$PATTERN" --local-dir "$DEST"
else
    echo "ERROR: Install huggingface_hub: uv tool install huggingface_hub" >&2
    exit 1
fi

echo
echo "→ Files in $DEST:"
ls -lh "$DEST"/*.gguf 2>/dev/null || echo "  (no .gguf files found — check pattern)"
