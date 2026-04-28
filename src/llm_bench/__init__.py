"""MLX vs GGUF inference benchmark for Apple Silicon."""

__version__ = "0.3.0"

# Bump whenever the measurement methodology changes (new scenarios, different
# warmup, changed memory accounting, etc). Stamped into every BenchResult so
# the manifest can decide when historical data is comparable.
BENCH_VERSION = "0.3"
