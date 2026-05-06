"""Multi-dimensional eval orchestration (lm-eval-harness wrapper + ModelServer)."""

from llm_bench.evals.server import ModelServer
from llm_bench.evals.suites import SUITES, full_suite, long_suite, smoke_suite

__all__ = ["ModelServer", "SUITES", "smoke_suite", "long_suite", "full_suite"]
