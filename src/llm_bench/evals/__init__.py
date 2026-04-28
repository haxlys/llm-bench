"""Multi-dimensional eval orchestration (lm-eval-harness wrapper + ModelServer)."""

from llm_bench.evals.server import ModelServer
from llm_bench.evals.suites import SUITES, smoke_suite, full_suite

__all__ = ["ModelServer", "SUITES", "smoke_suite", "full_suite"]
