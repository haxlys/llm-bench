"""Tests for the DS4 speed runner."""

from __future__ import annotations

from llm_bench.runners.ds4_runner import _parse_single_row


def test_parse_single_ds4_bench_row():
    row = _parse_single_row(
        "ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes\n"
        "256,256,123.45,128,34.56,789\n"
    )

    assert row["prefill_tps"] == "123.45"
    assert row["gen_tps"] == "34.56"
