"""Default scenario matrix."""

from __future__ import annotations

from llm_bench.runners.base import Scenario


def default_scenarios() -> list[Scenario]:
    """Cross product of prefill sizes × generation sizes.

    Skips degenerate combos (e.g. 8K prefill with 1K gen takes too long for
    a 1st-pass smoke run; we still include it but the CLI lets you slice).
    """
    prefills = [256, 1024, 4096, 8192]
    gens = [128, 512]
    out: list[Scenario] = []
    for p in prefills:
        for g in gens:
            out.append(Scenario(name=f"p{p}_g{g}", n_prompt=p, n_gen=g))
    return out


def smoke_scenarios() -> list[Scenario]:
    return [Scenario(name="p256_g128", n_prompt=256, n_gen=128)]
