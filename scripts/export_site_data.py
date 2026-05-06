#!/usr/bin/env python3
"""Export benchmark website JSON from committed results."""

from __future__ import annotations

import argparse
from pathlib import Path

from llm_bench.site_data import write_site_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, default=Path("site/src/data/benchmarks.json"))
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else args.repo_root / args.out
    write_site_data(args.repo_root, out)
    print(out)


if __name__ == "__main__":
    main()
