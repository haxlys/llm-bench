"""Aggregate MTPLX MTP-on/off raw results into speedup rows."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean


FIELDNAMES = [
    "pair_key",
    "scenario",
    "mtp_variant",
    "ar_variant",
    "mtp_runs",
    "ar_runs",
    "mtp_tg_tps_mean",
    "ar_tg_tps_mean",
    "speedup",
    "mtp_peak_mem_gb_mean",
    "ar_peak_mem_gb_mean",
    "verify_ms_per_call_mean",
    "accept_d1_mean",
    "accept_d2_mean",
    "accept_d3_mean",
]


@dataclass
class _Group:
    mtp_variant: str = ""
    ar_variant: str = ""
    mtp_tg: list[float] = field(default_factory=list)
    ar_tg: list[float] = field(default_factory=list)
    mtp_mem: list[float] = field(default_factory=list)
    ar_mem: list[float] = field(default_factory=list)
    verify_ms: list[float] = field(default_factory=list)
    accept_by_depth: list[list[float]] = field(default_factory=list)


def write_speedup_report(raw_dir: Path, out_csv: Path) -> Path:
    rows = build_speedup_rows(raw_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return out_csv


def build_speedup_rows(raw_dir: Path) -> list[dict[str, str]]:
    groups: dict[tuple[str, str], _Group] = defaultdict(_Group)
    for data in _load_raw(raw_dir):
        if data.get("backend") != "mtplx":
            continue
        key = str(data.get("variant_key") or "")
        mode = str(data.get("generation_mode") or "")
        pair_key = _pair_key(key)
        if pair_key is None or mode not in {"mtp", "ar"}:
            continue
        group = groups[(pair_key, str(data.get("scenario") or ""))]
        if mode == "mtp":
            group.mtp_variant = key
            group.mtp_tg.append(float(data.get("tg_tps") or 0.0))
            group.mtp_mem.append(float(data.get("peak_mem_gb") or 0.0))
            raw = data.get("raw") if isinstance(data.get("raw"), dict) else {}
            verify = raw.get("verify_ms_per_call")
            if verify is not None:
                group.verify_ms.append(float(verify))
            rates = raw.get("acceptance_rates_by_depth")
            if isinstance(rates, list):
                group.accept_by_depth.append([float(x) for x in rates])
        else:
            group.ar_variant = key
            group.ar_tg.append(float(data.get("tg_tps") or 0.0))
            group.ar_mem.append(float(data.get("peak_mem_gb") or 0.0))

    out: list[dict[str, str]] = []
    for (pair_key, scenario), group in sorted(groups.items()):
        if not group.mtp_tg or not group.ar_tg:
            continue
        mtp_tg = mean(group.mtp_tg)
        ar_tg = mean(group.ar_tg)
        row = {
            "pair_key": pair_key,
            "scenario": scenario,
            "mtp_variant": group.mtp_variant,
            "ar_variant": group.ar_variant,
            "mtp_runs": str(len(group.mtp_tg)),
            "ar_runs": str(len(group.ar_tg)),
            "mtp_tg_tps_mean": _fmt(mtp_tg),
            "ar_tg_tps_mean": _fmt(ar_tg),
            "speedup": _fmt(mtp_tg / ar_tg if ar_tg else 0.0),
            "mtp_peak_mem_gb_mean": _fmt(mean(group.mtp_mem)),
            "ar_peak_mem_gb_mean": _fmt(mean(group.ar_mem)),
            "verify_ms_per_call_mean": _fmt(mean(group.verify_ms)) if group.verify_ms else "",
        }
        for idx in range(3):
            values = [rates[idx] for rates in group.accept_by_depth if len(rates) > idx]
            row[f"accept_d{idx + 1}_mean"] = _fmt(mean(values)) if values else ""
        out.append(row)
    return out


def _load_raw(raw_dir: Path) -> list[dict]:
    if not raw_dir.exists():
        return []
    out = []
    for p in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict):
            out.append(data)
    return out


def _pair_key(variant_key: str) -> str | None:
    if variant_key.endswith("-mtp"):
        return variant_key[:-4]
    if variant_key.endswith("-ar"):
        return variant_key[:-3]
    return None


def _fmt(value: float) -> str:
    return f"{value:.3f}"
