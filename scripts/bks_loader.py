#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple


def load_bks_from_benchmarks(json_path: Path) -> Dict[str, float]:
    if not json_path.exists():
        return {}
    data = json.loads(json_path.read_text())
    return {k: float(v) for k, v in data.items()}


def parse_sol_cost(sol_path: Path) -> Optional[float]:
    if not sol_path.exists():
        return None
    for line in sol_path.read_text().splitlines():
        line = line.strip()
        if line.upper().startswith("COST"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return float(parts[1])
                except ValueError:
                    return None
    return None


def load_bks_from_solutions(folder: Path) -> Dict[str, Tuple[float, str]]:
    """
    Load BKS costs and sol text from *.sol files in a folder.
    Returns mapping: instance.vrp -> (cost, sol_text).
    """
    out: Dict[str, Tuple[float, str]] = {}
    for sol_file in folder.glob("*.sol"):
        cost = parse_sol_cost(sol_file)
        if cost is None:
            continue
        out[sol_file.with_suffix(".vrp").name] = (cost, sol_file.read_text())
    return out


def load_bks(folder: Path) -> Dict[str, Tuple[float, str]]:
    """
    Try benchmarks.json first; fall back to .sol files with Cost lines.
    Returns mapping: instance name -> (bks_cost, bks_sol_text or "").
    """
    benchmarks = load_bks_from_benchmarks(folder / "benchmarks.json")
    sol_map = load_bks_from_solutions(folder)
    merged: Dict[str, Tuple[float, str]] = {}
    for inst, cost in benchmarks.items():
        merged[inst] = (cost, sol_map.get(inst, ("", ""))[1])
    for inst, (cost, text) in sol_map.items():
        if inst not in merged:
            merged[inst] = (cost, text)
    return merged


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    args = parser.parse_args()
    data = load_bks(Path(args.folder))
    for k, (c, _) in data.items():
        print(k, c)
