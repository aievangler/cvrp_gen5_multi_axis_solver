#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import sqlite3
import sys
import time
from pathlib import Path

from gen3.v3 import (
    CleanupConfig,
    InitConfig,
    LSConfig,
    STNConfig,
    SoftConfig,
    build_initial_solution,
    build_stn,
    compute_total_cost,
    read_vrp_dimacs,
    snapshot_state,
    solve_with_state,
    validate_solution,
    emit_solution_for_controller,
)


def ensure_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset TEXT,
            distance_type INTEGER,
            time_limit_s REAL,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS solver_instances (
            run_id INTEGER,
            instance TEXT,
            solver_cost REAL,
            solver_sol_text TEXT,
            time_ms REAL,
            valid INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS bks (
            instance TEXT PRIMARY KEY,
            bks_cost REAL,
            bks_sol_text TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS instances (
            instance TEXT PRIMARY KEY,
            dataset TEXT
        )
        """
    )
    conn.commit()


def emit_solution_text(state) -> str:
    buf = io.StringIO()
    emit_solution_for_controller(state, out=buf)
    return buf.getvalue()


def run_one_instance(
    vrp_path: Path,
    distance_type: int,
    time_limit: float,
    soft_run_enabled: bool = False,
    ls_max_passes: int = 5,
    ls_max_candidates: int = 16,
) -> tuple[float | None, str, float, int]:
    """
    Run solver on a single instance.
    Returns (cost, sol_text, time_ms, valid_flag).
    """
    start = time.time()
    try:
        state = read_vrp_dimacs(vrp_path, distance_type=distance_type)
        # Build STN and initial solution explicitly (solve_with_state would do this too)
        # but we rely on on_improve to capture best snapshot.
        best_snap = None
        best_cost = float("inf")

        def on_improve(s):
            nonlocal best_snap, best_cost
            best_cost = s.current_cost
            best_snap = snapshot_state(s)

        solve_with_state(
            state=state,
            stn_config=STNConfig(),
            init_config=InitConfig(),
            soft_config=SoftConfig(soft_run_enabled=soft_run_enabled),
            cleanup_config=CleanupConfig(),
            ls_config=LSConfig(ls_max_passes=ls_max_passes, max_candidates=ls_max_candidates),
            max_plateau=3,
            time_limit_sec=time_limit,
            on_improve=on_improve,
        )
        # If we captured a better snapshot, restore it.
        if best_snap is not None:
            state = best_snap
        validate_solution(state)
        cost = compute_total_cost(state)
        sol_text = emit_solution_text(state)
        valid = 1
    except Exception as exc:  # noqa: BLE001
        cost = None
        sol_text = f"ERROR: {exc}"
        valid = 0
    time_ms = (time.time() - start) * 1000.0
    return cost, sol_text, time_ms, valid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gen3 v3 solver over a dataset and log to SQLite.")
    parser.add_argument("--dataset", required=True, help="Folder containing .vrp files")
    parser.add_argument("--distance-type", type=int, required=True, help="0=EUC, 1=EUC rounded, 2=EXPLICIT")
    parser.add_argument("--time-limit", type=float, required=True, help="time limit per instance (seconds)")
    parser.add_argument("--db", required=True, help="SQLite DB path to write results into")
    parser.add_argument("--notes", default="", help="Optional notes for the run")
    parser.add_argument("--bks-folder", default=None, help="Folder containing BKS .sol files or benchmarks.json")
    parser.add_argument("--soft-run-enabled", action="store_true", help="Enable soft_run (relaxed capacity) phase")
    parser.add_argument("--ls-max-passes", type=int, default=5, help="Max LS passes per call")
    parser.add_argument("--ls-max-candidates", type=int, default=16, help="Max triad candidate edges per node")
    args = parser.parse_args(argv)

    dataset_dir = Path(args.dataset)
    vrp_files = sorted(dataset_dir.glob("*.vrp"))
    if not vrp_files:
        print(f"No .vrp files found in {dataset_dir}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    ensure_schema(conn)
    # Load and upsert BKS if provided
    if args.bks_folder:
        from scripts.bks_loader import load_bks

        bks_map = load_bks(Path(args.bks_folder))
        cur = conn.cursor()
        for inst, (cost, text) in bks_map.items():
            cur.execute(
                "INSERT OR REPLACE INTO bks(instance, bks_cost, bks_sol_text) VALUES (?, ?, ?)",
                (inst, cost, text),
            )
        conn.commit()

    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs(dataset, distance_type, time_limit_s, notes) VALUES (?, ?, ?, ?)",
        (str(dataset_dir), args.distance_type, args.time_limit, args.notes),
    )
    run_id = cur.lastrowid
    conn.commit()

    # Upsert instances list for this dataset
    cur.executemany(
        "INSERT OR IGNORE INTO instances(instance, dataset) VALUES (?, ?)",
        [(vf.name, str(dataset_dir)) for vf in vrp_files],
    )
    conn.commit()

    for vrp_path in vrp_files:
        cost, sol_text, time_ms, valid = run_one_instance(
            vrp_path, distance_type=args.distance_type, time_limit=args.time_limit
        )
        cur.execute(
            """
            INSERT INTO solver_instances(run_id, instance, solver_cost, solver_sol_text, time_ms, valid)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (run_id, vrp_path.name, cost, sol_text, time_ms, valid),
        )
        conn.commit()
        print(f"[run {run_id}] {vrp_path.name}: cost={cost} valid={valid} time_ms={time_ms:.1f}", file=sys.stderr)

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
