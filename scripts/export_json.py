#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from gen3.v3.io import read_vrp_dimacs


def export_run(conn: sqlite3.Connection, run_id: int) -> dict:
    cur = conn.cursor()
    cur.execute("SELECT dataset, distance_type, time_limit_s, notes FROM runs WHERE run_id=?", (run_id,))
    row = cur.fetchone()
    if row is None:
        raise ValueError(f"run_id {run_id} not found")
    dataset, distance_type, time_limit_s, notes = row
    cur.execute(
        """
        SELECT instance, solver_cost, solver_sol_text, time_ms, valid
        FROM solver_instances
        WHERE run_id=?
        """,
        (run_id,),
    )
    solver_rows = cur.fetchall()
    cur.execute("SELECT instance, bks_cost, bks_sol_text FROM bks")
    bks_rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    instances = []
    for inst, solver_cost, solver_sol, time_ms, valid in solver_rows:
        bks_cost, bks_sol = bks_rows.get(inst, (None, ""))
        gap_pct = None
        if solver_cost is not None and bks_cost is not None and bks_cost != 0:
            gap_pct = 100.0 * (solver_cost - bks_cost) / bks_cost
        coords = {}
        demands = {}
        depot = "0"
        capacity = None
        vrp_path = Path(dataset) / inst
        try:
            state = read_vrp_dimacs(vrp_path, distance_type=distance_type)
            coords = {str(i): (state.x[i], state.y[i]) for i in range(state.N + 1)}
            demands = {str(i): state.demand[i] for i in range(state.N + 1)}
            capacity = state.Q
        except Exception:
            pass
        instances.append(
            {
                "instance": inst,
                "solver_cost": solver_cost,
                "solver_sol": solver_sol,
                "bks_cost": bks_cost,
                "bks_sol": bks_sol,
                "gap_pct": gap_pct,
                "valid": bool(valid),
                "time_ms": time_ms,
                "coords": coords,
                "demands": demands,
                "depot": depot,
                "capacity": capacity,
            }
        )
    return {
        "run_id": run_id,
        "dataset": dataset,
        "distance_type": distance_type,
        "time_limit_s": time_limit_s,
        "notes": notes,
        "instances": instances,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export run results to JSON.")
    parser.add_argument("--db", required=True, help="SQLite DB path")
    parser.add_argument("--run-id", type=int, required=True, help="Run ID to export")
    parser.add_argument("--out", required=True, help="Output JSON file")
    args = parser.parse_args(argv)

    conn = sqlite3.connect(args.db)
    data = export_run(conn, args.run_id)
    Path(args.out).write_text(json.dumps(data, indent=2))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
