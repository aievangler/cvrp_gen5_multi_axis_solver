#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import functools
import json
import sqlite3
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from cvrp import parser, evaluator
from cvrp.distance import build_distance
from cvrp.solution_io import parse_solution
from cvrp.evaluator import normalize_routes
from v5_solver.core.instance_adapter import load_instance
from v5_solver.core.repo_builder import build_repo
from v5_solver.core.config import DEFAULT_CONFIG
from v5_solver.core.v2_wrapper import run_v2_with_snapshots

app = FastAPI(title="V5 Viewer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INST_ROOT = ROOT / "data" / "instances"
SOL_ROOT = ROOT / "out" / "v5"
LOG_DB = ROOT / "runs.sqlite"
VIEWER_DIR = ROOT / "viewer"


def list_instances(folder: str | None = None) -> List[Dict]:
    out = []
    base = INST_ROOT if folder is None else INST_ROOT / folder
    glob_pat = "*.vrp" if folder else "*/*.vrp"
    for vrp in sorted(base.glob(glob_pat)):
        sol = vrp.with_suffix(".sol")
        out.append({"name": vrp.stem, "vrp": str(vrp), "bks": str(sol) if sol.exists() else None})
    return out


def load_routes(sol_path: Path) -> List[List[int]]:
    routes, _ = parse_solution(sol_path)
    return routes


def routes_json(vrp: Path, routes: List[List[int]], layer: str) -> Dict:
    inst = parser.parse_vrp(vrp)
    coords = {int(k): v for k, v in inst.coords.items()}
    dist = build_distance(inst)
    total, per = evaluator.route_costs(routes, dist, inst.depots[0])
    return {
        "layer": layer,
        "cost": total,
        "demands": {int(k): v for k, v in inst.demands.items()},
        "routes": [
            {"id": f"{layer}_{i+1}", "nodes": rt, "cost": per[i], "load": sum(inst.demands.get(n, 0) for n in rt)}
            for i, rt in enumerate(routes)
        ],
        "coords": coords,
        "depot": inst.depots[0],
    }


@app.get("/instances")
async def api_instances(folder: str | None = Query(default=None)):
    return JSONResponse(list_instances(folder))


@app.get("/folders")
async def api_folders():
    folders = [p.name for p in INST_ROOT.iterdir() if p.is_dir()]
    return JSONResponse(sorted(folders))


@app.post("/run")
async def api_run(instance: str, seed: Optional[int] = None):
    vrp_path = Path(instance)
    if not vrp_path.exists():
        vrp_path = INST_ROOT / instance
    if not vrp_path.exists():
        raise HTTPException(status_code=400, detail="Instance not found")
    SOL_ROOT.mkdir(parents=True, exist_ok=True)
    sol_path = SOL_ROOT / f"{vrp_path.stem}.sol"
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(ROOT / "scripts" / "run_v5.py"),
        "--instance",
        str(vrp_path),
        "--solution",
        str(sol_path),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    proc = await asyncio.create_subprocess_exec(*cmd, cwd=str(ROOT), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Solver failed: {err.decode()}")
    return {"solution": str(sol_path)}


@app.get("/solution")
async def api_solution(instance: str, layer: str = "solver"):
    vrp_path = Path(instance)
    if not vrp_path.exists():
        vrp_path = INST_ROOT / instance
    if not vrp_path.exists():
        raise HTTPException(status_code=400, detail="Instance not found")
    if layer == "solver":
        sol_path = SOL_ROOT / f"{vrp_path.stem}.sol"
    elif layer == "bks":
        sol_path = vrp_path.with_suffix(".sol")
    else:
        raise HTTPException(status_code=400, detail="Unsupported layer")
    if not sol_path.exists():
        raise HTTPException(status_code=404, detail="Solution not found")
    routes = load_routes(sol_path)
    return JSONResponse(routes_json(vrp_path, routes, layer))


def edges_from_routes(routes: List[List[int]], depot: int) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for rt in routes:
        seq = [depot] + rt + [depot]
        for a, b in zip(seq, seq[1:]):
            edge = tuple(sorted((a, b)))
            edges.add(edge)
    return edges


@app.get("/summary")
async def api_summary(instance: str):
    vrp_path = Path(instance)
    if not vrp_path.exists():
        vrp_path = INST_ROOT / instance
    if not vrp_path.exists():
        raise HTTPException(status_code=400, detail="Instance not found")
    sol_path = SOL_ROOT / f"{vrp_path.stem}.sol"
    inst = parser.parse_vrp(vrp_path)
    dist = build_distance(inst)
    solver_cost = None
    solver_routes = []
    if sol_path.exists():
        solver_routes, _ = parse_solution(sol_path)
        if solver_routes:
            solver_cost, _ = evaluator.route_costs(solver_routes, dist, inst.depots[0])
    bks_path = vrp_path.with_suffix(".sol")
    bks_cost = None
    bks_routes = []
    if bks_path.exists():
        bks_routes, _ = parse_solution(bks_path)
        if bks_routes:
            bks_cost, _ = evaluator.route_costs(bks_routes, dist, inst.depots[0])
    gap = None
    if solver_cost is not None and bks_cost:
        gap = (solver_cost - bks_cost) / bks_cost * 100

    coverage_pct = None
    covered = 0
    total = 0
    if bks_routes:
        # Build repo to estimate coverage
        v5_inst = load_instance(vrp_path)
        snaps, v2_final = run_v2_with_snapshots(v5_inst)
        repo = build_repo(v5_inst, snaps + [v2_final], DEFAULT_CONFIG)
        repo_edges = set()
        for rr in repo.routes:
            repo_edges |= edges_from_routes([rr.seq], v5_inst.depot)
        bks_edges = edges_from_routes(bks_routes, inst.depots[0])
        covered = len(bks_edges & repo_edges)
        total = len(bks_edges)
        coverage_pct = covered / total * 100 if total else None

    return JSONResponse(
        {
            "instance": vrp_path.stem,
            "solver_cost": solver_cost,
            "bks_cost": bks_cost,
            "gap_pct": gap,
            "coverage_pct": coverage_pct,
            "covered_edges": covered,
            "total_edges": total,
        }
    )


@app.get("/")
async def root(instance: str | None = Query(default=None)):
    index = ROOT / "viewer" / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Missing viewer/index.html")
    if instance:
        # Serve index with query parameter handled client-side
        return FileResponse(index)
    return FileResponse(index)


@app.get("/table")
async def table():
    page = ROOT / "viewer" / "table.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Missing viewer/table.html")
    return FileResponse(page)


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(LOG_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def _safe_load_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        val = json.loads(text)
    except Exception:
        return None
    return val if isinstance(val, dict) else None


def _config_summary(args: dict[str, Any] | None) -> str | None:
    if not args:
        return None
    parts: list[str] = []
    phis = args.get("phis")
    if phis:
        parts.append(f"phis={phis}")
    axis = args.get("axis_count_schedule")
    if axis:
        parts.append(f"axis={axis}")
    t = args.get("time_limit_s")
    if t:
        parts.append(f"t={t}s")
    k_remove = args.get("k_remove")
    if k_remove is not None:
        parts.append(f"k_remove={k_remove}")
    return " ".join(parts) if parts else None


def _parse_instance_arrays(vrp_path: Path) -> dict[str, Any]:
    inst = parser.parse_vrp(vrp_path)
    dim = inst.dimension
    x = [0.0] * (dim + 1)
    y = [0.0] * (dim + 1)
    for node_id, (xi, yi) in inst.coords.items():
        x[int(node_id)] = float(xi)
        y[int(node_id)] = float(yi)
    demand = [0] * (dim + 1)
    for node_id, d in inst.demands.items():
        demand[int(node_id)] = int(d)
    return {
        "name": vrp_path.stem,
        "vrp_path": str(vrp_path),
        "dimension": int(dim),
        "capacity": int(inst.capacity) if inst.capacity is not None else None,
        "depot": int(inst.depots[0]),
        "x": x,
        "y": y,
        "demand": demand,
    }


@functools.lru_cache(maxsize=128)
def _depot_for_vrp_path(vrp_path_str: str) -> int:
    inst = parser.parse_vrp(Path(vrp_path_str))
    return int(inst.depots[0])


@functools.lru_cache(maxsize=128)
def _capacity_for_vrp_path(vrp_path_str: str) -> int | None:
    inst = parser.parse_vrp(Path(vrp_path_str))
    return int(inst.capacity) if inst.capacity is not None else None


def _load_bks_for_run(vrp_path: Path, bks_path: Path | None) -> dict[str, Any] | None:
    if bks_path is None or not bks_path.exists():
        return None
    inst = parser.parse_vrp(vrp_path)
    depot = int(inst.depots[0])
    dist = build_distance(inst)
    routes_raw, _ = parse_solution(bks_path)
    norm = normalize_routes(inst, routes_raw, depot=depot, indexing="auto")
    total, per = evaluator.route_costs(norm.routes, dist, depot)
    return {
        "cost": int(total),
        "routes": [
            {"id": f"bks_{i+1}", "nodes": [int(n) for n in rt], "cost": int(per[i]), "load": sum(inst.demands.get(n, 0) for n in rt)}
            for i, rt in enumerate(norm.routes)
        ],
    }


@app.get("/gen5v5")
async def gen5v5_runs_page():
    page = VIEWER_DIR / "gen5_v5_runs.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Missing viewer/gen5_v5_runs.html")
    return FileResponse(page)


@app.get("/gen5v5/run")
async def gen5v5_run_page():
    page = VIEWER_DIR / "gen5_v5_run.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Missing viewer/gen5_v5_run.html")
    return FileResponse(page)


@app.get("/api/gen5v5/runs")
async def api_gen5v5_runs():
    conn = _db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
              r.run_id, r.ts, r.vrp_path, r.bks_path, r.out_sol_path, r.solver_path, r.notes, r.args_json,
              res.valid, res.solver_cost, res.bks_cost, res.gap_pct, res.edge_pct, res.runtime_s, res.error
            FROM gen5_v5_runs r
            LEFT JOIN gen5_v5_results res ON res.run_id = r.run_id
            ORDER BY r.run_id DESC
            """
        )
        rows = [_row_to_dict(x) for x in cur.fetchall()]
        for row in rows:
            row["instance"] = Path(str(row.get("vrp_path", ""))).stem
            args = _safe_load_json(row.get("args_json"))
            row["args"] = args
            row["config_summary"] = _config_summary(args)
        return JSONResponse(rows)
    finally:
        conn.close()


@app.get("/api/gen5v5/run/{run_id}")
async def api_gen5v5_run(run_id: int):
    conn = _db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM gen5_v5_runs WHERE run_id = ?", (int(run_id),))
        run = cur.fetchone()
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        out = _row_to_dict(run)
        out["instance"] = Path(str(out.get("vrp_path", ""))).stem
        args = _safe_load_json(out.get("args_json"))
        out["args"] = args
        out["config_summary"] = _config_summary(args)
        cur.execute("SELECT * FROM gen5_v5_results WHERE run_id = ? LIMIT 1", (int(run_id),))
        res = cur.fetchone()
        out["result"] = _row_to_dict(res) if res is not None else None
        return JSONResponse(out)
    finally:
        conn.close()


@app.get("/api/gen5v5/run/{run_id}/snapshots")
async def api_gen5v5_run_snapshots(run_id: int):
    conn = _db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
              snapshot_id, ts, stage, cycle_idx, block_idx, axes_k, phis_active, phi_hint, iter_local, elapsed_s,
              reason, valid_coverage, valid_strict, missing_count, extras_count, duplicates_count, cap_violations_count,
              solver_cost, bks_cost, gap_pct, edge_pct, route_count
            FROM gen5_v5_snapshots
            WHERE run_id = ?
            ORDER BY snapshot_id ASC
            """,
            (int(run_id),),
        )
        return JSONResponse([_row_to_dict(x) for x in cur.fetchall()])
    finally:
        conn.close()


def _route_edge_signature(nodes: list[int], depot: int) -> str:
    # Signature based on the undirected edge set in the depot-anchored tour.
    # Captures intra-route order changes (e.g. 2-opt) while being reversal-invariant.
    if not nodes:
        return ""
    seq = [int(depot)] + [int(n) for n in nodes] + [int(depot)]
    edges: list[tuple[int, int]] = []
    for a, b in zip(seq, seq[1:]):
        aa, bb = (a, b) if a <= b else (b, a)
        edges.append((aa, bb))
    edges.sort()
    return ";".join(f"{a}-{b}" for a, b in edges)


@app.get("/api/gen5v5/snapshot/{snapshot_id}")
async def api_gen5v5_snapshot(snapshot_id: int, delta: int = 0):
    conn = _db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT run_id, solver_routes_zlib FROM gen5_v5_snapshots WHERE snapshot_id = ?", (int(snapshot_id),))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        run_id = int(row["run_id"])
        payload = json.loads(zlib.decompress(row["solver_routes_zlib"]).decode("utf-8"))
        if not delta:
            payload["snapshot_id"] = int(snapshot_id)
            cur.execute("SELECT vrp_path FROM gen5_v5_runs WHERE run_id = ? LIMIT 1", (int(run_id),))
            run_row = cur.fetchone()
            if run_row is not None:
                q = _capacity_for_vrp_path(str(run_row["vrp_path"]))
                if q is not None:
                    payload["q_orig"] = int(q)
                    for r in payload.get("routes", []):
                        ld = int(r.get("load", 0))
                        over = max(0, ld - int(q))
                        r["over_by"] = int(over)
                        r["violated"] = bool(over > 0)
            return JSONResponse(payload)

        cur.execute("SELECT vrp_path FROM gen5_v5_runs WHERE run_id = ? LIMIT 1", (int(run_id),))
        run_row = cur.fetchone()
        if run_row is None:
            raise HTTPException(status_code=404, detail="Run not found for snapshot")
        depot = _depot_for_vrp_path(str(run_row["vrp_path"]))
        q = _capacity_for_vrp_path(str(run_row["vrp_path"]))

        cur.execute(
            "SELECT snapshot_id, solver_routes_zlib FROM gen5_v5_snapshots WHERE run_id = ? AND snapshot_id < ? ORDER BY snapshot_id DESC LIMIT 1",
            (int(run_id), int(snapshot_id)),
        )
        prev = cur.fetchone()
        if prev is None:
            payload["snapshot_id"] = int(snapshot_id)
            payload["delta_vs_snapshot_id"] = None
            payload["changed_route_indices"] = list(range(len(payload.get("routes", []))))
            return JSONResponse(payload)

        prev_payload = json.loads(zlib.decompress(prev["solver_routes_zlib"]).decode("utf-8"))
        prev_routes = prev_payload.get("routes", [])
        cur_routes = payload.get("routes", [])

        prev_sig_set: set[str] = set()
        for r in prev_routes:
            nodes = [int(n) for n in r.get("nodes", [])]
            prev_sig_set.add(_route_edge_signature(nodes, depot))

        changed_indices: list[int] = []
        changed_routes: list[dict[str, Any]] = []
        for idx, r in enumerate(cur_routes):
            nodes = [int(n) for n in r.get("nodes", [])]
            sig = _route_edge_signature(nodes, depot)
            if sig not in prev_sig_set:
                changed_indices.append(idx)
                changed_routes.append(r)

        out = dict(payload)
        out["snapshot_id"] = int(snapshot_id)
        out["delta_vs_snapshot_id"] = int(prev["snapshot_id"])
        out["changed_route_indices"] = changed_indices
        out["routes"] = changed_routes
        if q is not None:
            out["q_orig"] = int(q)
            for r in out.get("routes", []):
                ld = int(r.get("load", 0))
                over = max(0, ld - int(q))
                r["over_by"] = int(over)
                r["violated"] = bool(over > 0)
        return JSONResponse(out)
    finally:
        conn.close()


@app.get("/api/gen5v5/instance/{run_id}")
async def api_gen5v5_instance(run_id: int):
    conn = _db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT vrp_path, bks_path FROM gen5_v5_runs WHERE run_id = ?", (int(run_id),))
        row = cur.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        vrp_path = Path(str(row["vrp_path"]))
        bks_path = Path(str(row["bks_path"])) if row["bks_path"] else None
        if not vrp_path.exists():
            raise HTTPException(status_code=404, detail="VRP path does not exist")
        inst_json = _parse_instance_arrays(vrp_path)
        bks = _load_bks_for_run(vrp_path, bks_path)
        inst_json["bks"] = bks
        return JSONResponse(inst_json)
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("viewer.backend:app", host="0.0.0.0", port=8000, reload=False)
