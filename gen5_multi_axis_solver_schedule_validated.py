#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sqlite3
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cvrp import (  # noqa: E402
    build_distance,
    capacity_report,
    coverage_report,
    format_solution,
    normalize_routes,
    parse_solution,
    parse_vrp,
    route_costs,
    to_customer_indexing,
)


def load_external_solver(solver_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("ext_gen5_multi_axis_sched", solver_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to import solver from {solver_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def fmt_pct(x: float | None, nd: int) -> str:
    return "N/A" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}%"


def edges_undirected(routes: list[list[int]], depot: int) -> set[frozenset[int]]:
    edges: set[frozenset[int]] = set()
    for rt in routes:
        seq = [depot] + rt + [depot]
        for a, b in zip(seq, seq[1:]):
            edges.add(frozenset((a, b)))
    return edges


def parse_int_list_csv(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def take_first_k_phis(stn_by_phi: dict[float, Any], phis: list[float], k: int) -> dict[float, Any]:
    k = int(k)
    if k <= 0:
        return {}
    return {float(phi): stn_by_phi[float(phi)] for phi in phis[:k]}


def parse_str_list_csv(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    out: list[str] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(part)
    return out


def _normalize_stn_profile_name(name: str) -> str:
    n = (name or "").strip().lower()
    if n in ("lat", "lateral", "y"):
        return "lateral"
    if n in ("far", "r2", "radial"):
        return "far"
    if n in ("base", "default"):
        return "base"
    return n


def ensure_sqlite_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS gen5_v5_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            vrp_path TEXT NOT NULL,
            bks_path TEXT,
            out_sol_path TEXT,
            solver_path TEXT NOT NULL,
            notes TEXT,
            args_json TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS gen5_v5_results (
            run_id INTEGER NOT NULL,
            instance_name TEXT NOT NULL,
            valid INTEGER NOT NULL,
            solver_cost INTEGER,
            bks_cost INTEGER,
            gap_pct REAL,
            edge_pct REAL,
            runtime_s REAL NOT NULL,
            error TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS gen5_v5_snapshots (
            snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            ts DATETIME DEFAULT CURRENT_TIMESTAMP,
            stage TEXT NOT NULL,
            cycle_idx INTEGER NOT NULL,
            block_idx INTEGER NOT NULL,
            axes_k INTEGER NOT NULL,
            phis_active TEXT NOT NULL,
            phi_hint REAL,
            iter_local INTEGER,
            elapsed_s REAL NOT NULL,
            reason TEXT NOT NULL,
            valid_coverage INTEGER NOT NULL,
            valid_strict INTEGER NOT NULL,
            missing_count INTEGER NOT NULL,
            extras_count INTEGER NOT NULL,
            duplicates_count INTEGER NOT NULL,
            cap_violations_count INTEGER NOT NULL,
            solver_cost INTEGER,
            bks_cost INTEGER,
            gap_pct REAL,
            edge_pct REAL,
            route_count INTEGER NOT NULL,
            solver_routes_zlib BLOB NOT NULL
        )
        """
    )
    conn.commit()


def log_run_start(
    conn: sqlite3.Connection,
    *,
    vrp_path: Path,
    bks_path: Path | None,
    out_sol_path: Path,
    solver_path: Path,
    notes: str,
    args_json: str,
) -> int:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO gen5_v5_runs(vrp_path, bks_path, out_sol_path, solver_path, notes, args_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(vrp_path),
            str(bks_path) if bks_path is not None else None,
            str(out_sol_path),
            str(solver_path),
            notes,
            args_json,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_run_out_sol_path(conn: sqlite3.Connection, *, run_id: int, out_sol_path: Path) -> None:
    cur = conn.cursor()
    cur.execute(
        "UPDATE gen5_v5_runs SET out_sol_path = ? WHERE run_id = ?",
        (str(out_sol_path), int(run_id)),
    )
    conn.commit()


def runid_prefixed_out_sol(out_sol: Path, run_id: int) -> Path:
    """
    Prefix basename with runid_<run_id>_ for traceability.
    Avoid double-prefixing if already present.
    """
    name = out_sol.name
    if re.match(r"^runid_\\d+_", name):
        return out_sol
    return out_sol.with_name(f"runid_{int(run_id)}_{name}")


def log_run_result(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    instance_name: str,
    valid: bool,
    solver_cost: int | None,
    bks_cost: int | None,
    gap_pct: float | None,
    edge_pct: float | None,
    runtime_s: float,
    error: str | None,
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO gen5_v5_results(
            run_id, instance_name, valid, solver_cost, bks_cost, gap_pct, edge_pct, runtime_s, error
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(run_id),
            instance_name,
            1 if valid else 0,
            int(solver_cost) if solver_cost is not None else None,
            int(bks_cost) if bks_cost is not None else None,
            float(gap_pct) if gap_pct is not None else None,
            float(edge_pct) if edge_pct is not None else None,
            float(runtime_s),
            error,
        ),
    )
    conn.commit()


def solver_routes_to_vrp(mod: Any, vrp_path: Path, solver_routes: list[list[int]]) -> tuple[list[list[int]], int]:
    prob = mod.parse_vrp(str(vrp_path))
    model = mod.build_model(prob)
    vrp_routes: list[list[int]] = []
    for rt in solver_routes:
        vrp_routes.append([int(model.sol_to_orig[int(sid)]) for sid in rt if int(sid) != 0])
    depot_vrp = int(model.sol_to_orig[0])
    return vrp_routes, depot_vrp


def validate_vrp_routes(vrp_path: Path, routes_vrp: list[list[int]]) -> tuple[bool, int | None, list[list[int]]]:
    inst = parse_vrp(vrp_path)
    depot = inst.depots[0]
    dist = build_distance(inst)
    norm = normalize_routes(inst, routes_vrp, depot=depot, indexing="vrp")
    missing, extras, dupes = coverage_report(inst, norm.routes)
    cap = capacity_report(inst, norm.routes)
    valid = not missing and not extras and not dupes and not cap
    total, _ = route_costs(norm.routes, dist, depot)
    return valid, int(total), norm.routes


def strict_repair_to_capacity(model: Any, routes: list[list[int]]) -> list[list[int]]:
    """
    Repair routes to satisfy model.Q by relocating nodes out of overloaded routes.
    Assumes depot=0 omitted and nodes are in solver indexing.
    """
    import numpy as np

    D = model.D_obj
    q = model.q
    Q = int(model.Q)

    routes = [rt[:] for rt in routes if rt]
    loads = [int(sum(int(q[n]) for n in rt)) for rt in routes]

    # Radial feature r2 in [0,1] from depot (heuristic only; does not affect objective).
    dx = model.x - float(model.x[0])
    dy = model.y - float(model.y[0])
    r = np.hypot(dx, dy)
    rmax = float(np.max(r[1:])) if int(getattr(model, "N", 0)) > 0 else 1.0
    if rmax <= 0:
        rmax = 1.0
    r2 = (r / rmax) ** 2

    def remove_delta(rt: list[int], pos: int) -> int:
        a = 0 if pos == 0 else rt[pos - 1]
        node = rt[pos]
        b = 0 if pos == len(rt) - 1 else rt[pos + 1]
        return int(D[a, b]) - int(D[a, node]) - int(D[node, b])

    def detour_cost(rt: list[int], pos: int) -> int:
        # >=0 under metric distances (triangle inequality), but keep robust.
        a = 0 if pos == 0 else rt[pos - 1]
        node = rt[pos]
        b = 0 if pos == len(rt) - 1 else rt[pos + 1]
        return max(0, int(D[a, node]) + int(D[node, b]) - int(D[a, b]))

    def best_insert_delta(rt: list[int], node: int) -> tuple[int, int]:
        best = None
        best_pos = 0
        L = len(rt)
        for pos in range(L + 1):
            a = 0 if pos == 0 else rt[pos - 1]
            b = 0 if pos == L else rt[pos]
            delta = int(D[a, node]) + int(D[node, b]) - int(D[a, b])
            if best is None or delta < best:
                best = delta
                best_pos = pos
        return int(best if best is not None else 0), best_pos

    # Defaults tuned for: prefer removing near-depot nodes with low demand and low detour cost.
    w_r2 = float(getattr(model, "_repair_w_r2", 1.0))
    w_detour = float(getattr(model, "_repair_w_detour", 2.0))
    w_q = float(getattr(model, "_repair_w_q", 1.5))
    topk = int(getattr(model, "_repair_topk", 12))
    max_steps = int(getattr(model, "_repair_max_steps", 200000))

    steps = 0
    while steps < max_steps:
        over = [ld - Q for ld in loads]
        rid = int(np.argmax(over))
        if over[rid] <= 0:
            break
        rt = routes[rid]
        if not rt:
            loads[rid] = 0
            steps += 1
            continue

        # Candidate selection: remove "cheap" nodes first (low keep_score), but always include a few
        # high-demand nodes as fallback when overload is large.
        detours = [detour_cost(rt, pos) for pos in range(len(rt))]
        dmax = max(detours) if detours else 1
        if dmax <= 0:
            dmax = 1
        keep_by_pos: list[float] = [0.0] * len(rt)
        keep_scores: list[tuple[float, int]] = []
        for pos, node in enumerate(rt):
            node_i = int(node)
            det_norm = float(detours[pos]) / float(dmax)
            q_norm = float(int(q[node_i])) / float(Q) if Q > 0 else 0.0
            score = (w_r2 * float(r2[node_i])) + (w_detour * det_norm) + (w_q * q_norm)
            keep_by_pos[pos] = float(score)
            keep_scores.append((score, pos))
        keep_scores.sort(key=lambda t: (t[0], t[1]))
        cand_pos = [pos for _s, pos in keep_scores[: max(1, min(int(topk), len(keep_scores)))]]
        # Add up to 2 highest-demand nodes (by demand, then by lowest keep score) as fallback.
        if len(rt) > 1:
            q_rank = sorted(range(len(rt)), key=lambda p: (-int(q[int(rt[p])]), keep_by_pos[p], p))
            for p in q_rank[:2]:
                if p not in cand_pos:
                    cand_pos.append(int(p))

        best_move = None
        best_delta = None
        for pos in cand_pos:
            node = rt[pos]
            demand = int(q[node])
            drem = remove_delta(rt, pos)
            # try insert into other routes
            for r2 in range(len(routes)):
                if r2 == rid:
                    continue
                if loads[r2] + demand > Q:
                    continue
                delta_ins, ins_pos = best_insert_delta(routes[r2], node)
                delta = drem + delta_ins
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_move = (pos, node, r2, ins_pos)
            # always allow opening a new route
            delta_new = drem + int(D[0, node]) + int(D[node, 0])
            if best_delta is None or delta_new < best_delta:
                best_delta = delta_new
                best_move = (pos, node, -1, 0)

        if best_move is None:
            break
        pos, node, r2, ins_pos = best_move
        routes[rid].pop(pos)
        loads[rid] -= int(q[node])
        if r2 < 0:
            routes.append([node])
            loads.append(int(q[node]))
        else:
            routes[r2].insert(ins_pos, node)
            loads[r2] += int(q[node])

        if not routes[rid]:
            routes.pop(rid)
            loads.pop(rid)
        steps += 1

    return [rt for rt in routes if rt]


def build_stn_by_phi(
    mod: Any,
    model: Any,
    phis: list[float],
    *,
    prob_ewt: str,
    K_geo_pool: int,
    K_r_pool: int,
    K_y_pool: int,
    K_final: int,
    stn_r2_penalty_frac: float = 0.0,
    stn_far_y_mult: float = 0.0,
    stn_far_r_mult: float = 0.0,
) -> dict[float, Any]:
    import numpy as np

    N = int(getattr(model, "N"))
    K_geo_pool_eff = max(1, min(int(K_geo_pool), N))
    K_r_pool_eff = max(1, min(int(K_r_pool), N))
    K_y_pool_eff = max(1, min(int(K_y_pool), N))
    pool_size = K_geo_pool_eff + K_r_pool_eff + K_y_pool_eff
    max_neighbors = max(1, N - 1)
    K_final_eff = max(1, min(int(K_final), max_neighbors, pool_size))

    stn_by_phi: dict[float, Any] = {}
    use_objknn = prob_ewt != "EUC_2D"
    if use_objknn:
        D_obj = model.D_obj
        K = max(1, min(128, N - 1))
        stn = np.zeros((N + 1, K_final_eff), dtype=np.int32)
        for i in range(1, N + 1):
            d = D_obj[i, 1 : N + 1].copy()
            d[i - 1] = 10**9
            nn = np.argpartition(d, min(K, d.shape[0] - 1))[:K]
            neigh = (nn + 1).astype(np.int32)
            neigh = neigh[np.argsort(D_obj[i, neigh])]
            stn[i, : min(K_final_eff, neigh.shape[0])] = neigh[:K_final_eff]
        for phi in phis:
            stn_by_phi[float(phi)] = stn
        return stn_by_phi

    dx, dy, r2 = mod.compute_r_features(model)
    for phi in phis:
        try:
            stn_by_phi[float(phi)] = mod.build_stn_probe(
                model,
                dx,
                dy,
                r2,
                float(phi),
                K_geo_pool=K_geo_pool_eff,
                K_r_pool=K_r_pool_eff,
                K_y_pool=K_y_pool_eff,
                K_final=K_final_eff,
                r2_penalty_frac=float(stn_r2_penalty_frac),
                far_y_mult=float(stn_far_y_mult),
                far_r_mult=float(stn_far_r_mult),
            )
        except TypeError:
            stn_by_phi[float(phi)] = mod.build_stn_probe(
                model,
                dx,
                dy,
                r2,
                float(phi),
                K_geo_pool=K_geo_pool_eff,
                K_r_pool=K_r_pool_eff,
                K_y_pool=K_y_pool_eff,
                K_final=K_final_eff,
            )
    return stn_by_phi


def run_lns_prune(
    mod: Any,
    model: Any,
    base_routes: list[list[int]],
    stn_by_phi: dict[float, Any],
    *,
    iters: int,
    k_remove: int,
    accept_p: float,
    seed: int,
    steps_reloc: int,
    steps_2opt: int,
    plateau_window: int,
    prune_batch: int,
    eps: float,
    min_axes: int,
    time_limit_s: float | None = None,
    on_best: Callable[[int, int, float, float], None] | None = None,
    lookahead: int = 1,
    epn: Any | None = None,
    epn_cfg: Any | None = None,
    prefer_prune: bool = True,
    full_2opt_on_best: bool = False,
    full_2opt_max_len: int = 0,
    full_2opt_max_passes: int = 50,
) -> tuple[int, list[list[int]]]:
    if prefer_prune and hasattr(mod, "multi_axis_lns_prune"):
        try:
            best_cost, best_routes, _axes, _stats = mod.multi_axis_lns_prune(
                model,
                base_routes,
                stn_by_phi,
                iters=int(iters),
                k_remove=int(k_remove),
                accept_p=float(accept_p),
                seed=int(seed),
                steps_reloc=int(steps_reloc),
                steps_2opt=int(steps_2opt),
                plateau_window=int(plateau_window),
                min_axes=int(min_axes),
                prune_batch=int(prune_batch),
                eps=float(eps),
                lookahead=int(lookahead),
                time_limit_s=time_limit_s,
                on_best=on_best,
                epn=epn,
                epn_cfg=epn_cfg,
                full_2opt_on_best=bool(full_2opt_on_best),
                full_2opt_max_len=int(full_2opt_max_len),
                full_2opt_max_passes=int(full_2opt_max_passes),
            )
        except TypeError:
            best_cost, best_routes, _axes, _stats = mod.multi_axis_lns_prune(
                model,
                base_routes,
                stn_by_phi,
                iters=int(iters),
                k_remove=int(k_remove),
                accept_p=float(accept_p),
                seed=int(seed),
                steps_reloc=int(steps_reloc),
                steps_2opt=int(steps_2opt),
                plateau_window=int(plateau_window),
                min_axes=int(min_axes),
                prune_batch=int(prune_batch),
                eps=float(eps),
            )
        return int(best_cost), [list(map(int, rt)) for rt in best_routes]
    try:
        best_cost, best_routes = mod.multi_axis_lns(
            model,
            base_routes,
            stn_by_phi,
            iters=int(iters),
            k_remove=int(k_remove),
            accept_p=float(accept_p),
            seed=int(seed),
            steps_reloc=int(steps_reloc),
            steps_2opt=int(steps_2opt),
            lookahead=int(lookahead),
            time_limit_s=time_limit_s,
            on_best=on_best,
            epn=epn,
            epn_cfg=epn_cfg,
            full_2opt_on_best=bool(full_2opt_on_best),
            full_2opt_max_len=int(full_2opt_max_len),
            full_2opt_max_passes=int(full_2opt_max_passes),
        )
    except TypeError:
        best_cost, best_routes = mod.multi_axis_lns(
            model,
            base_routes,
            stn_by_phi,
            iters=int(iters),
            k_remove=int(k_remove),
            accept_p=float(accept_p),
            seed=int(seed),
            steps_reloc=int(steps_reloc),
            steps_2opt=int(steps_2opt),
        )
    return int(best_cost), [list(map(int, rt)) for rt in best_routes]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gen5 multi-axis schedule solver (validated) for multi-seed runner.")
    ap.add_argument("vrp", type=Path, help="Instance .vrp path")
    ap.add_argument("--bks", type=Path, default=None)
    ap.add_argument("--out-sol", "--out_sol", dest="out_sol", type=Path, required=True)
    ap.add_argument(
        "--db",
        type=Path,
        default=Path("runs.sqlite"),
        help="SQLite DB path for run logging (default: runs.sqlite).",
    )
    ap.add_argument("--notes", default="", help="Optional note/description for this run.")
    ap.add_argument(
        "--time-limit-s",
        type=float,
        default=0.0,
        help="Optional wall-clock time budget for the solve phases (0 = no limit).",
    )
    ap.add_argument(
        "--progress-every-s",
        type=float,
        default=5.0,
        help="Minimum seconds between progress lines (best cost/gap).",
    )
    ap.add_argument(
        "--soft-lookahead",
        type=int,
        default=5,
        help="Lookahead (beam) for soft stage candidates (default: 5).",
    )
    ap.add_argument(
        "--strict-lookahead",
        type=int,
        default=1,
        help="Lookahead (beam) for strict/tight stages (default: 1).",
    )
    ap.add_argument(
        "--no-epn-soft",
        action="store_true",
        help="Disable EPN (edge proximity neighbors) top-up during soft stage.",
    )
    ap.add_argument("--epn-k1", type=int, default=32, help="EPN KNN1 size for exclusion (default: 32).")
    ap.add_argument("--epn-k", type=int, default=16, help="EPN max nodes per edge query (default: 16).")
    ap.add_argument("--epn-max-tested", type=int, default=1024, help="EPN max tested nodes per edge (default: 1024).")
    ap.add_argument(
        "--epn-long-edges-per-route",
        type=int,
        default=10,
        help="EPN trigger edges per route (longest edges) (default: 10).",
    )
    ap.add_argument("--epn-max-moves", type=int, default=12, help="EPN applied moves per LNS step (default: 12).")
    ap.add_argument("--epn-d2-beta", type=float, default=0.04, help="EPN d2 threshold = beta*edge_len2 (default: 0.04).")
    ap.add_argument(
        "--stn-r2-penalty-frac",
        type=float,
        default=0.0,
        help="EUC_2D STN ranking penalty: +frac*max(depot_dist)*|r2(i)-r2(j)| (default: 0).",
    )
    ap.add_argument(
        "--stn-far-y-mult",
        type=float,
        default=0.0,
        help="EUC_2D STN: scale K_y_pool per node by (1 + mult*r2[i]) (default: 0).",
    )
    ap.add_argument(
        "--stn-far-r-mult",
        type=float,
        default=0.0,
        help="EUC_2D STN: scale K_r_pool per node by (1 + mult*r2[i]) (default: 0).",
    )
    ap.add_argument(
        "--stn-profile-schedule",
        default="",
        help="Comma-separated STN profiles to cycle per axis block, e.g. 'far,lateral' (default: disabled).",
    )
    ap.add_argument(
        "--stn-profile-blocks",
        type=int,
        default=10,
        help="Axis blocks per STN profile before switching (default: 10).",
    )
    ap.add_argument(
        "--stn-lateral-r2-penalty-frac",
        type=float,
        default=0.0,
        help="Lateral STN: r2 penalty frac (default: 0.0).",
    )
    ap.add_argument(
        "--stn-lateral-far-y-mult",
        type=float,
        default=0.0,
        help="Lateral STN: far-y mult scaling (default: 0.0).",
    )
    ap.add_argument(
        "--stn-lateral-far-r-mult",
        type=float,
        default=0.0,
        help="Lateral STN: far-r mult scaling (default: 0.0).",
    )
    ap.add_argument(
        "--stn-lateral-y-pool-mult",
        type=float,
        default=1.5,
        help="Lateral STN: multiply K_y_pool (default: 1.5).",
    )
    ap.add_argument(
        "--stn-lateral-r-pool-mult",
        type=float,
        default=0.8,
        help="Lateral STN: multiply K_r_pool (default: 0.8).",
    )
    ap.add_argument(
        "--full-2opt-on-best",
        action="store_true",
        help="Run a full per-route 2-opt local optimum pass whenever a new global best is found (off by default).",
    )
    ap.add_argument(
        "--full-2opt-stage",
        choices=["strict", "all"],
        default="strict",
        help="Where to apply --full-2opt-on-best (default: strict only).",
    )
    ap.add_argument(
        "--full-2opt-max-len",
        type=int,
        default=0,
        help="Only apply full 2-opt to routes with len<=N (0 = no cap) (default: 0).",
    )
    ap.add_argument(
        "--full-2opt-max-passes",
        type=int,
        default=50,
        help="Max improvement passes per route for full 2-opt (default: 50).",
    )
    ap.add_argument(
        "--axis-count-schedule",
        default="",
        help="Comma-separated active axis counts per block (prefix of --phis), e.g. '1,2,3,4,5,6,5,4,3,2,1'.",
    )
    ap.add_argument(
        "--axis-block-iters",
        type=int,
        default=20,
        help="Iters per axis-count block when --axis-count-schedule is set (default: 20).",
    )
    ap.add_argument(
        "--axis-block-time-s",
        type=float,
        default=0.0,
        help="Optional seconds per axis-count block (0=disabled; uses --axis-block-iters).",
    )
    ap.add_argument(
        "--axis-schedule-mode",
        choices=["cycle", "prune"],
        default="cycle",
        help="How to use multiple phis within each axis block: cycle=round-robin, prune=epsilon-greedy prune.",
    )
    ap.add_argument(
        "--axis-block-seed-step",
        type=int,
        default=1000,
        help="Deterministic seed increment per axis block (default: 1000).",
    )
    ap.add_argument("--repair-w-r2", type=float, default=1.0, help="Strict repair: keep weight for r2 (default: 1.0).")
    ap.add_argument(
        "--repair-w-detour",
        type=float,
        default=2.0,
        help="Strict repair: keep weight for detour cost (DDC) (default: 2.0).",
    )
    ap.add_argument("--repair-w-q", type=float, default=1.5, help="Strict repair: keep weight for demand q (default: 1.5).")
    ap.add_argument(
        "--repair-topk",
        type=int,
        default=12,
        help="Strict repair: evaluate moves for only top-k cheapest nodes in an overloaded route (default: 12).",
    )
    ap.add_argument("--repair-max-steps", type=int, default=200000, help="Strict repair: max relocation steps (default: 200000).")
    ap.add_argument(
        "--ffs-on-plateau",
        action="store_true",
        help="Apply FFS (farthest-first seeding) restructure when strict stage plateaus (off by default).",
    )
    ap.add_argument(
        "--ffs-plateau-blocks",
        type=int,
        default=40,
        help="Number of consecutive axis blocks with no global-best improvement before FFS triggers (default: 40).",
    )
    ap.add_argument("--ffs-far-percentile", type=float, default=0.80, help="FFS far percentile (default: 0.80).")
    ap.add_argument("--ffs-angle-bins", type=int, default=64, help="FFS angle bins (default: 64).")
    ap.add_argument("--ffs-angle-bin-radius", type=int, default=2, help="FFS angle bin radius (default: 2).")
    ap.add_argument(
        "--ffs-open-route-if-better",
        action="store_true",
        default=True,
        help="FFS may open a new route when cheaper than inserting into existing routes (default: enabled).",
    )
    ap.add_argument(
        "--no-ffs-open-route-if-better",
        action="store_false",
        dest="ffs_open_route_if_better",
        help="Disable FFS opening new routes based on cost comparison.",
    )
    ap.add_argument(
        "--ffs-open-route-margin",
        type=float,
        default=1.00,
        help="FFS opens a new route if best insertion delta > margin*(depot->node->depot) (default: 1.00).",
    )
    ap.add_argument(
        "--ffs-post-steps-reloc",
        type=int,
        default=160,
        help="FFS post-cleanup relocate steps (default: 160).",
    )
    ap.add_argument(
        "--ffs-post-steps-2opt",
        type=int,
        default=120,
        help="FFS post-cleanup 2-opt steps (default: 120).",
    )
    ap.add_argument(
        "--snapshot-on-best",
        action="store_true",
        default=True,
        help="Record a snapshot when the run's best improves (default: enabled).",
    )
    ap.add_argument(
        "--no-snapshot-on-best",
        action="store_false",
        dest="snapshot_on_best",
        help="Disable snapshot-on-best recording.",
    )
    ap.add_argument(
        "--snapshot-every-s",
        type=float,
        default=10.0,
        help="Record a periodic snapshot every N seconds (0 disables) (default: 10).",
    )
    ap.add_argument(
        "--snapshot-best-min-s",
        type=float,
        default=2.0,
        help="Minimum seconds between snapshot-on-best writes (default: 2).",
    )
    ap.add_argument(
        "--solver-path",
        type=Path,
        default=Path("gen5_multi_axis_solver_validated.py"),
    )

    ap.add_argument("--use-schedule", "--use_schedule", dest="use_schedule", action="store_true")
    ap.add_argument("--use-prune", "--use_prune", dest="use_prune", action="store_true")
    ap.add_argument("--phis", default="45,-45,90,0,135,-135")

    ap.add_argument("--tight-factor", "--tight_factor", dest="tight_factor", type=float, default=0.7)
    ap.add_argument("--soft-factor", "--soft_factor", dest="soft_factor", type=float, default=1.8)
    ap.add_argument("--tight-iters", "--tight_iters", dest="tight_iters", type=int, default=500)
    ap.add_argument("--soft-iters", "--soft_iters", dest="soft_iters", type=int, default=4000)
    ap.add_argument("--strict-iters", "--strict_iters", dest="strict_iters", type=int, default=4000)

    ap.add_argument("--k-remove", "--k_remove", dest="k_remove", type=int, default=180)
    ap.add_argument("--accept-p", "--accept_p", dest="accept_p", type=float, default=0.30)
    ap.add_argument("--K-final", "--K_final", dest="K_final", type=int, default=128)
    ap.add_argument("--K-geo-pool", "--K_geo_pool", dest="K_geo_pool", type=int, default=320)
    ap.add_argument("--K-y-pool", "--K_y_pool", dest="K_y_pool", type=int, default=320)
    ap.add_argument("--K-r-pool", "--K_r_pool", dest="K_r_pool", type=int, default=160)
    ap.add_argument("--steps-reloc", "--steps_reloc", dest="steps_reloc", type=int, default=200)
    ap.add_argument("--steps-2opt", "--steps_2opt", dest="steps_2opt", type=int, default=160)
    ap.add_argument("--lam-mult", "--lam_mult", dest="lam_mult", type=float, default=0.25)
    ap.add_argument("--plateau-window", "--plateau_window", dest="plateau_window", type=int, default=300)
    ap.add_argument("--prune-batch", "--prune_batch", dest="prune_batch", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.12)
    ap.add_argument("--min-axes", "--min_axes", dest="min_axes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args(argv)

    if args.bks is not None and not args.bks.exists():
        print(f"WARNING: --bks path does not exist: {args.bks}", file=sys.stderr)

    args_json = json.dumps(vars(args), default=str, sort_keys=True)
    args.out_sol.parent.mkdir(parents=True, exist_ok=True)
    args.db.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(args.db))
    ensure_sqlite_schema(conn)
    run_id = log_run_start(
        conn,
        vrp_path=args.vrp,
        bks_path=args.bks,
        out_sol_path=args.out_sol,
        solver_path=args.solver_path,
        notes=str(args.notes or ""),
        args_json=args_json,
    )
    args.out_sol = runid_prefixed_out_sol(args.out_sol, run_id)
    update_run_out_sol_path(conn, run_id=run_id, out_sol_path=args.out_sol)
    print(f"Run ID: {run_id}")
    sys.stdout.flush()

    inst_for_bks = parse_vrp(args.vrp)
    depot_for_bks = inst_for_bks.depots[0]
    dist_for_bks = build_distance(inst_for_bks)
    bks_cost: int | None = None
    bks_edges: set[frozenset[int]] | None = None
    bks_routes_vrp: list[list[int]] | None = None
    if args.bks is not None and args.bks.exists():
        try:
            bks_raw, _ = parse_solution(args.bks)
            bks_norm = normalize_routes(inst_for_bks, bks_raw, depot=depot_for_bks, indexing="auto")
            bks_cost_calc, _ = route_costs(bks_norm.routes, dist_for_bks, depot_for_bks)
            bks_cost = int(bks_cost_calc)
            bks_edges = edges_undirected(bks_norm.routes, depot_for_bks)
            bks_routes_vrp = [list(map(int, rt)) for rt in bks_norm.routes]
        except Exception as exc:  # noqa: BLE001
            # Common cause: BKS does not match instance (node ids out of range / different dataset).
            print(
                f"WARNING: failed to load/score --bks solution; ignoring BKS compare: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            bks_cost = None
            bks_edges = None
            bks_routes_vrp = None

    mod = load_external_solver(args.solver_path)
    prob = mod.parse_vrp(str(args.vrp))
    model = mod.build_model(prob)
    Q_orig = int(model.Q)

    epn_state = None
    epn_cfg = None
    if not bool(args.no_epn_soft) and hasattr(mod, "EPNConfig") and hasattr(mod, "build_epn_state"):
        try:
            epn_cfg = mod.EPNConfig(
                enabled=True,
                k1=int(args.epn_k1),
                k_epn=int(args.epn_k),
                max_tested=int(args.epn_max_tested),
                long_edges_per_route=int(args.epn_long_edges_per_route),
                max_moves_per_call=int(args.epn_max_moves),
                d2_beta=float(args.epn_d2_beta),
            )
            epn_state = mod.build_epn_state(model, epn_cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: EPN init failed; disabling EPN for this run: {type(exc).__name__}: {exc}", file=sys.stderr)
            epn_state = None
            epn_cfg = None

    phis = [float(x) for x in args.phis.split(",") if x.strip()]
    axis_counts = parse_int_list_csv(str(args.axis_count_schedule))
    if axis_counts:
        if not phis:
            raise SystemExit("--axis-count-schedule requires --phis to be non-empty")
        max_k = len(phis)
        for k in axis_counts:
            if k < 1 or k > max_k:
                raise SystemExit(f"--axis-count-schedule has invalid k={k}; must be in [1,{max_k}]")
        if float(args.axis_block_time_s) > 0 and int(args.axis_block_iters) > 0:
            # both are allowed; time takes precedence if >0
            pass
    prob_ewt = str(getattr(prob, "ewt", ""))
    profile_names_raw = parse_str_list_csv(str(args.stn_profile_schedule))
    profile_names = [_normalize_stn_profile_name(x) for x in profile_names_raw]
    if profile_names and prob_ewt != "EUC_2D":
        print("WARNING: --stn-profile-schedule is ignored for non-EUC_2D instances.", file=sys.stderr)
        profile_names = []

    stn_profiles: dict[str, dict[float, Any]] = {}
    if not profile_names:
        stn_profiles["base"] = build_stn_by_phi(
            mod,
            model,
            phis,
            prob_ewt=prob_ewt,
            K_geo_pool=args.K_geo_pool,
            K_r_pool=args.K_r_pool,
            K_y_pool=args.K_y_pool,
            K_final=args.K_final,
            stn_r2_penalty_frac=float(args.stn_r2_penalty_frac),
            stn_far_y_mult=float(args.stn_far_y_mult),
            stn_far_r_mult=float(args.stn_far_r_mult),
        )
        active_profile_cycle = ["base"]
    else:
        # Always build the requested profiles once (avoid rebuilding every block).
        # 'far' uses the main --stn-* knobs; 'lateral' uses the --stn-lateral-* knobs.
        active_profile_cycle = profile_names[:]
        for name in active_profile_cycle:
            if name == "far":
                stn_profiles[name] = build_stn_by_phi(
                    mod,
                    model,
                    phis,
                    prob_ewt=prob_ewt,
                    K_geo_pool=args.K_geo_pool,
                    K_r_pool=args.K_r_pool,
                    K_y_pool=args.K_y_pool,
                    K_final=args.K_final,
                    stn_r2_penalty_frac=float(args.stn_r2_penalty_frac),
                    stn_far_y_mult=float(args.stn_far_y_mult),
                    stn_far_r_mult=float(args.stn_far_r_mult),
                )
            elif name == "lateral":
                stn_profiles[name] = build_stn_by_phi(
                    mod,
                    model,
                    phis,
                    prob_ewt=prob_ewt,
                    K_geo_pool=args.K_geo_pool,
                    K_r_pool=int(round(float(args.K_r_pool) * float(args.stn_lateral_r_pool_mult))),
                    K_y_pool=int(round(float(args.K_y_pool) * float(args.stn_lateral_y_pool_mult))),
                    K_final=args.K_final,
                    stn_r2_penalty_frac=float(args.stn_lateral_r2_penalty_frac),
                    stn_far_y_mult=float(args.stn_lateral_far_y_mult),
                    stn_far_r_mult=float(args.stn_lateral_far_r_mult),
                )
            elif name == "base":
                stn_profiles[name] = build_stn_by_phi(
                    mod,
                    model,
                    phis,
                    prob_ewt=prob_ewt,
                    K_geo_pool=args.K_geo_pool,
                    K_r_pool=args.K_r_pool,
                    K_y_pool=args.K_y_pool,
                    K_final=args.K_final,
                    stn_r2_penalty_frac=0.0,
                    stn_far_y_mult=0.0,
                    stn_far_r_mult=0.0,
                )
            else:
                raise SystemExit(f"Unknown STN profile '{name}'. Use far,lateral,base.")

    def stn_profile_for_block(block_idx: int) -> tuple[str, dict[float, Any]]:
        if not active_profile_cycle:
            return ("base", stn_profiles["base"])
        span = max(1, int(args.stn_profile_blocks))
        idx = (int(block_idx) // span) % len(active_profile_cycle)
        name = active_profile_cycle[idx]
        return (name, stn_profiles[name])

    start = time.perf_counter()
    t_deadline = None
    if float(args.time_limit_s) > 0:
        t_deadline = start + float(args.time_limit_s)
    last_progress_t = start - 1e9

    last_snapshot_best_t = start - 1e9
    last_snapshot_periodic_t = start - 1e9

    def solver_routes_to_vrp_fast(routes_solver: list[list[int]]) -> tuple[list[list[int]], int]:
        depot_vrp = int(model.sol_to_orig[0])
        routes_vrp: list[list[int]] = []
        for rt in routes_solver:
            routes_vrp.append([int(model.sol_to_orig[int(sid)]) for sid in rt if int(sid) != 0])
        return routes_vrp, depot_vrp

    def compute_snapshot_metrics(routes_solver: list[list[int]]) -> tuple[dict[str, Any], list[list[int]]]:
        routes_vrp_raw, depot_vrp = solver_routes_to_vrp_fast(routes_solver)
        norm = normalize_routes(inst_for_bks, routes_vrp_raw, depot=depot_vrp, indexing="vrp")
        missing, extras, dupes = coverage_report(inst_for_bks, norm.routes)
        cap = capacity_report(inst_for_bks, norm.routes)
        total_cost, per_cost = route_costs(norm.routes, dist_for_bks, depot_for_bks)
        sol_edges = edges_undirected(norm.routes, depot_for_bks)
        edge_pct = None
        if bks_edges:
            edge_pct = (len(sol_edges & bks_edges) / len(bks_edges)) * 100.0 if bks_edges else None
        gap_pct = None
        if bks_cost is not None and bks_cost > 0:
            gap_pct = (int(total_cost) - int(bks_cost)) / int(bks_cost) * 100.0
        metrics = {
            "depot": int(depot_for_bks),
            "route_count": int(len(norm.routes)),
            "solver_cost": int(total_cost),
            "bks_cost": int(bks_cost) if bks_cost is not None else None,
            "gap_pct": float(gap_pct) if gap_pct is not None else None,
            "edge_pct": float(edge_pct) if edge_pct is not None else None,
            "missing_count": int(len(missing)),
            "extras_count": int(len(extras)),
            "duplicates_count": int(len(dupes)),
            "cap_violations_count": int(len(cap)),
            "valid_coverage": 1 if (not missing and not extras and not dupes) else 0,
            "valid_strict": 1 if (not missing and not extras and not dupes and not cap) else 0,
            "routes": [
                {
                    "nodes": [int(n) for n in norm.routes[i]],
                    "cost": int(per_cost[i]),
                    "load": int(sum(int(inst_for_bks.demands.get(n, 0)) for n in norm.routes[i])),
                }
                for i in range(len(norm.routes))
            ],
        }
        return metrics, [list(map(int, rt)) for rt in norm.routes]

    def write_snapshot(
        *,
        stage: str,
        cycle_idx: int,
        block_idx: int,
        axes_k: int,
        phis_active: list[float],
        phi_hint: float | None,
        iter_local: int | None,
        reason: str,
        routes_solver: list[list[int]],
    ) -> None:
        nonlocal last_snapshot_periodic_t
        elapsed = time.perf_counter() - start
        metrics, _routes_vrp_norm = compute_snapshot_metrics(routes_solver)
        payload = {
            "stage": stage,
            "cycle_idx": int(cycle_idx),
            "block_idx": int(block_idx),
            "axes_k": int(axes_k),
            "phis_active": [float(x) for x in phis_active],
            "phi_hint": float(phi_hint) if phi_hint is not None else None,
            "iter_local": int(iter_local) if iter_local is not None else None,
            "elapsed_s": float(elapsed),
            "routes": metrics["routes"],
            "solver_cost": metrics["solver_cost"],
        }
        blob = zlib.compress(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), level=6)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO gen5_v5_snapshots(
                run_id, stage, cycle_idx, block_idx, axes_k, phis_active, phi_hint, iter_local, elapsed_s, reason,
                valid_coverage, valid_strict, missing_count, extras_count, duplicates_count, cap_violations_count,
                solver_cost, bks_cost, gap_pct, edge_pct, route_count, solver_routes_zlib
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(run_id),
                str(stage),
                int(cycle_idx),
                int(block_idx),
                int(axes_k),
                ",".join(str(float(x)) for x in phis_active),
                float(phi_hint) if phi_hint is not None else None,
                int(iter_local) if iter_local is not None else None,
                float(elapsed),
                str(reason),
                int(metrics["valid_coverage"]),
                int(metrics["valid_strict"]),
                int(metrics["missing_count"]),
                int(metrics["extras_count"]),
                int(metrics["duplicates_count"]),
                int(metrics["cap_violations_count"]),
                int(metrics["solver_cost"]) if metrics["solver_cost"] is not None else None,
                int(metrics["bks_cost"]) if metrics["bks_cost"] is not None else None,
                float(metrics["gap_pct"]) if metrics["gap_pct"] is not None else None,
                float(metrics["edge_pct"]) if metrics["edge_pct"] is not None else None,
                int(metrics["route_count"]),
                sqlite3.Binary(blob),
            ),
        )
        conn.commit()
        last_snapshot_periodic_t = time.perf_counter()

    best_routes_seen_any: list[list[int]] | None = None
    best_cost_seen_any: int | None = None
    best_routes_seen_strict: list[list[int]] | None = None
    best_cost_seen_strict: int | None = None

    def is_strict_feasible(routes_solver: list[list[int]]) -> bool:
        # routes_solver already passed solver-side validate() under model.Q, so coverage/dupes are OK.
        # Here we only need to enforce original CVRP capacity Q_orig.
        Qs = int(Q_orig)
        q = model.q
        for rt in routes_solver:
            ld = 0
            for n in rt:
                ld += int(q[int(n)])
                if ld > Qs:
                    return False
        return True

    def make_on_best(
        stage: str,
        axes: int,
        phis_active: list[float],
        cycle_idx: int,
        block_idx: int,
        stn_profile: str,
    ) -> Callable[[int, int, float, float, list[list[int]]], None]:
        def _on_best(best_cost: int, it: int, phi: float, _elapsed_stage: float, best_routes: list[list[int]]) -> None:
            nonlocal last_progress_t
            nonlocal last_snapshot_best_t, best_routes_seen_any, best_cost_seen_any
            nonlocal best_routes_seen_strict, best_cost_seen_strict
            now = time.perf_counter()
            improved_any = False
            improved_strict = False

            if best_cost_seen_any is None or int(best_cost) < int(best_cost_seen_any):
                best_cost_seen_any = int(best_cost)
                best_routes_seen_any = [rt[:] for rt in best_routes]
                improved_any = True

            if is_strict_feasible(best_routes):
                if best_cost_seen_strict is None or int(best_cost) < int(best_cost_seen_strict):
                    best_cost_seen_strict = int(best_cost)
                    best_routes_seen_strict = [rt[:] for rt in best_routes]
                    improved_strict = True

            if (improved_any or improved_strict) and bool(args.snapshot_on_best):
                if now - last_snapshot_best_t >= float(args.snapshot_best_min_s):
                    last_snapshot_best_t = now
                    if improved_strict and best_routes_seen_strict is not None:
                        reason = f"best_strict/{stn_profile}"
                        snap_routes = best_routes_seen_strict
                    elif best_routes_seen_any is not None:
                        reason = f"best_any/{stn_profile}"
                        snap_routes = best_routes_seen_any
                    else:
                        reason = f"best_any/{stn_profile}"
                        snap_routes = [rt[:] for rt in best_routes]
                    write_snapshot(
                        stage=stage,
                        cycle_idx=cycle_idx,
                        block_idx=block_idx,
                        axes_k=axes,
                        phis_active=phis_active,
                        phi_hint=float(phi),
                        iter_local=int(it),
                        reason=reason,
                        routes_solver=snap_routes,
                    )

            if now - last_progress_t < float(args.progress_every_s):
                return
            last_progress_t = now
            elapsed = now - start
            report_cost = (
                int(best_cost_seen_strict)
                if best_cost_seen_strict is not None
                else (int(best_cost_seen_any) if best_cost_seen_any is not None else int(best_cost))
            )
            if bks_cost is not None and bks_cost > 0:
                gap = (report_cost - bks_cost) / bks_cost * 100.0
                gap_s = f"{gap:.2f}%"
            else:
                gap_s = "N/A"
            extra = ""
            if best_cost_seen_any is not None and best_cost_seen_strict is not None and int(best_cost_seen_any) < int(best_cost_seen_strict):
                extra = f" best_any={int(best_cost_seen_any)}"
            print(
                f"[run {run_id}] t={elapsed:.1f}s stage={stage} axes={axes} phi={phi:g} "
                f"it={it} best={report_cost} bks={bks_cost} gap={gap_s} stn={stn_profile}{extra}"
            )
            sys.stdout.flush()

        return _on_best

    k_remove_eff = max(1, min(int(args.k_remove), int(getattr(model, "N"))))

    # Stage 1: tight (smaller capacity)
    error: str | None = None
    ok = False
    solver_cost: int | None = None
    edge_pct: float | None = None
    gap_pct: float | None = None
    routes_best: list[list[int]] = []

    def run_blocks(
        *,
        stage_name: str,
        routes0: list[list[int]],
        stage_time_s: float | None,
        stage_iters: int,
        lookahead: int,
        pass_epn: bool,
    ) -> list[list[int]]:
        nonlocal best_routes_seen_any, best_cost_seen_any
        nonlocal best_routes_seen_strict, best_cost_seen_strict
        routes_cur = [list(map(int, rt)) for rt in routes0]
        best_routes = routes_cur
        best_cost = None
        if not axis_counts:
            stn_prof_name, stn_by_phi0 = stn_profile_for_block(0)
            c, routes_out = run_lns_prune(
                mod,
                model,
                routes_cur,
                stn_by_phi0,
                iters=int(stage_iters),
                k_remove=k_remove_eff,
                accept_p=float(args.accept_p),
                seed=int(args.seed),
                steps_reloc=int(args.steps_reloc),
                steps_2opt=int(args.steps_2opt),
                plateau_window=int(args.plateau_window),
                prune_batch=int(args.prune_batch),
                eps=float(args.eps),
                min_axes=int(args.min_axes),
                time_limit_s=stage_time_s,
                on_best=make_on_best(stage_name, len(stn_by_phi0), phis, 0, 0, stn_prof_name),
                lookahead=int(lookahead),
                epn=epn_state if pass_epn else None,
                epn_cfg=epn_cfg if pass_epn else None,
                full_2opt_on_best=bool(args.full_2opt_on_best)
                and (str(args.full_2opt_stage) == "all" or stage_name == "strict"),
                full_2opt_max_len=int(args.full_2opt_max_len),
                full_2opt_max_passes=int(args.full_2opt_max_passes),
            )
            _ = c
            return routes_out

        # Axis schedule enabled: run in blocks with a prefix subset of phis.
        if float(args.axis_block_time_s) > 0:
            block_time = float(args.axis_block_time_s)
            block_iters = max(1, int(stage_iters))  # large; time-limited per block
        else:
            block_time = None
            block_iters = max(1, int(args.axis_block_iters))

        stage_end = None if stage_time_s is None else (time.perf_counter() + float(stage_time_s))
        iters_remaining = int(stage_iters)
        block_idx = 0
        plateau_blocks = 0
        while True:
            now = time.perf_counter()
            best_for_snap = best_routes_seen_strict if best_routes_seen_strict is not None else best_routes_seen_any
            if float(args.snapshot_every_s) > 0 and best_for_snap is not None:
                if now - last_snapshot_periodic_t >= float(args.snapshot_every_s):
                    stn_prof_name, _stn_prof = stn_profile_for_block(block_idx)
                    write_snapshot(
                        stage=stage_name,
                        cycle_idx=0,
                        block_idx=block_idx,
                        axes_k=1 if not axis_counts else int(axis_counts[0]),
                        phis_active=phis[: 1 if not axis_counts else int(axis_counts[0])],
                        phi_hint=None,
                        iter_local=None,
                        reason=f"periodic/{stn_prof_name}",
                        routes_solver=best_for_snap,
                    )
            if stage_end is not None and time.perf_counter() >= stage_end:
                break
            if stage_end is None and iters_remaining <= 0:
                break

            for k in axis_counts:
                stn_prof_name, stn_by_phi = stn_profile_for_block(block_idx)
                if stage_end is not None:
                    remaining = stage_end - time.perf_counter()
                    if remaining <= 0:
                        return best_routes
                    t_block = remaining if block_time is None else min(remaining, block_time)
                else:
                    t_block = block_time

                if stage_end is None:
                    it_block = min(block_iters, iters_remaining)
                    if it_block <= 0:
                        return best_routes
                else:
                    it_block = int(block_iters)

                stn_sub = take_first_k_phis(stn_by_phi, phis, int(k))
                seed_block = int(args.seed) + block_idx * int(args.axis_block_seed_step) + int(k)
                best_cost_before = best_cost_seen_strict if stage_name == "strict" else best_cost_seen_any
                c, routes_out = run_lns_prune(
                    mod,
                    model,
                    routes_cur,
                    stn_sub,
                    iters=int(it_block),
                    k_remove=k_remove_eff,
                    accept_p=float(args.accept_p),
                    seed=int(seed_block),
                    steps_reloc=int(args.steps_reloc),
                    steps_2opt=int(args.steps_2opt),
                    plateau_window=int(args.plateau_window),
                    prune_batch=int(args.prune_batch),
                    eps=float(args.eps),
                    min_axes=max(1, min(int(args.min_axes), len(stn_sub))),
                    time_limit_s=t_block,
                    on_best=make_on_best(stage_name, len(stn_sub), phis[: int(k)], 0, block_idx, stn_prof_name),
                    lookahead=int(lookahead),
                    epn=epn_state if pass_epn else None,
                    epn_cfg=epn_cfg if pass_epn else None,
                    prefer_prune=(str(args.axis_schedule_mode) == "prune"),
                    full_2opt_on_best=bool(args.full_2opt_on_best)
                    and (str(args.full_2opt_stage) == "all" or stage_name == "strict"),
                    full_2opt_max_len=int(args.full_2opt_max_len),
                    full_2opt_max_passes=int(args.full_2opt_max_passes),
                )
                routes_cur = routes_out
                if best_cost is None or int(c) < int(best_cost):
                    best_cost = int(c)
                    best_routes = routes_out
                    if best_cost_seen_any is None or int(c) < int(best_cost_seen_any):
                        best_cost_seen_any = int(c)
                        best_routes_seen_any = [rt[:] for rt in best_routes]
                    if is_strict_feasible(best_routes) and (best_cost_seen_strict is None or int(c) < int(best_cost_seen_strict)):
                        best_cost_seen_strict = int(c)
                        best_routes_seen_strict = [rt[:] for rt in best_routes]
                    # Snapshot at block end for progress table (lightweight; already time-throttled for best).
                    write_snapshot(
                        stage=stage_name,
                        cycle_idx=0,
                        block_idx=block_idx,
                        axes_k=int(k),
                        phis_active=phis[: int(k)],
                        phi_hint=None,
                        iter_local=None,
                        reason=f"block_best/{stn_prof_name}",
                        routes_solver=best_routes,
                    )
                if stage_end is None:
                    iters_remaining -= int(it_block)
                    if iters_remaining <= 0:
                        return best_routes
                block_idx += 1

                # Plateau-triggered FFS restructure (strict stage only).
                if (
                    bool(args.ffs_on_plateau)
                    and stage_name == "strict"
                    and hasattr(mod, "ffs_restructure")
                    and axis_counts
                    and (best_cost_seen_strict == best_cost_before)
                ):
                    plateau_blocks += 1
                    if plateau_blocks >= max(1, int(args.ffs_plateau_blocks)):
                        plateau_blocks = 0
                        try:
                            # Use the first active phi's STN as the cleanup neighborhood.
                            phi0 = phis[0] if phis else None
                            stn_for_cleanup = stn_sub.get(float(phi0)) if phi0 is not None else None
                            cfg = getattr(mod, "FFSConfig")(
                                far_percentile=float(args.ffs_far_percentile),
                                angle_bins=int(args.ffs_angle_bins),
                                angle_bin_radius=int(args.ffs_angle_bin_radius),
                                open_route_if_better=bool(args.ffs_open_route_if_better),
                                open_route_margin=float(args.ffs_open_route_margin),
                                post_steps_reloc=int(args.ffs_post_steps_reloc),
                                post_steps_2opt=int(args.ffs_post_steps_2opt),
                            )
                            routes_input = (
                                best_routes_seen_strict
                                if best_routes_seen_strict is not None
                                else (best_routes_seen_any if best_routes_seen_any is not None else routes_cur)
                            )
                            routes_ffs = mod.ffs_restructure(
                                model,
                                routes_input,
                                stn=stn_for_cleanup,
                                cfg=cfg,
                                seed=int(seed_block) + 99991,
                            )
                            routes_cur = routes_ffs
                            # Always snapshot the restructure (even if not improving).
                            write_snapshot(
                                stage=stage_name,
                                cycle_idx=0,
                                block_idx=block_idx,
                                axes_k=int(k),
                                phis_active=phis[: int(k)],
                                phi_hint=None,
                                iter_local=None,
                                reason=f"ffs/{stn_prof_name}",
                                routes_solver=routes_cur,
                            )
                        except Exception:
                            pass
                else:
                    best_cost_after = best_cost_seen_strict if stage_name == "strict" else best_cost_seen_any
                    if best_cost_after != best_cost_before:
                        plateau_blocks = 0
        return best_routes

    try:
        if args.use_schedule:
            if t_deadline is not None and time.perf_counter() >= t_deadline:
                raise TimeoutError("time_limit_s exhausted before schedule started")
            # Allocate wall-clock budget across tight/soft/strict if requested.
            total_budget = None if t_deadline is None else max(0.0, t_deadline - start)
            t_tight = None
            t_soft = None
            t_strict = None
            if total_budget is not None:
                t_tight = total_budget * 0.20
                t_soft = total_budget * 0.40
                t_strict = total_budget * 0.40

            model.Q = max(1, int(Q_orig * float(args.tight_factor)))
            prof0, stn0 = stn_profile_for_block(0)
            base_routes = mod.construct_base(model, stn0[phis[0]], seed=int(args.seed))
            routes_t = run_blocks(
                stage_name="tight",
                routes0=base_routes,
                stage_time_s=t_tight,
                stage_iters=int(args.tight_iters),
                lookahead=int(args.strict_lookahead),
                pass_epn=False,
            )

            # Stage 2: soft (larger capacity)
            model.Q = max(1, int(Q_orig * float(args.soft_factor)))
            routes_s = run_blocks(
                stage_name="soft",
                routes0=routes_t,
                stage_time_s=t_soft,
                stage_iters=int(args.soft_iters),
                lookahead=int(args.soft_lookahead),
                pass_epn=not bool(args.no_epn_soft),
            )

            # Stage 3: strict repair + strict LNS
            model.Q = Q_orig
            # Configure strict repair preferences (protect far / high detour / high demand).
            model._repair_w_r2 = float(args.repair_w_r2)
            model._repair_w_detour = float(args.repair_w_detour)
            model._repair_w_q = float(args.repair_w_q)
            model._repair_topk = int(args.repair_topk)
            model._repair_max_steps = int(args.repair_max_steps)
            routes_r = strict_repair_to_capacity(model, routes_s)
            routes_best = run_blocks(
                stage_name="strict",
                routes0=routes_r,
                stage_time_s=t_strict,
                stage_iters=int(args.strict_iters),
                lookahead=int(args.strict_lookahead),
                pass_epn=False,
            )
        else:
            t_strict = None if t_deadline is None else max(0.0, t_deadline - start)
            # use current STN profile (base) for construction
            _prof0, stn0 = stn_profile_for_block(0)
            base_routes = mod.construct_base(model, stn0[phis[0]], seed=int(args.seed))
            routes_best = run_blocks(
                stage_name="strict",
                routes0=base_routes,
                stage_time_s=t_strict,
                stage_iters=int(args.strict_iters),
                lookahead=int(args.strict_lookahead),
                pass_epn=False,
            )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        routes_best = []

    runtime = time.perf_counter() - start

    if routes_best and error is None:
        # Prefer the global best snapshot (monotonic) over the last block's return value,
        # because time limits can expire mid-block and return a worse current solution.
        preferred = (
            best_routes_seen_strict
            if best_routes_seen_strict is not None
            else (best_routes_seen_any if best_routes_seen_any is not None else routes_best)
        )
        vrp_routes, depot = solver_routes_to_vrp(mod, args.vrp, preferred)
        ok, solver_cost, vrp_routes_norm = validate_vrp_routes(args.vrp, vrp_routes)
        if not ok or solver_cost is None:
            # Fallback to the stage return value if the preferred snapshot is invalid.
            vrp_routes, depot = solver_routes_to_vrp(mod, args.vrp, routes_best)
            ok, solver_cost, vrp_routes_norm = validate_vrp_routes(args.vrp, vrp_routes)
            if not ok or solver_cost is None:
                ok = False
                solver_cost = None
                error = "Solver produced invalid routes under strict CVRP validation."
    else:
        depot = parse_vrp(args.vrp).depots[0]
        vrp_routes_norm = []

    if ok and solver_cost is not None:
        inst = parse_vrp(args.vrp)
        out_routes = to_customer_indexing(inst, vrp_routes_norm)
        args.out_sol.write_text(format_solution(out_routes, solver_cost))

        edge_pct = None
        gap_pct = None
        if bks_edges is not None:
            sol_edges = edges_undirected(vrp_routes_norm, depot)
            if bks_edges:
                edge_pct = len(bks_edges & sol_edges) / len(bks_edges) * 100.0
        if bks_cost is not None and bks_cost > 0:
            gap_pct = (solver_cost - bks_cost) / bks_cost * 100.0

    # Always record a final snapshot of the best routes we are about to report.
    try:
        final_routes = (
            best_routes_seen_strict
            if best_routes_seen_strict is not None
            else (best_routes_seen_any if best_routes_seen_any is not None else routes_best)
        )
        if final_routes:
            write_snapshot(
                stage="final",
                cycle_idx=0,
                block_idx=-1,
                axes_k=len(phis) if phis else 1,
                phis_active=phis,
                phi_hint=None,
                iter_local=None,
                reason="final",
                routes_solver=final_routes,
            )
    except Exception:
        pass

    print(f"Instance: {args.vrp}")
    if ok and solver_cost is not None:
        print(f"Solver cost: {solver_cost}")
        if bks_cost is not None:
            print(f"BKS cost: {bks_cost}")
            print(f"Gap: {fmt_pct(gap_pct, 2)}")
            print(f"Edge matched: {fmt_pct(edge_pct, 1)}")
        print("Valid routes: 1")
        print(f"Run time (s): {runtime:.2f}")
        print(f"Wrote: {args.out_sol}")
    else:
        print("Valid routes: 0")
        print(f"Run time (s): {runtime:.2f}")
        print(f"ERROR: {error}")

    log_run_result(
        conn,
        run_id=run_id,
        instance_name=str(args.vrp.name),
        valid=ok and solver_cost is not None,
        solver_cost=solver_cost,
        bks_cost=bks_cost,
        gap_pct=gap_pct,
        edge_pct=edge_pct,
        runtime_s=runtime,
        error=error,
    )
    conn.close()
    print(f"Run ID: {run_id}")
    return 0 if ok and solver_cost is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
