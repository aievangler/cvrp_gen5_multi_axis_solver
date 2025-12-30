#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# -----------------------------
# Problem parsing (TSPLIB/CVRP)
# -----------------------------


@dataclass
class Problem:
    name: str
    dim: int
    Q: int
    depot_orig: int
    coords: np.ndarray  # (dim+1,2)
    demand: np.ndarray  # (dim+1,)
    ewt: str
    ewf: str
    weights: Optional[List[int]]  # for EXPLICIT


@dataclass
class Model:
    N: int
    Q: int
    x: np.ndarray  # (N+1,)
    y: np.ndarray  # (N+1,)
    q: np.ndarray  # (N+1,)
    sol_to_orig: np.ndarray  # (N+1,)
    orig_to_sol: Dict[int, int]
    D_obj: np.ndarray  # (N+1,N+1) int32 objective distance
    D_geo: np.ndarray  # (N+1,N+1) int32 geometric distance (heuristic only)


def parse_vrp(path: str) -> Problem:
    name = None
    dim = None
    Q = None
    ewt = None
    ewf = ""
    coords = {}
    demand = {}
    depot_ids = []
    weights = None

    def clean(s: str) -> str:
        return s.strip()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [clean(x) for x in f if clean(x)]

    section = None
    for ln in lines:
        up = ln.upper()

        if ":" in ln and section is None:
            k, v = ln.split(":", 1)
            k = k.strip().upper()
            v = v.strip()
            if k == "NAME":
                name = v
            elif k == "DIMENSION":
                dim = int(v)
            elif k == "CAPACITY":
                Q = int(float(v))
            elif k == "EDGE_WEIGHT_TYPE":
                ewt = v.upper()
            elif k == "EDGE_WEIGHT_FORMAT":
                ewf = v.upper()
            continue

        if up == "NODE_COORD_SECTION":
            section = "COORD"
            continue
        if up == "DEMAND_SECTION":
            section = "DEMAND"
            continue
        if up == "DEPOT_SECTION":
            section = "DEPOT"
            continue
        if up == "EDGE_WEIGHT_SECTION":
            section = "EDGE"
            weights = []
            continue
        if up == "EOF":
            break

        if section == "COORD":
            a, x, y = ln.split()[:3]
            coords[int(a)] = (float(x), float(y))
        elif section == "DEMAND":
            a, d = ln.split()[:2]
            demand[int(a)] = int(float(d))
        elif section == "DEPOT":
            if ln.startswith("-1"):
                section = None
            else:
                depot_ids.append(int(ln))
        elif section == "EDGE":
            assert weights is not None
            weights.extend([int(x) for x in ln.split()])

    if dim is None or Q is None or ewt is None:
        raise ValueError("Missing DIMENSION/CAPACITY/EDGE_WEIGHT_TYPE in vrp")

    depot = depot_ids[0] if depot_ids else 1

    coord_arr = np.zeros((dim + 1, 2), dtype=np.float64)
    dem_arr = np.zeros((dim + 1,), dtype=np.int32)
    for i in range(1, dim + 1):
        if i in coords:
            coord_arr[i, 0] = coords[i][0]
            coord_arr[i, 1] = coords[i][1]
        dem_arr[i] = int(demand.get(i, 0))

    return Problem(
        name=name or os.path.basename(path),
        dim=dim,
        Q=Q,
        depot_orig=depot,
        coords=coord_arr,
        demand=dem_arr,
        ewt=ewt,
        ewf=ewf,
        weights=weights,
    )


def dist_euc2d_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0] - 1
    D = np.zeros((n + 1, n + 1), dtype=np.int32)
    for i in range(n + 1):
        xi = x[i]
        yi = y[i]
        dx = xi - x
        dy = yi - y
        D[i] = (np.hypot(dx, dy) + 0.5).astype(np.int32)
    return D


def build_model(prob: Problem) -> Model:
    depot = prob.depot_orig
    customers = [i for i in range(1, prob.dim + 1) if i != depot]
    N = len(customers)

    sol_to_orig = np.zeros((N + 1,), dtype=np.int32)
    sol_to_orig[0] = depot
    orig_to_sol = {depot: 0}
    for sid, oid in enumerate(customers, start=1):
        sol_to_orig[sid] = oid
        orig_to_sol[oid] = sid

    x = np.zeros((N + 1,), dtype=np.float64)
    y = np.zeros((N + 1,), dtype=np.float64)
    q = np.zeros((N + 1,), dtype=np.int32)
    for sid in range(N + 1):
        oid = int(sol_to_orig[sid])
        x[sid] = prob.coords[oid, 0]
        y[sid] = prob.coords[oid, 1]
        q[sid] = int(prob.demand[oid])

    D_geo = dist_euc2d_matrix(x, y)

    if prob.ewt == "EUC_2D":
        D_obj = D_geo.copy()
    else:
        if prob.weights is None:
            raise ValueError("EXPLICIT instance missing EDGE_WEIGHT_SECTION weights")
        dim = prob.dim
        W = np.zeros((dim + 1, dim + 1), dtype=np.int32)
        w = prob.weights
        idx = 0
        for i in range(2, dim + 1):
            for j in range(1, i):
                W[i, j] = int(w[idx])
                W[j, i] = int(w[idx])
                idx += 1
        D_obj = np.zeros((N + 1, N + 1), dtype=np.int32)
        for a in range(N + 1):
            oa = int(sol_to_orig[a])
            for b in range(N + 1):
                ob = int(sol_to_orig[b])
                D_obj[a, b] = W[oa, ob]

    return Model(
        N=N,
        Q=prob.Q,
        x=x,
        y=y,
        q=q,
        sol_to_orig=sol_to_orig,
        orig_to_sol=orig_to_sol,
        D_obj=D_obj,
        D_geo=D_geo,
    )

@dataclass(frozen=True)
class FFSConfig:
    far_percentile: float = 0.80  # top (1-p) are "far"
    angle_bins: int = 64
    angle_bin_radius: int = 2
    # If inserting into any existing route is worse than opening a fresh route for the node,
    # allow creating a new route (K can grow).
    open_route_if_better: bool = True
    open_route_margin: float = 1.00
    # Local-improve cleanup after rebuilding (recommended)
    post_steps_reloc: int = 160
    post_steps_2opt: int = 120


def _angle_bin(model: Model, node: int, nbins: int) -> int:
    x0 = float(model.x[0])
    y0 = float(model.y[0])
    ang = math.atan2(float(model.y[node]) - y0, float(model.x[node]) - x0)  # [-pi,pi]
    t = (ang + math.pi) / (2.0 * math.pi)  # [0,1]
    b = int(t * nbins)
    if b < 0:
        return 0
    if b >= nbins:
        return nbins - 1
    return b


def ffs_restructure(
    model: Model,
    routes0: List[List[int]],
    *,
    stn: np.ndarray | None = None,
    cfg: FFSConfig | None = None,
    seed: int = 21,
) -> List[List[int]]:
    """
    Farthest-First Seeding restructure (strict-feasible).
    - Seeds routes with far customers first (from depot).
    - Inserts remaining far nodes first, then the rest.
    - Route count may grow if needed for feasibility.
    - Deterministic for a given seed/config (tie-breaks by node id / route id).
    """
    cfg = cfg or FFSConfig()
    random.seed(int(seed))

    N = int(model.N)
    Q = int(model.Q)
    D = model.D_obj

    # K floor = current non-empty routes if available; otherwise a conservative lower bound.
    base_routes = cleanup_routes([rt[:] for rt in routes0])
    K0 = len(base_routes)
    if K0 <= 0:
        total_demand = int(model.q[1:].sum())
        K0 = max(1, (total_demand + Q - 1) // Q)

    # Precompute radial distance^2 from depot and angle bins.
    dx = model.x[1:] - float(model.x[0])
    dy = model.y[1:] - float(model.y[0])
    r2 = (dx * dx + dy * dy).astype(np.float64)
    # nodes are 1..N; r2 index 0 corresponds to node 1
    order = list(range(1, N + 1))
    order.sort(key=lambda i: (float(r2[i - 1]), i), reverse=True)

    # Far threshold by percentile of r2.
    p = float(cfg.far_percentile)
    p = 0.0 if p < 0 else (1.0 if p > 1 else p)
    if N <= 1:
        far_thresh = float(r2[0]) if N == 1 else 0.0
    else:
        r2_sorted = np.sort(r2)
        idx = int(min(N - 1, max(0, math.floor(p * (N - 1)))))
        far_thresh = float(r2_sorted[idx])
    is_far = np.zeros((N + 1,), dtype=np.uint8)
    for i in range(1, N + 1):
        if float(r2[i - 1]) >= far_thresh:
            is_far[i] = 1

    nbins = max(8, int(cfg.angle_bins))
    bin_radius = max(0, int(cfg.angle_bin_radius))
    node_bin = [0] * (N + 1)
    for i in range(1, N + 1):
        node_bin[i] = _angle_bin(model, i, nbins)

    assigned = np.zeros((N + 1,), dtype=np.uint8)
    assigned[0] = 1

    routes: List[List[int]] = []
    loads: List[int] = []
    seed_bins: List[int] = []
    bin_routes: List[List[int]] = [[] for _ in range(nbins)]

    def add_route(seed_node: int) -> int:
        rid = len(routes)
        routes.append([int(seed_node)])
        loads.append(int(model.q[int(seed_node)]))
        b = int(node_bin[int(seed_node)])
        seed_bins.append(b)
        bin_routes[b].append(rid)
        assigned[int(seed_node)] = 1
        return rid

    # Seed K0 routes with farthest unassigned nodes (prefer far nodes, but fallback to any).
    seeds_used = 0
    for i in order:
        if assigned[i]:
            continue
        if seeds_used < K0 and int(model.q[i]) <= Q:
            add_route(i)
            seeds_used += 1
            if seeds_used >= K0:
                break
    if seeds_used < K0:
        for i in order:
            if assigned[i]:
                continue
            if int(model.q[i]) <= Q:
                add_route(i)
                seeds_used += 1
                if seeds_used >= K0:
                    break

    def candidate_route_ids_for(node: int) -> List[int]:
        b = int(node_bin[int(node)])
        if bin_radius <= 0:
            return bin_routes[b][:]
        cand: List[int] = []
        for db in range(-bin_radius, bin_radius + 1):
            bb = (b + db) % nbins
            cand.extend(bin_routes[bb])
        if cand:
            # Deterministic order.
            cand.sort()
            return cand
        return list(range(len(routes)))

    def best_end_insertion(rid: int, node: int) -> tuple[int, int]:
        rt = routes[rid]
        if not rt:
            return (0, 0)
        first = int(rt[0])
        last = int(rt[-1])
        d_start = int(D[0, node]) + int(D[node, first]) - int(D[0, first])
        d_end = int(D[last, node]) + int(D[node, 0]) - int(D[last, 0])
        if d_start < d_end:
            return d_start, 0
        return d_end, len(rt)

    def insert_node(node: int) -> None:
        nonlocal routes, loads, seed_bins, bin_routes
        if assigned[int(node)]:
            return
        demand = int(model.q[int(node)])
        if demand > Q:
            raise ValueError(f"FFS cannot place node demand>{Q}: node={node} demand={demand}")
        best_delta = None
        best_rid = None
        best_pos = None

        # Prefer angle-near routes; fallback to all routes; if still none feasible, open a new route.
        cand_lists = [candidate_route_ids_for(node), list(range(len(routes)))]
        tried_all = False
        for cand in cand_lists:
            if tried_all:
                break
            if cand and cand is cand_lists[1]:
                tried_all = True
            for rid in cand:
                if loads[rid] + demand > Q:
                    continue
                delta, pos = best_end_insertion(rid, int(node))
                key = (int(delta), int(rid), int(pos))
                if best_delta is None or key < (int(best_delta), int(best_rid), int(best_pos)):  # type: ignore[arg-type]
                    best_delta = int(delta)
                    best_rid = int(rid)
                    best_pos = int(pos)
            if best_rid is not None:
                break

        if best_rid is None:
            add_route(int(node))
            return

        if bool(cfg.open_route_if_better):
            new_route_cost = int(D[0, int(node)]) + int(D[int(node), 0])
            if float(best_delta) > float(cfg.open_route_margin) * float(new_route_cost):
                add_route(int(node))
                return

        routes[int(best_rid)].insert(int(best_pos), int(node))
        loads[int(best_rid)] += demand
        assigned[int(node)] = 1

    # Phase 1: assign remaining far nodes first, in descending radial order.
    for i in order:
        if assigned[i]:
            continue
        if is_far[i]:
            insert_node(i)

    # Phase 2: assign the rest.
    for i in order:
        if assigned[i]:
            continue
        insert_node(i)

    out = cleanup_routes(routes)
    if not validate(model, out):
        raise ValueError("FFS produced invalid (missing/dup/cap) solution")

    # Optional post-cleanup via existing local-improve, using the provided STN.
    if int(cfg.post_steps_reloc) > 0 or int(cfg.post_steps_2opt) > 0:
        out2 = local_improve(
            model,
            out,
            stn if stn is not None else np.zeros((N + 1, 1), dtype=np.int32),
            steps_reloc=int(cfg.post_steps_reloc),
            steps_2opt=int(cfg.post_steps_2opt),
        )
        out2 = cleanup_routes(out2)
        if validate(model, out2):
            return out2
    return out


# -----------------------------
# EPN (Edge Proximity Neighbors)
# -----------------------------


@dataclass(frozen=True)
class EPNConfig:
    enabled: bool = True
    # KNN1 exclusion set: reject w ∈ KNN1[u] ∪ KNN1[v]
    k1: int = 32
    # Grid build target occupancy
    target_cell_occ: int = 16
    # Query caps
    k_epn: int = 16
    max_cells: int = 25
    max_tested: int = 1024
    margin_cells: int = 1
    # Triggering
    long_edges_per_route: int = 10
    include_depot_edges: bool = True
    # Filters
    exclude_knn1: bool = True
    # Keep points within d2_beta * edge_len2
    d2_beta: float = 0.04
    # Apply budget per call
    max_moves_per_call: int = 12
    allow_intra_route: bool = False


@dataclass
class EPNState:
    nx: int
    ny: int
    xmin: float
    ymin: float
    inv_w: float
    inv_h: float
    cell_start: np.ndarray  # int32, len=nx*ny+1
    cell_nodes: np.ndarray  # int32, customers only
    knn1: np.ndarray  # int32, shape=(N+1, k1_eff)
    mark: list[int]
    stamp: int = 1


def build_knn1_geo(model: Model, k1: int) -> np.ndarray:
    D = model.D_geo
    n = int(model.N)
    k1_eff = max(1, min(int(k1), max(1, n - 1)))
    out = np.full((n + 1, k1_eff), 0, dtype=np.int32)
    if n <= 1:
        return out
    for i in range(1, n + 1):
        row = D[i, 1 : n + 1].copy()
        row[i - 1] = 10**9
        kk = min(k1_eff, row.shape[0])
        idx = np.argpartition(row, kk - 1)[:kk]
        neigh = (idx + 1).astype(np.int32)
        neigh = neigh[np.argsort(D[i, neigh])]
        out[i, : neigh.shape[0]] = neigh[:k1_eff]
    return out


def epn_build_grid(model: Model, target_cell_occ: int) -> tuple[int, int, float, float, float, float, np.ndarray, np.ndarray]:
    x = model.x
    y = model.y
    n = int(model.N)
    occ = max(4, int(target_cell_occ))
    ncells_target = int(np.clip(max(64, n // occ), 64, 4096))
    nx = int(np.clip(int(math.sqrt(ncells_target)), 8, 256))
    ny = int(np.clip(int(math.ceil(ncells_target / max(1, nx))), 8, 256))

    xmin = float(np.min(x[1:])) if n > 0 else float(x[0])
    xmax = float(np.max(x[1:])) if n > 0 else float(x[0])
    ymin = float(np.min(y[1:])) if n > 0 else float(y[0])
    ymax = float(np.max(y[1:])) if n > 0 else float(y[0])
    w = max(1e-9, (xmax - xmin) / float(nx))
    h = max(1e-9, (ymax - ymin) / float(ny))
    inv_w = 1.0 / w
    inv_h = 1.0 / h

    cx = np.clip(((x[1:] - xmin) * inv_w).astype(np.int32), 0, nx - 1)
    cy = np.clip(((y[1:] - ymin) * inv_h).astype(np.int32), 0, ny - 1)
    cell_id = (cx * ny + cy).astype(np.int32)
    order = np.argsort(cell_id, kind="stable")
    nodes_sorted = (order + 1).astype(np.int32)
    cell_sorted = cell_id[order]
    counts = np.bincount(cell_sorted, minlength=nx * ny).astype(np.int32)
    cell_start = np.zeros((nx * ny + 1,), dtype=np.int32)
    np.cumsum(counts, out=cell_start[1:])
    return nx, ny, xmin, ymin, inv_w, inv_h, cell_start, nodes_sorted


def build_epn_state(model: Model, cfg: EPNConfig) -> EPNState:
    nx, ny, xmin, ymin, inv_w, inv_h, cell_start, cell_nodes = epn_build_grid(model, cfg.target_cell_occ)
    knn1 = build_knn1_geo(model, cfg.k1)
    mark = [0] * (int(model.N) + 1)
    return EPNState(
        nx=nx,
        ny=ny,
        xmin=xmin,
        ymin=ymin,
        inv_w=inv_w,
        inv_h=inv_h,
        cell_start=cell_start,
        cell_nodes=cell_nodes,
        knn1=knn1,
        mark=mark,
        stamp=1,
    )


def _point_segment_d2(px: np.ndarray, py: np.ndarray, ax: float, ay: float, bx: float, by: float) -> np.ndarray:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-12:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy
    t = (apx * abx + apy * aby) / denom
    t = np.clip(t, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    dx = px - cx
    dy = py - cy
    return dx * dx + dy * dy


def epn_query_edge(
    model: Model,
    epn: EPNState,
    cfg: EPNConfig,
    u: int,
    v: int,
    *,
    counters: dict[str, int] | None = None,
) -> list[int]:
    n = int(model.N)
    if u < 0 or u > n or v < 0 or v > n or u == v:
        return []
    ax = float(model.x[u])
    ay = float(model.y[u])
    bx = float(model.x[v])
    by = float(model.y[v])

    xmin = min(ax, bx)
    xmax = max(ax, bx)
    ymin = min(ay, by)
    ymax = max(ay, by)

    cx0 = int(math.floor((xmin - epn.xmin) * epn.inv_w)) - int(cfg.margin_cells)
    cx1 = int(math.floor((xmax - epn.xmin) * epn.inv_w)) + int(cfg.margin_cells)
    cy0 = int(math.floor((ymin - epn.ymin) * epn.inv_h)) - int(cfg.margin_cells)
    cy1 = int(math.floor((ymax - epn.ymin) * epn.inv_h)) + int(cfg.margin_cells)
    cx0 = max(0, min(epn.nx - 1, cx0))
    cx1 = max(0, min(epn.nx - 1, cx1))
    cy0 = max(0, min(epn.ny - 1, cy0))
    cy1 = max(0, min(epn.ny - 1, cy1))

    tested_nodes: list[int] = []
    cells_scanned = 0
    for cx in range(cx0, cx1 + 1):
        for cy in range(cy0, cy1 + 1):
            if cells_scanned >= int(cfg.max_cells):
                break
            cells_scanned += 1
            cid = cx * epn.ny + cy
            s = int(epn.cell_start[cid])
            e = int(epn.cell_start[cid + 1])
            if e <= s:
                continue
            chunk = epn.cell_nodes[s:e]
            remaining = int(cfg.max_tested) - len(tested_nodes)
            if remaining <= 0:
                break
            if chunk.shape[0] > remaining:
                tested_nodes.extend(chunk[:remaining].tolist())
            else:
                tested_nodes.extend(chunk.tolist())
        if cells_scanned >= int(cfg.max_cells) or len(tested_nodes) >= int(cfg.max_tested):
            break

    if counters is not None:
        counters["epn_cells_scanned"] = counters.get("epn_cells_scanned", 0) + int(cells_scanned)
        counters["epn_points_tested"] = counters.get("epn_points_tested", 0) + int(len(tested_nodes))

    if not tested_nodes:
        return []

    cand = np.asarray(tested_nodes, dtype=np.int32)
    cand = cand[(cand != u) & (cand != v)]
    if cand.size == 0:
        return []

    d2 = _point_segment_d2(model.x[cand], model.y[cand], ax, ay, bx, by)
    k = min(int(cfg.k_epn), int(cand.size))
    if k <= 0:
        return []
    if cand.size > k:
        idx = np.argpartition(d2, k - 1)[:k]
        cand = cand[idx]
        d2 = d2[idx]
    order = np.argsort(d2, kind="stable")
    cand = cand[order]
    d2 = d2[order]

    dx = bx - ax
    dy = by - ay
    edge_d2 = dx * dx + dy * dy
    d2_thresh = float(cfg.d2_beta) * float(edge_d2)
    if d2_thresh > 0:
        cand = cand[d2 <= d2_thresh]
    return cand.tolist()


def _select_long_edges_for_route(
    model: Model,
    rt: list[int],
    *,
    include_depot: bool,
    long_edges_per_route: int,
) -> list[tuple[int, int, int]]:
    if not rt:
        return []
    seq = [0] + rt + [0]
    edges: list[tuple[float, int, int, int]] = []
    for i in range(len(seq) - 1):
        u = int(seq[i])
        v = int(seq[i + 1])
        if not include_depot and (u == 0 or v == 0):
            continue
        dx = float(model.x[u]) - float(model.x[v])
        dy = float(model.y[u]) - float(model.y[v])
        d2 = dx * dx + dy * dy
        insert_pos = int(i)
        edges.append((d2, u, v, insert_pos))
    if not edges:
        return []
    edges.sort(reverse=True)
    out: list[tuple[int, int, int]] = []
    for _d2, u, v, pos in edges[: max(1, int(long_edges_per_route))]:
        out.append((u, v, pos))
    return out


def _epn_filter_exclude_knn1(epn: EPNState, cfg: EPNConfig, u: int, v: int, w_list: list[int]) -> list[int]:
    if not cfg.exclude_knn1 or not w_list:
        return w_list
    epn.stamp += 1
    if epn.stamp >= 2_000_000_000:
        for i in range(len(epn.mark)):
            epn.mark[i] = 0
        epn.stamp = 1
    st = epn.stamp
    if u > 0:
        for nn in epn.knn1[u].tolist():
            if nn > 0:
                epn.mark[int(nn)] = st
    if v > 0:
        for nn in epn.knn1[v].tolist():
            if nn > 0:
                epn.mark[int(nn)] = st
    out: list[int] = []
    for w in w_list:
        if w > 0 and epn.mark[int(w)] != st:
            out.append(int(w))
    return out


def epn_topup_long_edges(
    model: Model,
    routes: list[list[int]],
    *,
    epn: EPNState,
    cfg: EPNConfig,
    counters: dict[str, int] | None = None,
) -> list[list[int]]:
    if not cfg.enabled:
        return routes
    D = model.D_obj
    Q = int(model.Q)
    routes = cleanup_routes([rt[:] for rt in routes])
    if not routes:
        return routes
    loads = [int(sum(int(model.q[n]) for n in rt)) for rt in routes]
    ro = build_route_of(model, routes)

    def removal_delta(rt_from: list[int], idx: int) -> int:
        a = 0 if idx == 0 else rt_from[idx - 1]
        node = rt_from[idx]
        b = 0 if idx == len(rt_from) - 1 else rt_from[idx + 1]
        return int(D[a, b]) - int(D[a, node]) - int(D[node, b])

    moves_applied = 0
    while moves_applied < int(cfg.max_moves_per_call):
        best = None
        best_move = None

        for rid_tgt, rt in enumerate(routes):
            if not rt:
                continue
            edges = _select_long_edges_for_route(
                model,
                rt,
                include_depot=bool(cfg.include_depot_edges),
                long_edges_per_route=int(cfg.long_edges_per_route),
            )
            if counters is not None:
                counters["epn_edges_triggered"] = counters.get("epn_edges_triggered", 0) + int(len(edges))
            for u, v, insert_pos in edges:
                w_list = epn_query_edge(model, epn, cfg, int(u), int(v), counters=counters)
                w_list = _epn_filter_exclude_knn1(epn, cfg, int(u), int(v), w_list)
                if not w_list:
                    continue

                for w in w_list:
                    rid_from = int(ro[int(w)])
                    if rid_from < 0:
                        continue
                    if not cfg.allow_intra_route and rid_from == rid_tgt:
                        continue
                    rt_from = routes[rid_from]
                    if not rt_from:
                        continue
                    try:
                        idx_w = rt_from.index(int(w))
                    except ValueError:
                        continue
                    if rid_from != rid_tgt and loads[rid_tgt] + int(model.q[int(w)]) > Q:
                        continue
                    drem = removal_delta(rt_from, idx_w)
                    ins = int(D[int(u), int(w)]) + int(D[int(w), int(v)]) - int(D[int(u), int(v)])
                    delta = drem + ins
                    if delta >= 0:
                        continue
                    if best is None or delta < best:
                        best = int(delta)
                        best_move = (rid_from, idx_w, rid_tgt, int(insert_pos), int(w))

        if best_move is None:
            break

        rid_from, idx_w, rid_tgt, insert_pos, w = best_move
        rt_from = routes[rid_from]
        rt_tgt = routes[rid_tgt]
        if idx_w < 0 or idx_w >= len(rt_from) or rt_from[idx_w] != w:
            break

        rt_from.pop(idx_w)
        loads[rid_from] -= int(model.q[w])
        if rid_from == rid_tgt and insert_pos > idx_w:
            insert_pos -= 1
        insert_pos = max(0, min(insert_pos, len(rt_tgt)))
        rt_tgt.insert(insert_pos, w)
        loads[rid_tgt] += int(model.q[w])
        ro[w] = int(rid_tgt)

        if not rt_from:
            routes[rid_from] = []
            loads[rid_from] = 0
        moves_applied += 1
        if counters is not None:
            counters["epn_moves_applied"] = counters.get("epn_moves_applied", 0) + 1

    return cleanup_routes(routes)


# -----------------------------
# Feature computation
# -----------------------------


def compute_r_features(model: Model) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = model.x - model.x[0]
    dy = model.y - model.y[0]
    r = np.hypot(dx, dy)
    rmax = float(np.max(r[1:])) if model.N > 0 else 1.0
    rN = r / (rmax if rmax > 0 else 1.0)
    r2 = rN * rN
    return dx, dy, r2


def compute_y2(dx: np.ndarray, dy: np.ndarray, phi_deg: float) -> np.ndarray:
    phi = math.radians(phi_deg)
    c = math.cos(phi)
    s = math.sin(phi)
    y = dx * c + dy * s
    ymax = float(np.max(np.abs(y[1:]))) if y.shape[0] > 1 else 1.0
    yN = y / (ymax if ymax > 0 else 1.0)
    return yN * yN


# -----------------------------
# STN builder (probe windows + pools)
# -----------------------------


def build_stn_probe(
    model: Model,
    dx: np.ndarray,
    dy: np.ndarray,
    r2: np.ndarray,
    phi_deg: float,
    K_geo_pool: int = 160,
    K_r_pool: int = 80,
    K_y_pool: int = 160,
    K_final: int = 96,
    *,
    r2_penalty_frac: float = 0.0,
    far_y_mult: float = 0.0,
    far_r_mult: float = 0.0,
) -> np.ndarray:
    N = model.N
    D_geo = model.D_geo
    D_obj = model.D_obj

    # r ordering for window
    r = np.hypot(dx, dy)
    rmax = float(np.max(r[1:])) if N > 0 else 1.0
    rN = r / (rmax if rmax > 0 else 1.0)
    order_r = np.argsort(rN[1:]) + 1
    pos_r = np.zeros((N + 1,), dtype=np.int32)
    pos_r[order_r] = np.arange(N, dtype=np.int32)

    # y ordering for window
    phi = math.radians(phi_deg)
    c = math.cos(phi)
    s = math.sin(phi)
    y = dx * c + dy * s
    ymax = float(np.max(np.abs(y[1:]))) if N > 0 else 1.0
    yN = y / (ymax if ymax > 0 else 1.0)
    order_y = np.argsort(yN[1:]) + 1
    pos_y = np.zeros((N + 1,), dtype=np.int32)
    pos_y[order_y] = np.arange(N, dtype=np.int32)

    # clamp pools
    N1 = max(1, N - 1)
    K_geo_pool = min(int(K_geo_pool), N1)
    K_r_pool = min(int(K_r_pool), N1)
    K_y_pool = min(int(K_y_pool), N1)
    K_final = min(int(K_final), N1)

    stn = np.zeros((N + 1, K_final), dtype=np.int32)

    mark = np.zeros((N + 1,), dtype=np.int32)
    stamp = 1

    r2_penalty_frac = float(r2_penalty_frac)
    far_y_mult = float(far_y_mult)
    far_r_mult = float(far_r_mult)
    r2_penalty_scale = 0.0
    if r2_penalty_frac > 0.0:
        # Use depot-distance scale so r2 penalty is in "distance units".
        r2_penalty_scale = float(np.max(D_geo[0, 1:])) if N > 0 else 1.0
        if r2_penalty_scale <= 0:
            r2_penalty_scale = 1.0

    for i in range(1, N + 1):
        stamp += 1
        cand: List[int] = []

        # geo pool
        d = D_geo[i].copy()
        d[i] = 10**9
        kth = max(0, min(K_geo_pool - 1, d.shape[0] - 1))
        nn = np.argpartition(d, kth)[:K_geo_pool]
        for j in nn:
            mark[int(j)] = stamp
            cand.append(int(j))

        # r window pool
        Kr = int(K_r_pool)
        if far_r_mult != 0.0:
            Kr = int(min(N1, max(1, round(float(K_r_pool) * (1.0 + far_r_mult * float(r2[i]))))))
        pi = int(pos_r[i])
        lo = max(0, pi - Kr // 2)
        hi = min(N, lo + Kr)
        for j in order_r[lo:hi]:
            jj = int(j)
            if jj != i and mark[jj] != stamp:
                mark[jj] = stamp
                cand.append(jj)

        # y window pool
        Ky = int(K_y_pool)
        if far_y_mult != 0.0:
            Ky = int(min(N1, max(1, round(float(K_y_pool) * (1.0 + far_y_mult * float(r2[i]))))))
        pi = int(pos_y[i])
        lo = max(0, pi - Ky // 2)
        hi = min(N, lo + Ky)
        for j in order_y[lo:hi]:
            jj = int(j)
            if jj != i and mark[jj] != stamp:
                mark[jj] = stamp
                cand.append(jj)

        cand = [j for j in cand if j != i]
        if not cand:
            cand = [int(j) for j in nn if int(j) != i]

        cand_arr = np.array(cand, dtype=np.int32)
        dtrue = D_obj[i, cand_arr].astype(np.float64)
        if r2_penalty_frac > 0.0:
            dtrue = dtrue + (r2_penalty_frac * r2_penalty_scale) * np.abs(float(r2[i]) - r2[cand_arr])
        order = np.argsort(dtrue, kind="stable")[:K_final]
        stn[i, :] = cand_arr[order]

    return stn


def build_obj_knn(model: Model, K_final: int) -> np.ndarray:
    N = model.N
    K = min(int(K_final), max(1, N - 1))
    stn = np.zeros((N + 1, K), dtype=np.int32)
    D = model.D_obj
    for i in range(1, N + 1):
        row = D[i, 1 : N + 1].copy()
        row[i - 1] = 10**9
        kth = max(0, min(K - 1, row.shape[0] - 1))
        nn = np.argpartition(row, kth)[:K]
        neigh = (nn + 1).astype(np.int32)
        neigh = neigh[np.argsort(D[i, neigh])]
        stn[i, :] = neigh[:K]
    return stn


def build_stn_hybrid(
    model: Model,
    *,
    dx: np.ndarray,
    dy: np.ndarray,
    r2: np.ndarray,
    phis: List[float],
    K_geo_pool: int,
    K_r_pool: int,
    K_y_pool: int,
    K_final: int,
) -> Dict[float, np.ndarray]:
    """
    Hybrid candidates for EXPLICIT:
    - per-phi probe windows (geo/r/y) to diversify candidate pools
    - objective-KNN to ensure good true-distance neighbors are always present
    Final per-node candidates are the top-K_final by D_obj within the union.
    """
    N = model.N
    K_final = min(int(K_final), max(1, N - 1))
    obj_knn = build_obj_knn(model, K_final)
    D = model.D_obj

    out: Dict[float, np.ndarray] = {}
    mark = np.zeros((N + 1,), dtype=np.int32)
    for phi in phis:
        probe = build_stn_probe(
            model,
            dx,
            dy,
            r2,
            float(phi),
            K_geo_pool=int(K_geo_pool),
            K_r_pool=int(K_r_pool),
            K_y_pool=int(K_y_pool),
            K_final=int(K_final),
        )
        stn = np.zeros((N + 1, K_final), dtype=np.int32)
        stamp = 1
        for i in range(1, N + 1):
            stamp += 1
            cand: List[int] = []
            for j in obj_knn[i]:
                jj = int(j)
                if jj <= 0 or jj == i:
                    continue
                if mark[jj] != stamp:
                    mark[jj] = stamp
                    cand.append(jj)
            for j in probe[i]:
                jj = int(j)
                if jj <= 0 or jj == i:
                    continue
                if mark[jj] != stamp:
                    mark[jj] = stamp
                    cand.append(jj)
            if not cand:
                continue
            cand_arr = np.asarray(cand, dtype=np.int32)
            order = np.argsort(D[i, cand_arr])[:K_final]
            sel = cand_arr[order]
            stn[i, : sel.shape[0]] = sel
        out[float(phi)] = stn
    return out


# -----------------------------
# Solution cost + validation
# -----------------------------


def sol_cost(model: Model, routes: List[List[int]]) -> int:
    D = model.D_obj
    tot = 0
    for rt in routes:
        if not rt:
            continue
        seq = [0] + rt + [0]
        for i in range(len(seq) - 1):
            tot += int(D[seq[i], seq[i + 1]])
    return int(tot)


def validate(model: Model, routes: List[List[int]]) -> bool:
    N = model.N
    Q = model.Q
    seen = np.zeros((N + 1,), dtype=np.uint8)
    for rt in routes:
        ld = 0
        for n in rt:
            if n <= 0 or n > N:
                return False
            if seen[n]:
                return False
            seen[n] = 1
            ld += int(model.q[n])
        if ld > Q:
            return False
    return int(seen[1:].sum()) == N


def cleanup_routes(routes: List[List[int]]) -> List[List[int]]:
    return [rt for rt in routes if rt]


def two_opt_full_route(
    model: Model,
    rt: List[int],
    *,
    max_passes: int = 50,
    max_len: int = 0,
) -> List[int]:
    """
    Full 2-opt local optimum for a single route (best-improvement per pass).
    - max_len=0 means no cap; otherwise routes longer than max_len are returned unchanged.
    """
    L = len(rt)
    if L < 4:
        return rt
    if int(max_len) > 0 and L > int(max_len):
        return rt
    D = model.D_obj
    passes = 0
    rt = rt[:]  # copy
    while passes < int(max_passes):
        passes += 1
        best_delta = 0
        best_delta_f = 0.0
        best_p = None
        best_q = None
        # p is the index of b in rt (edge a->b where a is depot if p==0 else rt[p-1])
        for p in range(0, L - 1):
            a = 0 if p == 0 else int(rt[p - 1])
            b = int(rt[p])
            ab = int(D[a, b])
            for q in range(p + 2, L + 1):  # q is edge index, corresponds to segment ending at rt[q-1]
                c = int(rt[q - 1])
                d = 0 if q == L else int(rt[q])
                delta = (int(D[a, c]) + int(D[b, d])) - (ab + int(D[c, d]))
                # If delta is 0 under rounded objective, it can still remove crossings (true length improves).
                # Break ties using float Euclidean improvement (strictly decreasing) to avoid cycles.
                if delta < best_delta or delta == best_delta:
                    ax = float(model.x[a])
                    ay = float(model.y[a])
                    bx = float(model.x[b])
                    by = float(model.y[b])
                    cx = float(model.x[c])
                    cy = float(model.y[c])
                    dx = float(model.x[d])
                    dy = float(model.y[d])
                    before = math.hypot(ax - bx, ay - by) + math.hypot(cx - dx, cy - dy)
                    after = math.hypot(ax - cx, ay - cy) + math.hypot(bx - dx, by - dy)
                    delta_f = after - before
                    if delta < best_delta or (delta == best_delta and delta_f < best_delta_f - 1e-12):
                        best_delta = int(delta)
                        best_delta_f = float(delta_f)
                        best_p = int(p)
                        best_q = int(q)
        if best_p is None or best_q is None:
            break
        if best_delta > 0:
            break
        if best_delta == 0 and not (best_delta_f < -1e-12):
            break
        rt[best_p:best_q] = reversed(rt[best_p:best_q])
        # unchanged length
    return rt


def two_opt_full_solution(
    model: Model,
    routes: List[List[int]],
    *,
    max_passes: int = 50,
    max_len: int = 0,
) -> List[List[int]]:
    out: List[List[int]] = []
    for rt in routes:
        if not rt:
            continue
        out.append(two_opt_full_route(model, rt, max_passes=int(max_passes), max_len=int(max_len)))
    return cleanup_routes(out)


def build_route_of(model: Model, routes: List[List[int]]) -> np.ndarray:
    ro = -np.ones((model.N + 1,), dtype=np.int32)
    for rid, rt in enumerate(routes):
        for n in rt:
            ro[int(n)] = rid
    return ro


# -----------------------------
# Construction
# -----------------------------


def construct_base(model: Model, stn: np.ndarray, seed: int = 21) -> List[List[int]]:
    random.seed(seed)
    N = model.N
    Q = model.Q
    D = model.D_obj

    un = np.ones((N + 1,), dtype=np.uint8)
    un[0] = 0

    routes: List[List[int]] = []

    while un[1:].any():
        rem = np.where(un == 1)[0]
        rem = rem[rem != 0]
        if rem.size == 0:
            break
        i = int(rem[np.argmax(D[0, rem])])
        rt = [i]
        un[i] = 0
        ld = int(model.q[i])
        cur = i

        while True:
            cand = []
            for j in stn[cur]:
                j = int(j)
                if j == 0:
                    continue
                if un[j] and ld + int(model.q[j]) <= Q:
                    cand.append(j)
                    if len(cand) >= 64:
                        break
            if not cand:
                break
            cand.sort(key=lambda j: int(D[cur, j]))
            nxt = int(cand[0])
            rt.append(nxt)
            un[nxt] = 0
            ld += int(model.q[nxt])
            cur = nxt

        routes.append(rt)

    return cleanup_routes(routes)


# -----------------------------
# Local improve (relocate + 2-opt)
# -----------------------------


def local_improve(
    model: Model, routes: List[List[int]], stn: np.ndarray, steps_reloc: int = 60, steps_2opt: int = 40
) -> List[List[int]]:
    D = model.D_obj
    Q = model.Q

    routes = [rt[:] for rt in routes]
    routes = cleanup_routes(routes)
    if not routes:
        return routes
    loads = [int(sum(int(model.q[n]) for n in rt)) for rt in routes]

    # Maintain route-of mapping without rebuilding globally every move.
    ro = build_route_of(model, routes)

    for _ in range(int(steps_reloc)):
        if not routes:
            break
        rid_from = random.randrange(len(routes))
        if not routes[rid_from]:
            continue
        idx = random.randrange(len(routes[rid_from]))
        node = routes[rid_from][idx]

        rt_from = routes[rid_from]
        a = 0 if idx == 0 else rt_from[idx - 1]
        b = 0 if idx == len(rt_from) - 1 else rt_from[idx + 1]
        drem = int(D[a, b]) - int(D[a, node]) - int(D[node, b])

        cand_routes = set()
        for k in stn[node][:128]:
            k = int(k)
            if k == 0:
                continue
            rid = int(ro[k])
            if rid >= 0:
                cand_routes.add(rid)
                if len(cand_routes) >= 60:
                    break
        if not cand_routes and loads:
            cand_routes = set(np.argsort(loads)[: min(25, len(loads))].tolist())

        best = None
        best_r = None
        best_pos = None

        for rid in cand_routes:
            if rid == rid_from:
                continue
            if loads[rid] + int(model.q[node]) > Q:
                continue
            rt = routes[rid]
            seq = [0] + rt + [0]
            cand_pos = {0, len(rt)}
            if rt:
                pos_map = {rt[i]: i for i in range(len(rt))}
                added = 0
                for k in stn[node][:24]:
                    k = int(k)
                    if k and ro[k] == rid and k in pos_map:
                        p = pos_map[k]
                        cand_pos.add(p)
                        cand_pos.add(p + 1)
                        added += 1
                        if added >= 8:
                            break

            for pos in cand_pos:
                aa = seq[pos]
                bb = seq[pos + 1]
                ins = int(D[aa, node]) + int(D[node, bb]) - int(D[aa, bb])
                if best is None or (drem + ins) < best:
                    best = drem + ins
                    best_r = rid
                    best_pos = pos

        if best_r is None or best is None or best >= 0:
            continue

        routes[rid_from].pop(idx)
        loads[rid_from] -= int(model.q[node])
        routes[best_r].insert(int(best_pos), int(node))
        loads[best_r] += int(model.q[node])
        ro[int(node)] = int(best_r)

        if not routes[rid_from]:
            # Keep placeholder to avoid re-indexing ro; cleaned up at the end.
            routes[rid_from] = []
            loads[rid_from] = 0

    for _ in range(int(steps_2opt)):
        if not routes:
            break
        rid = random.randrange(len(routes))
        rt = routes[rid]
        L = len(rt)
        if L < 4:
            continue
        seq = [0] + rt + [0]

        edges = [(int(D[seq[i], seq[i + 1]]), i) for i in range(len(seq) - 1)]
        edges.sort(reverse=True)
        i = edges[0][1]

        improved = False
        for j in range(i + 2, min(i + 18, len(seq) - 1)):
            a, b = seq[i], seq[i + 1]
            c, d = seq[j], seq[j + 1]
            delta = (int(D[a, c]) + int(D[b, d])) - (int(D[a, b]) + int(D[c, d]))
            if delta < 0:
                rt[i:j] = reversed(rt[i:j])
                improved = True
                break
        if improved:
            routes[rid] = rt

        # Inter-route 2-opt* (tail swap) attempt to escape local basins.
        if len(routes) >= 2 and random.random() < 0.25:
            u = int(random.choice(rt))
            rid_u = int(ro[u])
            if rid_u < 0:
                continue
            # pick v from KNN candidates
            v = None
            for cand in stn[u][:24]:
                cand = int(cand)
                if cand != 0 and ro[cand] >= 0 and int(ro[cand]) != rid_u:
                    v = cand
                    break
            if v is None:
                continue
            rid_v = int(ro[v])
            rt1 = routes[rid_u]
            rt2 = routes[rid_v]
            if not rt1 or not rt2:
                continue
            try:
                p1 = rt1.index(u)
                p2 = rt2.index(v)
            except ValueError:
                continue
            u_next = 0 if p1 == len(rt1) - 1 else rt1[p1 + 1]
            v_next = 0 if p2 == len(rt2) - 1 else rt2[p2 + 1]
            delta = (int(D[u, v_next]) + int(D[v, u_next])) - (int(D[u, u_next]) + int(D[v, v_next]))
            if delta >= 0:
                continue
            tail1 = rt1[p1 + 1 :]
            tail2 = rt2[p2 + 1 :]
            tail1_load = int(sum(int(model.q[n]) for n in tail1))
            tail2_load = int(sum(int(model.q[n]) for n in tail2))
            load1 = loads[rid_u]
            load2 = loads[rid_v]
            if load1 - tail1_load + tail2_load > Q or load2 - tail2_load + tail1_load > Q:
                continue
            rt1_new = rt1[: p1 + 1] + tail2
            rt2_new = rt2[: p2 + 1] + tail1
            routes[rid_u] = rt1_new
            routes[rid_v] = rt2_new
            loads[rid_u] = load1 - tail1_load + tail2_load
            loads[rid_v] = load2 - tail2_load + tail1_load
            # Update route-of for affected tails
            for n in tail2:
                ro[int(n)] = rid_u
            for n in tail1:
                ro[int(n)] = rid_v

    return cleanup_routes(routes)


# -----------------------------
# Multi-axis LNS (baseline)
# -----------------------------

OnBestFn = Callable[[int, int, float, float, List[List[int]]], None]


def multi_axis_lns(
    model: Model,
    base_routes: List[List[int]],
    stn_by_phi: Dict[float, np.ndarray],
    iters: int = 600,
    k_remove: int = 60,
    accept_p: float = 0.35,
    seed: int = 21,
    steps_reloc: int = 60,
    steps_2opt: int = 40,
    lookahead: int = 1,
    time_limit_s: float | None = None,
    on_best: OnBestFn | None = None,
    epn: EPNState | None = None,
    epn_cfg: EPNConfig | None = None,
    full_2opt_on_best: bool = False,
    full_2opt_max_len: int = 0,
    full_2opt_max_passes: int = 50,
) -> Tuple[int, List[List[int]]]:
    random.seed(seed)
    t0 = time.perf_counter()
    t_end = (t0 + float(time_limit_s)) if time_limit_s is not None else None

    best_routes = [rt[:] for rt in base_routes]
    best_cost = sol_cost(model, best_routes)
    cur_routes = [rt[:] for rt in base_routes]
    cur_cost = best_cost

    phis = list(stn_by_phi.keys())
    if on_best is not None and phis:
        on_best(int(best_cost), 0, float(phis[0]), 0.0, [rt[:] for rt in best_routes])

    for it in range(int(iters)):
        if t_end is not None and time.perf_counter() >= t_end:
            break
        phi = phis[it % len(phis)]
        stn = stn_by_phi[phi]

        best_cand_routes: List[List[int]] | None = None
        best_cand_cost: int | None = None
        la = max(1, int(lookahead))
        for _ in range(la):
            if t_end is not None and time.perf_counter() >= t_end:
                break
            routes = lns_one_step(
                model,
                cur_routes,
                stn,
                k_remove=int(k_remove),
                steps_reloc=int(steps_reloc),
                steps_2opt=int(steps_2opt),
                epn=epn,
                epn_cfg=epn_cfg,
            )
            if routes is None:
                continue
            c = sol_cost(model, routes)
            if best_cand_cost is None or c < best_cand_cost:
                best_cand_cost = int(c)
                best_cand_routes = routes

        if best_cand_routes is None or best_cand_cost is None:
            continue
        c = int(best_cand_cost)
        routes = best_cand_routes

        if c < best_cost:
            best_routes_candidate = routes
            if bool(full_2opt_on_best):
                best_routes_candidate = two_opt_full_solution(
                    model,
                    best_routes_candidate,
                    max_passes=int(full_2opt_max_passes),
                    max_len=int(full_2opt_max_len),
                )
                c2 = sol_cost(model, best_routes_candidate)
                if c2 <= c:
                    c = int(c2)
                    routes = best_routes_candidate
            best_cost = int(c)
            best_routes = [rt[:] for rt in routes]
            if on_best is not None:
                on_best(int(best_cost), int(it), float(phi), float(time.perf_counter() - t0), [rt[:] for rt in best_routes])

        if c <= cur_cost or random.random() < float(accept_p):
            cur_cost = c
            cur_routes = [rt[:] for rt in routes]

    return best_cost, best_routes


def lns_one_step(
    model: Model,
    cur_routes: List[List[int]],
    stn: np.ndarray,
    *,
    k_remove: int,
    steps_reloc: int,
    steps_2opt: int,
    epn: EPNState | None = None,
    epn_cfg: EPNConfig | None = None,
) -> List[List[int]] | None:
    """
    Single destroy/repair + local improve step.
    Returns improved routes, or None if the move produced an invalid solution.
    """
    routes = [rt[:] for rt in cur_routes]
    routes = cleanup_routes(routes)
    if not routes:
        return None
    loads = [int(sum(int(model.q[n]) for n in rt)) for rt in routes]

    # Destroy: endpoints of worst edges + a few random extras.
    edges: List[Tuple[int, int, int]] = []
    for rt in routes:
        seq = [0] + rt + [0]
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            edges.append((int(model.D_obj[a, b]), int(a), int(b)))
    edges.sort(reverse=True)
    removed_set = set()
    for _w, a, b in edges:
        if a != 0:
            removed_set.add(a)
        if b != 0:
            removed_set.add(b)
        if len(removed_set) >= int(k_remove):
            break
    # add extras for diversification
    extras_target = max(0, min(model.N, max(5, int(k_remove) // 5)))
    pool = [n for rt in routes for n in rt if n not in removed_set]
    random.shuffle(pool)
    for n in pool[:extras_target]:
        removed_set.add(int(n))
    removed = list(removed_set)
    rem_set = set(removed)

    routes = [[n for n in rt if n not in rem_set] for rt in routes]
    routes = cleanup_routes(routes)
    loads = [int(sum(int(model.q[n]) for n in rt)) for rt in routes]
    ro = build_route_of(model, routes)

    for node in removed:
        cand_routes = set()
        for k in stn[node][:128]:
            k = int(k)
            if k == 0:
                continue
            rid = int(ro[k])
            if rid >= 0:
                cand_routes.add(rid)
                if len(cand_routes) >= 80:
                    break
        if not cand_routes and loads:
            cand_routes = set(np.argsort(loads)[:25].tolist())

        best = None
        best_r = None
        best_pos = None

        for rid in cand_routes:
            if loads[rid] + int(model.q[node]) > model.Q:
                continue
            rt = routes[rid]
            seq = [0] + rt + [0]
            cand_pos = {0, len(rt)}
            if rt:
                pos_map = {rt[i]: i for i in range(len(rt))}
                added = 0
                for k in stn[node][:24]:
                    k = int(k)
                    if k and ro[k] == rid and k in pos_map:
                        p = pos_map[k]
                        cand_pos.add(p)
                        cand_pos.add(p + 1)
                        added += 1
                        if added >= 8:
                            break

            for pos in cand_pos:
                a = seq[pos]
                b = seq[pos + 1]
                ins = int(model.D_obj[a, node]) + int(model.D_obj[node, b]) - int(model.D_obj[a, b])
                if best is None or ins < best:
                    best = ins
                    best_r = rid
                    best_pos = pos

        if best_r is None:
            routes.append([node])
            loads.append(int(model.q[node]))
            ro[int(node)] = len(routes) - 1
        else:
            routes[best_r].insert(int(best_pos), node)
            loads[best_r] += int(model.q[node])
            ro[int(node)] = int(best_r)

    if not validate(model, routes):
        return None

    routes = local_improve(model, routes, stn, steps_reloc=steps_reloc, steps_2opt=steps_2opt)
    if not validate(model, routes):
        return None
    if epn is not None and epn_cfg is not None and epn_cfg.enabled:
        routes = epn_topup_long_edges(model, routes, epn=epn, cfg=epn_cfg)
        if not validate(model, routes):
            return None
    return routes


# -----------------------------
# Multi-axis LNS prune (lightweight axis pruning)
# -----------------------------


def multi_axis_lns_prune(
    model: Model,
    base_routes: List[List[int]],
    stn_by_phi: Dict[float, np.ndarray],
    iters: int = 1200,
    k_remove: int = 80,
    accept_p: float = 0.30,
    seed: int = 21,
    steps_reloc: int = 80,
    steps_2opt: int = 60,
    plateau_window: int = 200,
    min_axes: int = 3,
    prune_batch: int = 1,
    eps: float = 0.12,
    lookahead: int = 1,
    time_limit_s: float | None = None,
    on_best: OnBestFn | None = None,
    epn: EPNState | None = None,
    epn_cfg: EPNConfig | None = None,
    full_2opt_on_best: bool = False,
    full_2opt_max_len: int = 0,
    full_2opt_max_passes: int = 50,
) -> Tuple[int, List[List[int]], List[float], Dict]:
    random.seed(seed)
    t0 = time.perf_counter()
    t_end = (t0 + float(time_limit_s)) if time_limit_s is not None else None

    phis = list(stn_by_phi.keys())
    active = phis[:]
    last_improve_iter = 0

    best_routes = [rt[:] for rt in base_routes]
    best_cost = sol_cost(model, best_routes)

    cur_routes = [rt[:] for rt in base_routes]
    cur_cost = best_cost

    stats = {"prunes": 0}

    # Epsilon-greedy axis choice with pruning on plateau.
    picked = {phi: 0 for phi in phis}
    accepted = {phi: 0 for phi in phis}
    best_hits = {phi: 0 for phi in phis}
    sum_improve = {phi: 0.0 for phi in phis}

    def score(phi: float) -> float:
        return (sum_improve[phi] / max(1, picked[phi])) + 10.0 * best_hits[phi] + 0.1 * accepted[phi]

    no_best_improve = 0
    eps_local = float(eps)
    if on_best is not None and active:
        on_best(int(best_cost), 0, float(active[0]), 0.0, [rt[:] for rt in best_routes])

    for it in range(int(iters)):
        if t_end is not None and time.perf_counter() >= t_end:
            break
        if not active:
            break
        if random.random() < eps_local:
            phi = random.choice(active)
        else:
            phi = max(active, key=score)

        picked[phi] += 1
        stn = stn_by_phi[phi]

        best_cand_routes: List[List[int]] | None = None
        best_cand_cost: int | None = None
        la = max(1, int(lookahead))
        for _ in range(la):
            if t_end is not None and time.perf_counter() >= t_end:
                break
            routes = lns_one_step(
                model,
                cur_routes,
                stn,
                k_remove=int(k_remove),
                steps_reloc=int(steps_reloc),
                steps_2opt=int(steps_2opt),
                epn=epn,
                epn_cfg=epn_cfg,
            )
            if routes is None:
                continue
            c = sol_cost(model, routes)
            if best_cand_cost is None or c < best_cand_cost:
                best_cand_cost = int(c)
                best_cand_routes = routes

        if best_cand_routes is None or best_cand_cost is None:
            continue
        c = int(best_cand_cost)
        routes = best_cand_routes

        if c < best_cost:
            best_routes_candidate = routes
            if bool(full_2opt_on_best):
                best_routes_candidate = two_opt_full_solution(
                    model,
                    best_routes_candidate,
                    max_passes=int(full_2opt_max_passes),
                    max_len=int(full_2opt_max_len),
                )
                c2 = sol_cost(model, best_routes_candidate)
                if c2 <= c:
                    c = int(c2)
                    routes = best_routes_candidate
            best_cost = int(c)
            best_routes = [rt[:] for rt in routes]
            best_hits[phi] += 1
            no_best_improve = 0
            if on_best is not None:
                on_best(int(best_cost), int(it), float(phi), float(time.perf_counter() - t0), [rt[:] for rt in best_routes])
        else:
            no_best_improve += 1

        if c <= cur_cost or random.random() < float(accept_p):
            if c < cur_cost:
                sum_improve[phi] += float(cur_cost - c)
            accepted[phi] += 1
            cur_cost = c
            cur_routes = [rt[:] for rt in routes]

        if no_best_improve >= int(plateau_window) and len(active) > int(min_axes):
            active_sorted = sorted(active, key=score)
            drop = min(int(prune_batch), len(active) - int(min_axes))
            for _ in range(drop):
                active.pop(active.index(active_sorted.pop(0)))
                stats["prunes"] += 1
            no_best_improve = 0
            eps_local = max(0.05, eps_local * 0.8)

    return best_cost, best_routes, active, stats


# -----------------------------
# SOL I/O
# -----------------------------


def parse_sol(path: str) -> Tuple[List[List[int]], Optional[int]]:
    routes = []
    cost = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.lower().startswith("route"):
                _, rhs = ln.split(":", 1)
                routes.append([int(x) for x in rhs.split()])
            elif ln.lower().startswith("cost"):
                m = re.search(r"(-?\d+(\.\d+)?)", ln)
                if m:
                    cost = int(round(float(m.group(1))))
    return routes, cost


def bks_cost_from_sol(model: Model, prob: Problem, sol_routes: List[List[int]]) -> int:
    if not sol_routes:
        return 0
    flat = [n for rt in sol_routes for n in rt]
    mn, mx = min(flat), max(flat)
    if mn >= 1 and mx <= model.N:
        mapped = sol_routes
    else:
        mapped = [[int(model.orig_to_sol[n]) for n in rt] for rt in sol_routes]
    return sol_cost(model, mapped)


def write_sol_file(path: str, routes: List[List[int]], cost: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for k, rt in enumerate(routes, 1):
            f.write(f"Route #{k}: " + " ".join(map(str, rt)) + "\n")
        f.write(f"Cost {int(cost)}\n")


def validate_full(model: Model, routes: List[List[int]]) -> Tuple[bool, str]:
    N = model.N
    Q = model.Q
    seen = np.zeros((N + 1,), dtype=np.uint8)
    for rt in routes:
        ld = 0
        for n in rt:
            if n <= 0 or n > N:
                return False, f"node_out_of_range:{n}"
            if seen[n]:
                return False, f"duplicate:{n}"
            seen[n] = 1
            ld += int(model.q[n])
        if ld > Q:
            return False, f"capacity_violation:{ld}>{Q}"
    if int(seen[1:].sum()) != N:
        return False, f"missing:{N-int(seen[1:].sum())}"
    _ = sol_cost(model, routes)
    return True, "ok"


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("vrp", help="instance .vrp")
    ap.add_argument("--bks", default=None, help="optional .sol for BKS compare")
    ap.add_argument("--out_sol", default=None, help="write best solution to this .sol file")
    ap.add_argument("--phis", default="45,-45,90,0", help="comma-separated phi degrees")

    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--k_remove", type=int, default=100)
    ap.add_argument("--accept_p", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=21)

    ap.add_argument("--K_final", type=int, default=128)
    ap.add_argument("--K_geo_pool", type=int, default=256)
    ap.add_argument("--K_r_pool", type=int, default=128)
    ap.add_argument("--K_y_pool", type=int, default=256)

    ap.add_argument("--steps_reloc", type=int, default=120)
    ap.add_argument("--steps_2opt", type=int, default=80)

    ap.add_argument("--use_prune", action="store_true")
    ap.add_argument(
        "--axis_schedule",
        action="store_true",
        help="Start with best single-axis, then multi-axis, then prune to 1 axis.",
    )
    ap.add_argument("--sweep_iters", type=int, default=0, help="Optional per-phi warmup iters (0=auto)")
    ap.add_argument("--final_single_iters", type=int, default=0, help="Optional final single-axis iters (0=auto)")
    ap.add_argument("--plateau_window", type=int, default=200)
    ap.add_argument("--min_axes", type=int, default=3)
    ap.add_argument("--prune_batch", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.12)

    args = ap.parse_args()
    phis = [float(x) for x in args.phis.split(",") if x.strip()]

    prob = parse_vrp(args.vrp)
    model = build_model(prob)
    dx, dy, r2 = compute_r_features(model)

    N = model.N
    N1 = max(1, N - 1)
    K_geo_pool = min(int(args.K_geo_pool), N1)
    K_y_pool = min(int(args.K_y_pool), N1)
    K_r_pool = min(int(args.K_r_pool), N1)
    K_final = min(int(args.K_final), K_geo_pool, K_y_pool, N1)

    stn_by_phi: Dict[float, np.ndarray] = {}
    t0 = time.time()
    if prob.ewt != "EUC_2D":
        stn_by_phi = build_stn_hybrid(
            model,
            dx=dx,
            dy=dy,
            r2=r2,
            phis=phis,
            K_geo_pool=K_geo_pool,
            K_r_pool=K_r_pool,
            K_y_pool=K_y_pool,
            K_final=K_final,
        )
    else:
        for phi in phis:
            stn_by_phi[phi] = build_stn_probe(
                model,
                dx,
                dy,
                r2,
                phi,
                K_geo_pool=K_geo_pool,
                K_r_pool=K_r_pool,
                K_y_pool=K_y_pool,
                K_final=K_final,
            )
    t_stn = time.time() - t0

    base_phi = phis[0]
    base_routes = construct_base(model, stn_by_phi[base_phi], seed=args.seed)
    base_c = sol_cost(model, base_routes)

    t1 = time.time()
    axes_remaining = len(phis)

    if args.axis_schedule:
        total_iters = int(args.iters)
        sweep_iters = int(args.sweep_iters) if int(args.sweep_iters) > 0 else max(50, min(200, total_iters // 50))
        final_single_iters = (
            int(args.final_single_iters) if int(args.final_single_iters) > 0 else max(50, min(200, total_iters // 50))
        )
        grow_iters = int(total_iters * 0.65)
        prune_iters = max(1, total_iters - grow_iters)

        # Stage 1: evaluate each phi independently to find strongest start.
        best_routes = base_routes
        best_c = base_c
        best_start_phi = base_phi
        for phi in phis:
            rt0 = construct_base(model, stn_by_phi[phi], seed=args.seed)
            c1, r1 = multi_axis_lns(
                model,
                rt0,
                {phi: stn_by_phi[phi]},
                iters=sweep_iters,
                k_remove=args.k_remove,
                accept_p=args.accept_p,
                seed=args.seed,
                steps_reloc=args.steps_reloc,
                steps_2opt=args.steps_2opt,
            )
            if c1 < best_c:
                best_c = c1
                best_routes = r1
                best_start_phi = phi

        # Stage 2: multi-axis improve (all axes).
        c2, r2 = multi_axis_lns(
            model,
            best_routes,
            stn_by_phi,
            iters=grow_iters,
            k_remove=args.k_remove,
            accept_p=args.accept_p,
            seed=args.seed,
            steps_reloc=args.steps_reloc,
            steps_2opt=args.steps_2opt,
        )
        best_c, best_routes = c2, r2

        # Stage 3: prune down toward 1 axis.
        c3, r3, remaining, _stats = multi_axis_lns_prune(
            model,
            best_routes,
            stn_by_phi,
            iters=prune_iters,
            k_remove=args.k_remove,
            accept_p=args.accept_p,
            seed=args.seed,
            steps_reloc=args.steps_reloc,
            steps_2opt=args.steps_2opt,
            plateau_window=args.plateau_window,
            min_axes=1,
            prune_batch=args.prune_batch,
            eps=args.eps,
        )
        best_c, best_routes = c3, r3

        # Stage 4: ensure axes_remaining=1 by final single-axis refinement among remaining.
        phi_best = remaining[0] if remaining else best_start_phi
        for phi in (remaining if remaining else [best_start_phi]):
            c4, r4 = multi_axis_lns(
                model,
                best_routes,
                {phi: stn_by_phi[phi]},
                iters=final_single_iters,
                k_remove=args.k_remove,
                accept_p=args.accept_p,
                seed=args.seed,
                steps_reloc=args.steps_reloc,
                steps_2opt=args.steps_2opt,
            )
            if c4 < best_c:
                best_c, best_routes, phi_best = c4, r4, phi
        axes_remaining = 1
    else:
        if args.use_prune:
            best_c, best_routes, remaining, stats = multi_axis_lns_prune(
                model,
                base_routes,
                stn_by_phi,
                iters=args.iters,
                k_remove=args.k_remove,
                accept_p=args.accept_p,
                seed=args.seed,
                steps_reloc=args.steps_reloc,
                steps_2opt=args.steps_2opt,
                plateau_window=args.plateau_window,
                min_axes=args.min_axes,
                prune_batch=args.prune_batch,
                eps=args.eps,
            )
            axes_remaining = len(remaining)
            _ = stats
        else:
            best_c, best_routes = multi_axis_lns(
                model,
                base_routes,
                stn_by_phi,
                iters=args.iters,
                k_remove=args.k_remove,
                accept_p=args.accept_p,
                seed=args.seed,
                steps_reloc=args.steps_reloc,
                steps_2opt=args.steps_2opt,
            )
            axes_remaining = len(phis)

    t_solve = time.time() - t1

    ok, why = validate_full(model, best_routes)
    if not ok:
        raise RuntimeError(f"INVALID FINAL SOLUTION: {why}")

    bks_c = None
    if args.bks and os.path.exists(args.bks):
        sol_routes, _ = parse_sol(args.bks)
        bks_c = bks_cost_from_sol(model, prob, sol_routes)

    print(
        f"{prob.name}: base={base_c} best={best_c} routes={len(best_routes)} "
        f"axes={len(phis)} axes_remaining={axes_remaining}"
    )
    print(f"timing: stn={t_stn:.3f}s solve={t_solve:.3f}s total={(t_stn + t_solve):.3f}s")
    if bks_c is not None and bks_c > 0:
        gap = (best_c - bks_c) / bks_c * 100.0
        print(f"BKS(calc)={bks_c} gap={gap:.3f}%")
    if args.out_sol:
        write_sol_file(args.out_sol, best_routes, best_c)
        print("WROTE:", args.out_sol)


if __name__ == "__main__":
    main()
