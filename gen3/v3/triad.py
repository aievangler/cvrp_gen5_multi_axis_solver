from __future__ import annotations

from typing import List, Optional, Set, Tuple

from .core import CoreState


def get_current_triad(state: CoreState, j: int) -> Optional[Tuple[int, int, int, int, int]]:
    """Return (route_id, pos, p, j, s) for node j, or None if not placed or at depot."""
    rid = state.node_route[j]
    if rid is None or rid < 0:
        return None
    route = state.route_nodes[rid]
    pos = state.node_pos[j]
    if pos is None or pos <= 0 or pos >= len(route) - 1:
        return None
    p = route[pos - 1]
    s = route[pos + 1]
    return rid, pos, p, j, s


def delta_remove(state: CoreState, p: int, j: int, s: int) -> float:
    return state.dist(p, j) + state.dist(j, s) - state.dist(p, s)


def delta_insert(state: CoreState, i: int, j: int, k: int) -> float:
    return state.dist(i, j) + state.dist(j, k) - state.dist(i, k)


def build_triad_candidates(state: CoreState, j: int, max_edges: int = 16) -> List[Tuple[int, int]]:
    """
    Build candidate edges (i,k) for inserting j using STN neighbors' local edges.
    Returns a small list of unique edges.
    """
    seen: Set[Tuple[int, int]] = set()
    edges: List[Tuple[int, int]] = []
    neighbors = state.stn[j] if state.stn else []
    for t in neighbors:
        rid_t = state.node_route[t]
        if rid_t is None or rid_t < 0:
            continue
        route_t = state.route_nodes[rid_t]
        pos_t = state.node_pos[t]
        if pos_t is None:
            continue
        if pos_t > 0:
            a = route_t[pos_t - 1]
            b = route_t[pos_t]
            if a != j and b != j and (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
        if pos_t < len(route_t) - 1:
            a = route_t[pos_t]
            b = route_t[pos_t + 1]
            if a != j and b != j and (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
        if len(edges) >= max_edges:
            break
    return edges[:max_edges]


def apply_triad_move(state: CoreState, j: int, src_route: int, src_pos: int, tgt_route: int, tgt_pos: int, delta: float) -> None:
    """
    Apply relocation of node j from (src_route, src_pos) to tgt_route between tgt_pos and tgt_pos+1.
    Assumes feasibility and capacity checks done by caller.
    """
    # Remove from source route
    src_nodes = state.route_nodes[src_route]
    removed = src_nodes.pop(src_pos)
    assert removed == j
    state.route_load[src_route] -= state.demand[j]

    # Insert into target route
    tgt_nodes = state.route_nodes[tgt_route]
    tgt_nodes.insert(tgt_pos + 1, j)
    state.route_load[tgt_route] += state.demand[j]

    # Update node_route/node_pos for affected routes
    for idx, node in enumerate(state.route_nodes[src_route]):
        if node == 0:
            continue
        state.node_route[node] = src_route
        state.node_pos[node] = idx
    for idx, node in enumerate(state.route_nodes[tgt_route]):
        if node == 0:
            continue
        state.node_route[node] = tgt_route
        state.node_pos[node] = idx

    # Mark empty source route inactive if needed
    if len(state.route_nodes[src_route]) <= 2:
        state.route_active[src_route] = False

    # Ensure routes still start/end at depot
    for rid in (src_route, tgt_route):
        if rid < len(state.route_nodes) and state.route_active[rid]:
            rt = state.route_nodes[rid]
            if rt and rt[0] != 0:
                rt.insert(0, 0)
            if not rt or rt[-1] != 0:
                rt.append(0)
            state.route_nodes[rid] = rt

    state.current_cost += delta
