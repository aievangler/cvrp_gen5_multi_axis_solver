from __future__ import annotations

from typing import Optional

from .core import CoreState, compute_total_cost


def two_opt_route(state: CoreState, rid: int, max_iters: Optional[int] = None) -> bool:
    """
    Perform intra-route 2-opt on a single route.
    Returns True if the route was improved.
    """
    route = state.route_nodes[rid]
    n = len(route)
    if n <= 4:
        return False
    improved = False
    iters = 0
    while True:
        changed = False
        for i in range(0, n - 3):
            a, b = route[i], route[i + 1]
            for j in range(i + 2, n - 1):
                c, d = route[j], route[j + 1]
                # Skip adjacent edges
                if a == c or a == d or b == c or b == d:
                    continue
                old = state.dist(a, b) + state.dist(c, d)
                new = state.dist(a, c) + state.dist(b, d)
                if new + 1e-9 < old:
                    # reverse segment between b and c (inclusive of b..c)
                    route[i + 1 : j + 1] = reversed(route[i + 1 : j + 1])
                    changed = True
                    improved = True
                    break
            if changed:
                break
        iters += 1
        if not changed:
            break
        if max_iters is not None and iters >= max_iters:
            break
    if improved:
        # update node_route/node_pos for this route
        for pos, node in enumerate(route):
            if node == 0:
                continue
            state.node_route[node] = rid
            state.node_pos[node] = pos
        state.route_nodes[rid] = route
    return improved


def two_opt_cleanup(state: CoreState, max_iters_per_route: Optional[int] = None) -> bool:
    """
    Run 2-opt on all active routes. Recomputes cost if any improvement found.
    """
    improved_any = False
    for rid, active in enumerate(state.route_active):
        if not active:
            continue
        if two_opt_route(state, rid, max_iters=max_iters_per_route):
            improved_any = True
    if improved_any:
        state.current_cost = compute_total_cost(state)
        state.C0 = state.current_cost
        state.cum_delta = 0.0
    return improved_any
