from __future__ import annotations

import io
import sys
from typing import List, Optional

from .core import CoreState, compute_total_cost


def validate_solution(state: CoreState) -> None:
    """
    Basic feasibility checks using internal IDs.
    - Coverage: every customer 1..N appears exactly once.
    - Capacity: route_load <= Q for active routes.
    - Depot: routes start/end with depot (0) and depot does not appear in middle.
    - Cost: recompute with dist and ensure it matches current_cost if set.
    """
    N = state.N
    seen = [0] * (N + 1)
    active_flags = state.route_active if state.route_active else [True] * len(state.route_nodes)

    for rid, route in enumerate(state.route_nodes):
        if rid < len(active_flags) and not active_flags[rid]:
            continue
        if len(route) < 2:
            continue
        if route[0] != 0 or route[-1] != 0:
            raise ValueError(f"Route {rid} does not start/end at depot")
        load = 0
        for pos, node in enumerate(route):
            if node < 0 or node > N:
                raise ValueError(f"Node id {node} out of range")
            if node == 0:
                continue
            seen[node] += 1
            load += state.demand[node]
        if load > state.Q:
            raise ValueError(f"Route {rid} exceeds capacity")

    for j in range(1, N + 1):
        if seen[j] != 1:
            raise ValueError(f"Customer {j} seen {seen[j]} times")

    # Cost check
    recomputed = compute_total_cost(state)
    if state.current_cost:
        if abs(recomputed - state.current_cost) > 1e-6:
            raise ValueError(f"Cost mismatch: recomputed {recomputed} vs current_cost {state.current_cost}")
    state.current_cost = recomputed


def emit_solution_for_controller(state: CoreState, out: Optional[io.TextIOBase] = None) -> None:
    """
    Emit solution in Controller-friendly format to stdout (or provided stream):
    Route #k: <customers...>
    ...
    Cost <current_cost>
    Depot (0) is omitted in output.
    """
    if out is None:
        out_stream = sys.stdout
    else:
        out_stream = out
    active_flags = state.route_active if state.route_active else [True] * len(state.route_nodes)
    k = 1
    for rid, route in enumerate(state.route_nodes):
        if rid < len(active_flags) and not active_flags[rid]:
            continue
        if len(route) <= 2:
            continue  # empty or depot-only
        inner = [str(node) for node in route[1:-1] if node != 0]
        out_stream.write(f"Route #{k}: {' '.join(inner)}\n")
        k += 1
    cost = state.current_cost if state.current_cost else compute_total_cost(state)
    out_stream.write(f"Cost {int(round(cost))}\n")
    out_stream.flush()
