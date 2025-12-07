from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Set

from .core import CoreState, compute_total_cost


@dataclass
class InitConfig:
    """Configuration for the initial constructor."""

    # future knobs can live here; currently a placeholder to mirror the HLD
    pass


def _prox(state: CoreState, i: int, j: int) -> float:
    """
    Proximity score for seed/growth decisions.
    For explicit instances we rely on the actual distance;
    for EUC_2D we can use squared geometry as a monotone proxy.
    """
    if state.distance_type == 2:
        return state.dist(i, j)
    dx = state.x[i] - state.x[j]
    dy = state.y[i] - state.y[j]
    return dx * dx + dy * dy


def build_initial_solution(state: CoreState, cfg: Optional[InitConfig] = None) -> None:
    """
    Farthest-first STN-guided greedy constructor.
    - Seeds each route with the farthest unassigned customer (by depot distance).
    - Grows using STN neighbors that fit capacity, preferring nearest/proximal.
    - Falls back to global nearest feasible if STN is exhausted.
    """
    if cfg is None:
        cfg = InitConfig()

    # Reset any existing solution structures
    state.route_nodes = []
    state.route_load = []
    state.route_active = []
    state.route_mask = []
    size = state.N + 1
    state.node_route = [-1] * size
    state.node_pos = [-1] * size

    unassigned: Set[int] = set(range(1, state.N + 1))

    while unassigned:
        # Seed: farthest from depot (dist(0, j)); tie-break by lowest id for determinism.
        seed = max(unassigned, key=lambda j: (state.dist(0, j), -j))
        route = [0, seed]
        load = state.demand[seed]
        state.node_route[seed] = len(state.route_nodes)
        state.node_pos[seed] = 1
        unassigned.remove(seed)
        u = seed

        while True:
            # STN candidates that are unassigned and fit capacity
            stn_neighbors = state.stn[u] if state.stn else []
            cands = [
                v
                for v in stn_neighbors
                if v in unassigned and load + state.demand[v] <= state.Q
            ]

            chosen = None
            if cands:
                chosen = min(cands, key=lambda v: (_prox(state, u, v), v))
            else:
                # Global fallback
                global_cands = [
                    v for v in unassigned if load + state.demand[v] <= state.Q
                ]
                if global_cands:
                    chosen = min(global_cands, key=lambda v: (state.dist(u, v), v))

            if chosen is None:
                break

            route.append(chosen)
            load += state.demand[chosen]
            state.node_route[chosen] = len(state.route_nodes)
            state.node_pos[chosen] = len(route) - 1
            unassigned.remove(chosen)
            u = chosen

        # Close route with depot
        route.append(0)
        state.route_nodes.append(route)
        state.route_load.append(load)
        state.route_active.append(True)
        state.route_mask.append(0)

    # Cost bookkeeping
    state.current_cost = compute_total_cost(state)
    state.C0 = state.current_cost
    state.cum_delta = 0.0
