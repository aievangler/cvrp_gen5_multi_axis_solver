from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CoreState:
    """
    Core solver state for Gen3_v3.

    Internal indices: depot = 0, customers = 1..N.
    distance_type:
        0 = EUC_2D unrounded
        1 = EUC_2D rounded (int(d + 0.5))
        2 = explicit matrix (dist_matrix required)
    """

    N: int
    Q: int
    distance_type: int = 1

    # Geometry (always present)
    x: List[float] = field(default_factory=list)
    y: List[float] = field(default_factory=list)

    # Explicit distance matrix when distance_type == 2 (row-major, size (N+1)*(N+1))
    dist_matrix: Optional[List[int]] = None

    # Demand (demand[0] = 0)
    demand: List[int] = field(default_factory=list)

    # Solution state
    route_nodes: List[List[int]] = field(default_factory=list)  # each route: [0, j1, ..., jk, 0]
    route_load: List[int] = field(default_factory=list)
    route_mask: List[int] = field(default_factory=list)
    route_active: List[bool] = field(default_factory=list)
    node_route: List[Optional[int]] = field(default_factory=list)
    node_pos: List[Optional[int]] = field(default_factory=list)
    stn: List[List[int]] = field(default_factory=list)

    # Cost bookkeeping
    C0: float = 0.0
    cum_delta: float = 0.0
    current_cost: float = 0.0

    def dist(self, i: int, j: int) -> float:
        """Distance between i and j according to distance_type."""
        if i == j:
            return 0.0
        if self.distance_type == 2 and self.dist_matrix is not None:
            size = self.N + 1
            return self.dist_matrix[i * size + j]
        dx = self.x[i] - self.x[j]
        dy = self.y[i] - self.y[j]
        d = math.hypot(dx, dy)
        if self.distance_type == 0:
            return d
        return int(d + 0.5)


def compute_total_cost(state: CoreState) -> float:
    """Compute total cost across all active routes using state.dist."""
    cost = 0.0
    for rid, route in enumerate(state.route_nodes):
        if state.route_active and rid < len(state.route_active) and not state.route_active[rid]:
            continue
        for pos in range(len(route) - 1):
            a = route[pos]
            b = route[pos + 1]
            cost += state.dist(a, b)
    return cost


def snapshot_state(state: CoreState) -> CoreState:
    """Deep copy mutable parts of the solution; geometry/demand reused."""
    snap = CoreState(
        N=state.N,
        Q=state.Q,
        distance_type=state.distance_type,
        x=state.x,
        y=state.y,
        dist_matrix=copy.deepcopy(state.dist_matrix) if state.dist_matrix is not None else None,
        demand=state.demand,
        route_nodes=[list(r) for r in state.route_nodes],
        route_load=list(state.route_load),
        route_mask=list(state.route_mask),
        route_active=list(state.route_active),
        node_route=list(state.node_route),
        node_pos=list(state.node_pos),
        stn=[list(nbrs) for nbrs in state.stn],
        C0=state.C0,
        cum_delta=state.cum_delta,
        current_cost=state.current_cost,
    )
    return snap


def restore_snapshot(target: CoreState, snap: CoreState) -> None:
    """Copy mutable solution state from snapshot into target."""
    target.route_nodes = [list(r) for r in snap.route_nodes]
    target.route_load = list(snap.route_load)
    target.route_mask = list(snap.route_mask)
    target.route_active = list(snap.route_active)
    target.node_route = list(snap.node_route)
    target.node_pos = list(snap.node_pos)
    target.stn = [list(nbrs) for nbrs in snap.stn]
    target.distance_type = snap.distance_type
    target.dist_matrix = copy.deepcopy(snap.dist_matrix) if snap.dist_matrix is not None else None
    target.C0 = snap.C0
    target.cum_delta = snap.cum_delta
    target.current_cost = snap.current_cost
