from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .core import CoreState


@dataclass
class STNConfig:
    stn_k1: int = 100
    stn_k2: int = 20
    stn_k3: int = 5
    stn_min: int = 10


def _prox(state: CoreState, i: int, j: int) -> float:
    """
    Proximity score for STN ordering.

    For explicit matrices (distance_type == 2) we must rely on the real distance.
    For EUC_2D we can use squared geometry as a monotone proxy.
    """
    if state.distance_type == 2:
        return state.dist(i, j)
    dx = state.x[i] - state.x[j]
    dy = state.y[i] - state.y[j]
    return dx * dx + dy * dy


def build_stn(state: CoreState, cfg: STNConfig | None = None) -> None:
    """
    Build a symmetric STN per node using KNN layers with min-size enforcement.
    Depot (index 0) is excluded.
    """
    if cfg is None:
        cfg = STNConfig()
    size = state.N + 1  # include depot
    knn1: List[List[int]] = [[] for _ in range(size)]
    stn_raw: List[set[int]] = [set() for _ in range(size)]

    # KNN1: nearest stn_k1 customers per node (exclude depot and self).
    for i in range(1, size):
        dists = []
        for j in range(1, size):
            if i == j:
                continue
            dists.append((_prox(state, i, j), j))
        dists.sort(key=lambda t: (t[0], t[1]))
        knn1[i] = [j for _, j in dists[: cfg.stn_k1]]

    # KNN2/KNN3 expansion per node.
    for i in range(1, size):
        stn_raw[i].update(knn1[i])
        if cfg.stn_k2 > 0:
            for u in knn1[i]:
                stn_raw[i].update(knn1[u][: cfg.stn_k2])
        if cfg.stn_k3 > 0:
            for u in list(stn_raw[i]):
                stn_raw[i].update(knn1[u][: cfg.stn_k3])

    # Symmetrize once.
    for i in range(1, size):
        for j in list(stn_raw[i]):
            stn_raw[j].add(i)

    # Enforce minimum size.
    for i in range(1, size):
        if len(stn_raw[i]) >= cfg.stn_min:
            continue
        extras = [j for j in range(1, size) if j != i and j not in stn_raw[i]]
        extras.sort(key=lambda j: (state.dist(i, j), j))
        for j in extras:
            if len(stn_raw[i]) >= cfg.stn_min:
                break
            stn_raw[i].add(j)

    # Final symmetrization.
    for i in range(1, size):
        for j in list(stn_raw[i]):
            stn_raw[j].add(i)

    # Finalize sorted lists; depot index 0 stays empty.
    stn_final: List[List[int]] = [[] for _ in range(size)]
    for i in range(1, size):
        stn_final[i] = sorted(stn_raw[i])
    state.stn = stn_final
