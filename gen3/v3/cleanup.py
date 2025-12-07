from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .core import CoreState
from .triad import apply_triad_move, build_triad_candidates, delta_insert, delta_remove, get_current_triad


@dataclass
class CleanupConfig:
    cleanup_max_iters: int = 1000
    cleanup_penalty_budget: float = 0.0  # allow small positive deltas if budget permits
    max_candidates: int = 16


def enforce_strict_capacity(state: CoreState, cfg: Optional[CleanupConfig] = None) -> bool:
    """
    Attempt to fix overloaded routes using relocation moves.
    Returns True if all routes are within capacity after running (or already were),
    False if could not restore feasibility within budget.
    """
    if cfg is None:
        cfg = CleanupConfig()
    penalty_used = 0.0
    iters = 0

    def overloaded_routes() -> list[int]:
        return [
            rid
            for rid, active in enumerate(state.route_active)
            if active and state.route_load[rid] > state.Q
        ]

    while iters < cfg.cleanup_max_iters:
        over = overloaded_routes()
        if not over:
            return True
        # Pick most overloaded route
        over.sort(key=lambda r: state.route_load[r] - state.Q, reverse=True)
        src = over[0]
        route_nodes = state.route_nodes[src]
        # Candidate nodes: all non-depot nodes in src
        candidates = [node for node in route_nodes if node != 0]
        moved = False
        for j in candidates:
            triad = get_current_triad(state, j)
            if triad is None:
                continue
            rid_src, pos, p, j_node, s = triad
            delta_rem = delta_remove(state, p, j_node, s)
            best_delta = None
            best_move = None
            for (i, k) in build_triad_candidates(state, j_node, max_edges=cfg.max_candidates):
                rid_tgt = state.node_route[i]
                if rid_tgt is None or rid_tgt < 0 or not state.route_active[rid_tgt]:
                    continue
                # Capacity check (strict)
                load_src_new = state.route_load[rid_src] - state.demand[j_node]
                load_tgt_new = state.route_load[rid_tgt] + state.demand[j_node]
                if load_src_new > state.Q or load_tgt_new > state.Q:
                    continue
                delta_ins = delta_insert(state, i, j_node, k)
                delta_total = delta_ins - delta_rem
                if delta_total < 0 or (delta_total >= 0 and penalty_used + delta_total <= cfg.cleanup_penalty_budget):
                    if best_delta is None or delta_total < best_delta:
                        best_delta = delta_total
                        best_move = (rid_src, pos, rid_tgt, i, k, delta_total)
            if best_move:
                rid_src, pos, rid_tgt, i_best, k_best, delta_total = best_move
                tgt_nodes = state.route_nodes[rid_tgt]
                try:
                    tgt_pos = tgt_nodes.index(i_best)
                except ValueError:
                    continue
                apply_triad_move(state, j_node, rid_src, pos, rid_tgt, tgt_pos, delta_total)
                penalty_used += max(0.0, delta_total)
                moved = True
                break
        iters += 1
        if not moved:
            break
    return len([r for r in range(len(state.route_nodes)) if state.route_active[r] and state.route_load[r] > state.Q]) == 0
