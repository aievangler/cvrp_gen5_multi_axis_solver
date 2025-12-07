from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .core import CoreState
from .triad import apply_triad_move, build_triad_candidates, delta_insert, delta_remove, get_current_triad


@dataclass
class LSConfig:
    ls_max_passes: int = 5
    max_candidates: int = 16


def run_ls(state: CoreState, cfg: Optional[LSConfig] = None) -> bool:
    """
    Simple depth-1 triad relocation LS with strict capacity and delta<0 acceptance.
    Returns True if any improvement was applied.
    """
    if cfg is None:
        cfg = LSConfig()
    improved_any = False
    for _ in range(cfg.ls_max_passes):
        improved_pass = False
        # Process nodes in order 1..N
        for j in range(1, state.N + 1):
            triad = get_current_triad(state, j)
            if triad is None:
                continue
            rid_src, pos, p, j_node, s = triad
            if not state.route_active[rid_src]:
                continue
            delta_rem = delta_remove(state, p, j_node, s)
            best_delta = 0.0
            best_move = None
            candidates = build_triad_candidates(state, j_node, max_edges=cfg.max_candidates)
            for (i, k) in candidates:
                rid_tgt = state.node_route[i]
                if rid_tgt is None or rid_tgt < 0 or not state.route_active[rid_tgt]:
                    continue
                # disallow trivial insertion at same place
                if rid_tgt == rid_src and (i == p and k == j_node):
                    continue
                delta_ins = delta_insert(state, i, j_node, k)
                delta_total = delta_ins - delta_rem
                if delta_total >= best_delta:
                    continue
                # capacity check
                load_src_new = state.route_load[rid_src] - state.demand[j_node]
                load_tgt_new = state.route_load[rid_tgt] + state.demand[j_node]
                if load_src_new > state.Q or load_tgt_new > state.Q:
                    continue
                best_delta = delta_total
                best_move = (rid_src, pos, rid_tgt, i, k)
            if best_move is not None:
                rid_src, pos, rid_tgt, i_best, k_best = best_move
                tgt_route_nodes = state.route_nodes[rid_tgt]
                # find position of edge (i_best, k_best)
                try:
                    tgt_pos = tgt_route_nodes.index(i_best)
                except ValueError:
                    continue
                apply_triad_move(state, j_node, rid_src, pos, rid_tgt, tgt_pos, best_delta)
                improved_pass = True
                improved_any = True
        if not improved_pass:
            break
    return improved_any
