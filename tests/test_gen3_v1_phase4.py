import pytest

from gen3.v1 import (
    InitConfig,
    STNConfig,
    TriadConfig,
    build_initial_solution,
    build_stn,
    build_triad_candidates,
    delta_insert,
    delta_remove,
    get_current_triad,
    read_vrp,
)


def _vrp_basic(tmp_path):
    vrp = tmp_path / "triad.vrp"
    lines = [
        "NAME : triad",
        "TYPE : CVRP",
        "DIMENSION : 4",
        "CAPACITY : 10",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        "1 0 0",
        "2 0 10",
        "3 10 10",
        "4 10 0",
        "DEMAND_SECTION",
        "1 0",
        "2 1",
        "3 1",
        "4 1",
        "DEPOT_SECTION",
        "1",
        "-1",
        "EOF",
    ]
    vrp.write_text("\n".join(lines) + "\n")
    return vrp


def _build_state_with_route(tmp_path):
    state = read_vrp(_vrp_basic(tmp_path))
    build_stn(state, STNConfig(stn_k1=3, stn_k2=0, stn_k3=0, stn_min=2))
    build_initial_solution(state, InitConfig())
    return state


def test_get_current_triad_and_deltas(tmp_path):
    state = _build_state_with_route(tmp_path)
    # Force a known route for determinism: 0-2-3-4-0
    state.route_nodes = [[0, 1, 2, 3, 0]]
    state.route_active = [True]
    state.route_load = [state.demand[1] + state.demand[2] + state.demand[3]]
    state.route_mask = [(1 << 1) | (1 << 2) | (1 << 3)]
    state.node_route = [0, 0, 0, 0]
    state.node_pos = [0, 1, 2, 3]
    p, j, s = get_current_triad(state, 2)
    assert (p, j, s) == (1, 2, 3)
    rem = delta_remove(state, 2)
    ins = delta_insert(state, 2, 0, 1)
    # Delta remove: dist(1,2)+dist(2,3)-dist(1,3)
    assert rem == state.dist(1, 2) + state.dist(2, 3) - state.dist(1, 3)
    assert ins == state.dist(0, 2) + state.dist(2, 1) - state.dist(0, 1)


def test_triad_candidates_bounded_and_local(tmp_path):
    state = _build_state_with_route(tmp_path)
    cfg = TriadConfig(max_candidates=2)
    cands = build_triad_candidates(state, 1, cfg)
    assert len(cands) <= cfg.max_candidates
    # Candidates derived from neighbors of node 1
    for (i, k) in cands:
        assert i != 1 and k != 1
        assert i in state.stn[1] or k in state.stn[1] or i == 0 or k == 0
