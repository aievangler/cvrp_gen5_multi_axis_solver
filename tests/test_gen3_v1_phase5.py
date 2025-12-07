import pytest

from gen3.v1 import (
    InitConfig,
    STNConfig,
    TriadConfig,
    build_initial_solution,
    build_stn,
    compute_total_cost,
    read_vrp,
    run_ls,
)


def _vrp_ls(tmp_path):
    vrp = tmp_path / "ls.vrp"
    lines = [
        "NAME : ls",
        "TYPE : CVRP",
        "DIMENSION : 5",
        "CAPACITY : 10",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        "1 0 0",
        "2 0 10",
        "3 10 10",
        "4 10 0",
        "5 5 0",
        "DEMAND_SECTION",
        "1 0",
        "2 1",
        "3 1",
        "4 1",
        "5 1",
        "DEPOT_SECTION",
        "1",
        "-1",
        "EOF",
    ]
    vrp.write_text("\n".join(lines) + "\n")
    return vrp


def _attach_routes(state):
    # Two routes: [0,2,3,0] and [0,4,5,0]; moving node3 to route2 between 4,5 reduces cost.
    state.route_nodes = [
        [0, 1, 2, 0],  # internal ids: 1->ext2, 2->ext3
        [0, 3, 4, 0],  # internal ids: 3->ext4, 4->ext5
    ]
    state.route_active = [True, True]
    state.route_load = [state.demand[1] + state.demand[2], state.demand[3] + state.demand[4]]
    state.route_mask = [(1 << 1) | (1 << 2), (1 << 3) | (1 << 4)]
    size = state.N + 1
    state.node_route = [None] * size
    state.node_pos = [None] * size
    for rid, route in enumerate(state.route_nodes):
        for pos, node in enumerate(route):
            state.node_route[node] = rid
            state.node_pos[node] = pos
    cost = compute_total_cost(state)
    state.C0 = cost
    state.current_cost = cost
    state.cum_delta = 0.0


def test_run_ls_applies_improving_move(tmp_path):
    state = read_vrp(_vrp_ls(tmp_path))
    build_stn(state, STNConfig(stn_k1=4, stn_k2=0, stn_k3=0, stn_min=3))
    build_initial_solution(state, InitConfig())  # sets geometry etc., but we override routes
    _attach_routes(state)
    before = compute_total_cost(state)
    moved = run_ls(state, max_passes=3, triad_cfg=TriadConfig(max_candidates=5))
    after = compute_total_cost(state)
    assert moved is True
    assert after == state.current_cost
    assert after < before
    # We expect consolidation: node 3 should move into the first route, leaving the second empty/inactive.
    assert state.node_route[3] == 0
    assert state.route_active[1] is False or len(state.route_nodes[1]) == 2


def test_run_ls_respects_capacity(tmp_path):
    state = read_vrp(_vrp_ls(tmp_path))
    state.Q = 1  # make moves infeasible
    build_stn(state, STNConfig(stn_k1=4, stn_k2=0, stn_k3=0, stn_min=3))
    build_initial_solution(state, InitConfig())
    _attach_routes(state)
    before = compute_total_cost(state)
    moved = run_ls(state, max_passes=2, triad_cfg=TriadConfig(max_candidates=5), strict_Q=True)
    after = compute_total_cost(state)
    assert moved is False
    assert after == before
