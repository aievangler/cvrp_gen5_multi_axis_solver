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
    soft_run,
)


def _vrp_soft(tmp_path):
    vrp = tmp_path / "soft.vrp"
    lines = [
        "NAME : soft",
        "TYPE : CVRP",
        "DIMENSION : 4",
        "CAPACITY : 3",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        "1 0 0",
        "2 0 10",   # demand 3
        "3 0 11",   # demand 1
        "4 10 0",   # demand 1
        "DEMAND_SECTION",
        "1 0",
        "2 3",
        "3 1",
        "4 1",
        "DEPOT_SECTION",
        "1",
        "-1",
        "EOF",
    ]
    vrp.write_text("\n".join(lines) + "\n")
    return vrp


def _prepare_state(tmp_path):
    state = read_vrp(_vrp_soft(tmp_path))
    build_stn(state, STNConfig(stn_k1=3, stn_k2=0, stn_k3=0, stn_min=2))
    build_initial_solution(state, InitConfig())
    # Override routes: [0,2,0] (load3) and [0,3,4,0] (load2)
    state.route_nodes = [
        [0, 1, 0],  # internal 1=ext2
        [0, 2, 3, 0],  # internal 2=ext3, 3=ext4
    ]
    state.route_active = [True, True]
    state.route_load = [state.demand[1], state.demand[2] + state.demand[3]]
    state.route_mask = [(1 << 1), (1 << 2) | (1 << 3)]
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
    return state


def test_soft_run_allows_over_Q_move(tmp_path):
    state = _prepare_state(tmp_path)
    before = compute_total_cost(state)
    moved_soft = soft_run(state, soft_Q=4, max_passes=2, triad_cfg=TriadConfig(max_candidates=5))
    after = compute_total_cost(state)
    assert moved_soft is True
    assert after < before
    # Route0 should now include node2 (internal id 2)
    assert any(2 in r for r in state.route_nodes)
    # Load on route containing node1 should now be 4 (>Q but <=soft_Q)
    rid = state.node_route[1]
    assert state.route_load[rid] == 4


def test_strict_ls_blocks_over_Q_move(tmp_path):
    state = _prepare_state(tmp_path)
    before = compute_total_cost(state)
    moved_strict = run_ls(state, max_passes=2, triad_cfg=TriadConfig(max_candidates=5), strict_Q=True)
    after = compute_total_cost(state)
    assert moved_strict is False
    assert after == before
