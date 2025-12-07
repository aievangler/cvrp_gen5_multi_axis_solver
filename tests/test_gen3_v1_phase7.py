import pytest

from gen3.v1 import (
    InitConfig,
    STNConfig,
    TriadConfig,
    build_initial_solution,
    build_stn,
    compute_total_cost,
    enforce_strict_capacity,
    read_vrp,
)


def _vrp_capacity(tmp_path):
    vrp = tmp_path / "cap.vrp"
    lines = [
        "NAME : cap",
        "TYPE : CVRP",
        "DIMENSION : 4",
        "CAPACITY : 3",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        "1 0 0",
        "2 0 1",   # demand 2
        "3 0 2",   # demand 2
        "4 100 0", # demand 1
        "DEMAND_SECTION",
        "1 0",
        "2 2",
        "3 2",
        "4 1",
        "DEPOT_SECTION",
        "1",
        "-1",
        "EOF",
    ]
    vrp.write_text("\n".join(lines) + "\n")
    return vrp


def _setup_overloaded_state(tmp_path):
    state = read_vrp(_vrp_capacity(tmp_path))
    build_stn(state, STNConfig(stn_k1=3, stn_k2=0, stn_k3=0, stn_min=2))
    build_initial_solution(state, InitConfig())
    # Routes: [0, node2, node3, 0] load=4>Q, [0, node4, 0] load=1
    state.route_nodes = [
        [0, 1, 2, 0],
        [0, 3, 0],
    ]
    state.route_active = [True, True]
    state.route_load = [state.demand[1] + state.demand[2], state.demand[3]]
    state.route_mask = [(1 << 1) | (1 << 2), (1 << 3)]
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


def test_cleanup_fixes_overload_with_nonpositive_delta(tmp_path):
    state = _setup_overloaded_state(tmp_path)
    ok = enforce_strict_capacity(state, max_iters=10, penalty_budget=0.0, triad_cfg=TriadConfig(max_candidates=5))
    assert ok is True
    assert all(load <= state.Q for load in state.route_load if load is not None)


def test_cleanup_allows_penalty_move(tmp_path):
    state = _setup_overloaded_state(tmp_path)
    # Adjust geometry to force delta > 0 move: keep overloaded route cost similar, moving node1 (demand2) into far route increases cost.
    state.route_nodes = [
        [0, 1, 2, 0],
        [0, 3, 0],
    ]
    state.route_active = [True, True]
    state.route_load = [state.demand[1] + state.demand[2], state.demand[3]]
    state.route_mask = [(1 << 1) | (1 << 2), (1 << 3)]
    size = state.N + 1
    state.node_route = [None] * size
    state.node_pos = [None] * size
    for rid, route in enumerate(state.route_nodes):
        for pos, node in enumerate(route):
            state.node_route[node] = rid
            state.node_pos[node] = pos
    # Move node2 into route1 will slightly increase cost; allow small penalty.
    before = compute_total_cost(state)
    ok = enforce_strict_capacity(state, max_iters=10, penalty_budget=10.0, triad_cfg=TriadConfig(max_candidates=5))
    after = compute_total_cost(state)
    assert ok is True
    assert all(load <= state.Q for load in state.route_load if load is not None)
    assert after >= before  # allow cost increase due to penalty budget


def test_cleanup_returns_false_when_stuck(tmp_path):
    state = _setup_overloaded_state(tmp_path)
    # Disallow moves by setting penalty budget 0 and removing target slack
    state.route_load[1] = state.Q  # make target full
    ok = enforce_strict_capacity(state, max_iters=2, penalty_budget=0.0, triad_cfg=TriadConfig(max_candidates=2))
    assert ok is False
