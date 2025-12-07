import pytest

from gen3.v1 import (
    InitConfig,
    STNConfig,
    TriadConfig,
    build_initial_solution,
    build_stn,
    compute_total_cost,
    enforce_strict_capacity,
    init_best,
    read_vrp,
    rollback_to_best,
    update_best_if_improved,
)


def _vrp_snapshot(tmp_path):
    vrp = tmp_path / "snap.vrp"
    lines = [
        "NAME : snap",
        "TYPE : CVRP",
        "DIMENSION : 4",
        "CAPACITY : 3",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        "1 0 0",
        "2 0 1",
        "3 0 2",
        "4 10 0",
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


def _prepare_state(tmp_path):
    state = read_vrp(_vrp_snapshot(tmp_path))
    build_stn(state, STNConfig(stn_k1=3, stn_k2=0, stn_k3=0, stn_min=2))
    build_initial_solution(state, InitConfig())
    # Two routes: overloaded [0, node2, node3, 0], slack [0, node4,0]
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


def test_init_and_update_best(tmp_path):
    state = _prepare_state(tmp_path)
    best_sol, best_feasible, best_cost = init_best(state)
    assert best_cost == state.current_cost
    # Worsen cost: add detour, should not update best
    state.route_nodes[0].insert(1, 3)
    state.route_load[0] += state.demand[3]
    state.node_route[3] = 0
    state.node_pos[3] = 1
    state.current_cost = compute_total_cost(state)
    new_best_sol, new_best_feasible, new_best_cost = update_best_if_improved(
        state, best_sol, best_feasible, best_cost
    )
    assert new_best_cost == best_cost
    assert new_best_sol.route_nodes == best_sol.route_nodes


def test_rollback_on_cleanup_failure(tmp_path):
    state = _prepare_state(tmp_path)
    best_sol, best_feasible, best_cost = init_best(state)
    # Make target route full to force cleanup failure
    state.route_load[1] = state.Q
    ok = enforce_strict_capacity(state, max_iters=2, penalty_budget=0.0, triad_cfg=TriadConfig(max_candidates=2))
    if not ok:
        rollback_to_best(state, best_feasible)
    assert state.route_load == best_feasible.route_load
    assert state.route_nodes == best_feasible.route_nodes
