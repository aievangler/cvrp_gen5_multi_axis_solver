import pytest

from gen3.v1 import (
    InitConfig,
    STNConfig,
    build_initial_solution,
    build_stn,
    compute_total_cost,
    read_vrp,
    validate_solution,
)


def _vrp_for_constructor(tmp_path, capacity=5):
    vrp = tmp_path / "construct.vrp"
    lines = [
        "NAME : construct",
        "TYPE : CVRP",
        "DIMENSION : 5",
        "CAPACITY : {}".format(capacity),
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        "1 0 0",
        "2 5 0",
        "3 5 5",
        "4 0 5",
        "5 10 0",
        "DEMAND_SECTION",
        "1 0",
        "2 3",
        "3 2",
        "4 2",
        "5 1",
        "DEPOT_SECTION",
        "1",
        "-1",
        "EOF",
    ]
    vrp.write_text("\n".join(lines) + "\n")
    return vrp


def test_constructor_assigns_all_and_respects_capacity(tmp_path):
    state = read_vrp(_vrp_for_constructor(tmp_path))
    build_stn(state, STNConfig(stn_k1=3, stn_k2=1, stn_k3=0, stn_min=2))
    build_initial_solution(state, InitConfig())
    validate_solution(state)
    for load in state.route_load:
        assert load <= state.Q


def test_constructor_farthest_first_tie_break(tmp_path):
    # Two farthest nodes at equal radius; expect smallest ID chosen as first seed.
    vrp = tmp_path / "tie.vrp"
    lines = [
        "NAME : tie",
        "TYPE : CVRP",
        "DIMENSION : 4",
        "CAPACITY : 10",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        "1 0 0",
        "2 3 4",   # r=5
        "3 -3 4",  # r=5
        "4 1 1",   # r~1.41
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
    state = read_vrp(vrp)
    build_stn(state, STNConfig(stn_k1=3, stn_k2=0, stn_k3=0, stn_min=2))
    build_initial_solution(state, InitConfig())
    first_route = state.route_nodes[0]
    assert first_route[0] == 0  # depot
    assert first_route[1] == 1  # internal id of external node 2 (smallest farthest)


def test_constructor_uses_global_fallback(tmp_path):
    # Capacity blocks STN candidate; should fall back to global feasible node.
    vrp = tmp_path / "fallback.vrp"
    lines = [
        "NAME : fallback",
        "TYPE : CVRP",
        "DIMENSION : 4",
        "CAPACITY : 4",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        "1 0 0",
        "2 0 10",   # demand 3 (seed)
        "3 0 9",    # demand 2 (too heavy to fit)
        "4 1 0",    # demand 1 (fallback)
        "DEMAND_SECTION",
        "1 0",
        "2 3",
        "3 2",
        "4 1",
        "DEPOT_SECTION",
        "1",
        "-1",
        "EOF",
    ]
    vrp.write_text("\n".join(lines) + "\n")
    state = read_vrp(vrp)
    build_stn(state, STNConfig(stn_k1=1, stn_k2=0, stn_k3=0, stn_min=1))
    build_initial_solution(state, InitConfig())
    validate_solution(state)
    # Seed is node2 (internal id 1); node3 cannot fit, so fallback should pick node4 (internal id 3)
    first_route = state.route_nodes[0]
    assert first_route[1] == 1
    assert 3 in first_route
    assert 2 not in first_route  # internal id 2 (external 3) should not be in the route due to capacity


def test_constructor_cost_bookkeeping(tmp_path):
    state = read_vrp(_vrp_for_constructor(tmp_path))
    build_stn(state, STNConfig(stn_k1=3, stn_k2=1, stn_k3=0, stn_min=2))
    build_initial_solution(state, InitConfig())
    recomputed = compute_total_cost(state)
    assert state.C0 == recomputed
    assert state.current_cost == recomputed
    assert state.cum_delta == 0.0
