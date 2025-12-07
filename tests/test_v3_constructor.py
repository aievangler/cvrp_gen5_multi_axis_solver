from __future__ import annotations

from gen3.v3 import (
    STNConfig,
    InitConfig,
    build_initial_solution,
    build_stn,
    read_vrp_dimacs,
    validate_solution,
)


def test_constructor_tiny_euc():
    state = read_vrp_dimacs("tests/tmp_tiny.vrp", distance_type=1)
    build_stn(state, STNConfig(stn_k1=2, stn_k2=0, stn_k3=0, stn_min=1))
    build_initial_solution(state, InitConfig())
    validate_solution(state)
    # All customers assigned
    assert all(rid != -1 for rid in state.node_route[1:])
    assert len(state.route_nodes) >= 1


def test_constructor_a_instance():
    state = read_vrp_dimacs("data/instances/A/A-n32-k5.vrp", distance_type=1)
    build_stn(state, STNConfig())
    build_initial_solution(state, InitConfig())
    validate_solution(state)
    assert all(rid != -1 for rid in state.node_route[1:])
