from __future__ import annotations

from gen3.v3 import (
    STNConfig,
    InitConfig,
    LSConfig,
    two_opt_cleanup,
    build_initial_solution,
    build_stn,
    read_vrp_dimacs,
    run_ls,
    validate_solution,
    compute_total_cost,
)


def test_ls_runs_and_keeps_feasible():
    state = read_vrp_dimacs("tests/tmp_tiny.vrp", distance_type=1)
    build_stn(state, STNConfig(stn_k1=2, stn_k2=0, stn_k3=0, stn_min=1))
    build_initial_solution(state, InitConfig())
    validate_solution(state)
    cost_before = compute_total_cost(state)
    run_ls(state, LSConfig(ls_max_passes=2, max_candidates=4))
    validate_solution(state)
    cost_after = compute_total_cost(state)
    assert cost_after <= cost_before
