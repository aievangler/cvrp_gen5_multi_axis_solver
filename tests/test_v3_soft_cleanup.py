from __future__ import annotations

import copy

from gen3.v3 import (
    STNConfig,
    InitConfig,
    LSConfig,
    SoftConfig,
    CleanupConfig,
    build_initial_solution,
    build_stn,
    compute_total_cost,
    enforce_strict_capacity,
    read_vrp_dimacs,
    run_ls,
    soft_run,
    validate_solution,
)


def test_soft_and_cleanup_keep_feasible():
    state = read_vrp_dimacs("tests/tmp_tiny.vrp", distance_type=1)
    build_stn(state, STNConfig(stn_k1=2, stn_k2=0, stn_k3=0, stn_min=1))
    build_initial_solution(state, InitConfig())
    validate_solution(state)
    base_cost = compute_total_cost(state)

    # soft run (relaxed Q) followed by strict cleanup and LS
    soft_run(state, SoftConfig(soft_run_enabled=True, soft_run_max_passes=1))
    ok = enforce_strict_capacity(state, CleanupConfig(cleanup_max_iters=10))
    assert ok
    improved = run_ls(state, LSConfig(ls_max_passes=1, max_candidates=4))
    validate_solution(state)
    # Cost should not increase after cleanup+LS
    assert compute_total_cost(state) <= base_cost or improved
