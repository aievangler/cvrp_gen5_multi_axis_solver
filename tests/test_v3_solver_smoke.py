from __future__ import annotations

from gen3.v3 import (
    STNConfig,
    InitConfig,
    LSConfig,
    SoftConfig,
    CleanupConfig,
    solve_with_state,
    read_vrp_dimacs,
    validate_solution,
    two_opt_cleanup,
)


def test_solver_smoke_tiny():
    state = read_vrp_dimacs("tests/tmp_tiny.vrp", distance_type=1)
    solve_with_state(
        state,
        stn_config=STNConfig(stn_k1=2, stn_k2=0, stn_k3=0, stn_min=1),
        init_config=InitConfig(),
        soft_config=SoftConfig(soft_run_enabled=False),
        cleanup_config=CleanupConfig(cleanup_max_iters=10),
        ls_config=LSConfig(ls_max_passes=2, max_candidates=4),
        max_plateau=2,
        two_opt_enabled=True,
        two_opt_max_iters_per_route=2,
    )
    validate_solution(state)
