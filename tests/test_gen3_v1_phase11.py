import pathlib

from gen3.v1 import (
    CleanupConfig,
    InitConfig,
    LSConfig,
    SoftConfig,
    STNConfig,
    analyze,
    compute_total_cost,
    solve,
    solve_and_write,
)


def _vrp_integration(tmp_path):
    vrp = tmp_path / "int.vrp"
    lines = [
        "NAME : int",
        "TYPE : CVRP",
        "DIMENSION : 5",
        "CAPACITY : 5",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        "1 0 0",
        "2 0 5",
        "3 5 5",
        "4 5 0",
        "5 2 2",
        "DEMAND_SECTION",
        "1 0",
        "2 2",
        "3 2",
        "4 1",
        "5 1",
        "DEPOT_SECTION",
        "1",
        "-1",
        "EOF",
    ]
    vrp.write_text("\n".join(lines) + "\n")
    return vrp


def test_solve_and_restore_best(tmp_path):
    vrp = _vrp_integration(tmp_path)
    state, cost = solve(
        str(vrp),
        stn_config=STNConfig(stn_k1=4, stn_k2=1, stn_k3=0, stn_min=3),
        init_config=InitConfig(),
        soft_config=SoftConfig(soft_q_factor=1.5, soft_run_max_passes=1, soft_run_enabled=True),
        cleanup_config=CleanupConfig(cleanup_max_iters=50, cleanup_penalty_budget=0.0),
        ls_config=LSConfig(ls_max_passes=3),
        max_plateau=2,
    )
    # Feasible and cost consistent with routes
    assert cost == compute_total_cost(state)
    seen = [0] * (state.N + 1)
    for rid, route in enumerate(state.route_nodes):
        if state.route_active and rid < len(state.route_active) and not state.route_active[rid]:
            continue
        load = 0
        for node in route[1:-1]:
            seen[node] += 1
            load += state.demand[node]
        assert load <= state.Q
    assert all(cnt == 1 for cnt in seen[1:])


def test_solve_and_analyze_round_trip(tmp_path):
    vrp = _vrp_integration(tmp_path)
    sol_path = tmp_path / "out.sol"
    cost = solve_and_write(
        str(vrp),
        str(sol_path),
        stn_config=STNConfig(stn_k1=4, stn_k2=1, stn_k3=0, stn_min=3),
        init_config=InitConfig(),
        soft_config=SoftConfig(soft_q_factor=1.5, soft_run_max_passes=1, soft_run_enabled=True),
        cleanup_config=CleanupConfig(cleanup_max_iters=50, cleanup_penalty_budget=0.0),
        ls_config=LSConfig(ls_max_passes=3),
        max_plateau=2,
    )
    assert sol_path.exists()
    # Run analysis against itself and ensure fields present
    result = analyze(
        str(vrp),
        str(sol_path),
        str(sol_path),
        STNConfig(stn_k1=4, stn_k2=1, stn_k3=0, stn_min=3),
    )
    assert "coverage" in result and "ddc_bks" in result and "edge_overlap" in result
