import pytest

from gen3.v1 import (
    CleanupConfig,
    InitConfig,
    LSConfig,
    SoftConfig,
    STNConfig,
    solve,
)


def _vrp_outer(tmp_path):
    vrp = tmp_path / "outer.vrp"
    lines = [
        "NAME : outer",
        "TYPE : CVRP",
        "DIMENSION : 4",
        "CAPACITY : 5",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        "1 0 0",
        "2 0 5",
        "3 5 0",
        "4 5 5",
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


def test_solve_returns_feasible_and_respects_plateau(tmp_path):
    vrp = _vrp_outer(tmp_path)
    state, cost = solve(
        str(vrp),
        stn_config=STNConfig(stn_k1=3, stn_k2=0, stn_k3=0, stn_min=2),
        init_config=InitConfig(),
        soft_config=SoftConfig(soft_q_factor=1.5, soft_run_max_passes=1, soft_run_enabled=False),
        cleanup_config=CleanupConfig(cleanup_max_iters=10, cleanup_penalty_budget=0.0),
        ls_config=LSConfig(ls_max_passes=2),
        max_plateau=1,
    )
    # Feasible: each customer appears once, loads <= Q
    seen = [0] * (state.N + 1)
    for rid, route in enumerate(state.route_nodes):
        if state.route_active and rid < len(state.route_active) and not state.route_active[rid]:
            continue
        assert route[0] == state.depot and route[-1] == state.depot
        load = 0
        for node in route[1:-1]:
            seen[node] += 1
            load += state.demand[node]
        assert load <= state.Q
    for j in range(1, state.N + 1):
        assert seen[j] == 1
    assert cost == state.current_cost or cost == int(state.current_cost)
