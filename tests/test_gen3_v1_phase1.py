import pytest

from gen3.v1 import (
    CoreState,
    compute_total_cost,
    read_vrp,
    snapshot_state,
    validate_solution,
)


def _write_vrp(tmp_path, demands=None, capacity=10):
    if demands is None:
        demands = {1: 0, 2: 4, 3: 3, 4: 2}
    vrp = tmp_path / "tiny.vrp"
    lines = [
        "NAME : tiny",
        "TYPE : CVRP",
        "DIMENSION : 4",
        "CAPACITY : {}".format(capacity),
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        "1 0 0",
        "2 3 4",
        "3 0 3",
        "4 3 0",
        "DEMAND_SECTION",
    ]
    for idx in range(1, 5):
        lines.append(f"{idx} {demands.get(idx, 0)}")
    lines += ["DEPOT_SECTION", "1", "-1", "EOF"]
    vrp.write_text("\n".join(lines) + "\n")
    return vrp


def test_read_vrp_parses_and_shifts(tmp_path):
    vrp_path = _write_vrp(tmp_path)
    state = read_vrp(vrp_path)
    assert state.N == 3
    assert state.Q == 10
    assert state.depot == 0
    # Depot shifted to origin
    assert state.x[0] == 0
    assert state.y[0] == 0
    # Distance rounding check: between nodes (3,4) and (3,0) -> 4
    assert state.dist(1, 3) == 4
    # Geometry computed
    assert pytest.approx(state.r[1]) == 5.0
    assert pytest.approx(state.theta[1]) == pytest.approx(0.927295218)  # atan2(4,3)


def test_infeasible_demand_raises(tmp_path):
    vrp_path = _write_vrp(tmp_path, demands={1: 0, 2: 15, 3: 1, 4: 1}, capacity=10)
    with pytest.raises(ValueError):
        read_vrp(vrp_path)


def _attach_single_route(state: CoreState) -> None:
    route = [0, 1, 2, 3, 0]
    state.route_nodes = [route]
    state.route_active = [True]
    state.route_load = [state.demand[1] + state.demand[2] + state.demand[3]]
    state.route_mask = [(1 << 1) | (1 << 2) | (1 << 3)]
    state.node_route = [None] * (state.N + 1)
    state.node_pos = [None] * (state.N + 1)
    for idx, node in enumerate(route):
        state.node_route[node] = 0
        state.node_pos[node] = idx
    cost = compute_total_cost(state)
    state.C0 = cost
    state.current_cost = cost


def test_validate_solution_happy_path(tmp_path):
    state = read_vrp(_write_vrp(tmp_path))
    _attach_single_route(state)
    validate_solution(state)


def test_validate_solution_detects_duplicates(tmp_path):
    state = read_vrp(_write_vrp(tmp_path))
    bad_route = [0, 1, 1, 2, 3, 0]
    state.route_nodes = [bad_route]
    state.route_active = [True]
    with pytest.raises(ValueError):
        validate_solution(state)


def test_snapshot_preserves_geometry_and_routes(tmp_path):
    state = read_vrp(_write_vrp(tmp_path))
    _attach_single_route(state)
    snap = snapshot_state(state)
    assert snap.route_nodes == state.route_nodes
    # Mutate original and ensure snapshot unaffected
    state.route_nodes[0].insert(1, 2)
    assert snap.route_nodes != state.route_nodes
