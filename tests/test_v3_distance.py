from gen3.v3 import CoreState, compute_total_cost, read_vrp_dimacs


def make_euc_state(distance_type: int) -> CoreState:
    # Depot at (0,0), customers at (0,1) and (0,2); demands irrelevant here.
    state = CoreState(
        N=2,
        Q=100,
        distance_type=distance_type,
        x=[0.0, 0.0, 0.0],
        y=[0.0, 1.0, 2.0],
        demand=[0, 1, 1],
    )
    # Single route visiting both customers
    state.route_nodes = [[0, 1, 2, 0]]
    state.route_active = [True]
    return state


def test_distance_rounded():
    state = make_euc_state(distance_type=1)
    assert state.dist(1, 2) == 1  # rounded hypot(0,1)
    cost = compute_total_cost(state)
    # 0->1 (1), 1->2 (1), 2->0 (2) = 4
    assert cost == 4


def test_distance_unrounded():
    state = make_euc_state(distance_type=0)
    assert state.dist(1, 2) == 1.0
    cost = compute_total_cost(state)
    assert cost == 4.0


def test_distance_explicit():
    # Use a tiny explicit instance via the parser.
    vrp_text = "\n".join(
        [
            "NAME: tiny_explicit",
            "TYPE: CVRP",
            "DIMENSION: 3",
            "CAPACITY: 100",
            "EDGE_WEIGHT_TYPE: EXPLICIT",
            "EDGE_WEIGHT_FORMAT: LOWER_ROW",
            "EDGE_WEIGHT_SECTION",
            "5",
            "7 3",
            "NODE_COORD_SECTION",
            "1 0 0",
            "2 0 0",
            "3 0 0",
            "DEMAND_SECTION",
            "1 0",
            "2 1",
            "3 1",
            "DEPOT_SECTION",
            "1",
            "-1",
            "EOF",
        ]
    )
    path = "tests/tmp_tiny_explicit_v3.vrp"
    with open(path, "w") as f:
        f.write(vrp_text)
    state = read_vrp_dimacs(path, distance_type=2)
    state.route_nodes = [[0, 1, 2, 0]]
    state.route_active = [True]
    assert state.dist(0, 1) == 5
    assert state.dist(1, 2) == 3
    assert state.dist(0, 2) == 7
    cost = compute_total_cost(state)
    assert cost == 15
