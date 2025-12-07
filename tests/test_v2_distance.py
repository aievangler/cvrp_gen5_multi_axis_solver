from gen3.v1.core import CoreState
from gen3.v2 import read_vrp


def make_state(distance_type):
    # Tiny 2-customer VRP: depot=0, customers 1,2 at (0,1) and (0,2)
    vrp_text = """NAME: tiny
TYPE: CVRP
DIMENSION: 3
CAPACITY: 100
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0 0
2 0 1
3 0 2
DEMAND_SECTION
1 0
2 1
3 1
DEPOT_SECTION
1
-1
EOF
"""
    path = "tests/tmp_tiny.vrp"
    with open(path, "w") as f:
        f.write(vrp_text)
    state = read_vrp(path, distance_type=distance_type)
    return state


def test_distance_rounded():
    state = make_state(1)
    # dist(1,2): between (0,1) and (0,2) = 1.0 -> rounded to 1
    assert state.dist(1, 2) == 1


def test_distance_unrounded():
    state = make_state(0)
    assert state.dist(1, 2) == 1  # same here, but uses int(hypot)


def test_distance_explicit():
    # Build a 3x3 explicit matrix with asymmetric values
    vrp_text = "\n".join(
        [
            "NAME: tiny_explicit",
            "TYPE: CVRP",
            "DIMENSION: 3",
            "CAPACITY: 100",
            "EDGE_WEIGHT_TYPE: EXPLICIT",
            "EDGE_WEIGHT_FORMAT: FULL_MATRIX",
            "EDGE_WEIGHT_SECTION",
            "0 1 2",
            "1 0 3",
            "2 3 0",
            "NODE_COORD_SECTION",
            "1 0 0",
            "2 0 1",
            "3 0 2",
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
    path = "tests/tmp_tiny_explicit.vrp"
    with open(path, "w") as f:
        f.write(vrp_text)
    state = read_vrp(path, distance_type=2)
    # dist_matrix is size (N+1)x(N+1) with depot=0, customers 1..N
    assert len(state.dist_matrix) == state.N + 1
    assert len(state.dist_matrix[0]) == state.N + 1
    assert state.distance_type == 2
    # Internal ids: depot=0, customers 1,2
    assert state.dist(1, 2) == 3  # row 1 col 2
    assert state.dist(2, 1) == 3  # row 2 col 1
    assert state.dist(0, 1) == 1  # row 0 col 1
