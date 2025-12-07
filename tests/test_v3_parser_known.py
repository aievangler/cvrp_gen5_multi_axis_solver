from gen3.v3 import compute_total_cost, read_vrp_dimacs


def test_a_n32_k5_cost_matches_bks():
    # Known BKS: 784; distance_type = 1 (EUC_2D rounded)
    state = read_vrp_dimacs("data/instances/A/A-n32-k5.vrp", distance_type=1)
    # Known optimal routes from .sol (Route #k: ... customers, depot implicit)
    routes = [
        [0, 21, 31, 19, 17, 13, 7, 26, 0],
        [0, 12, 1, 16, 30, 0],
        [0, 27, 24, 0],
        [0, 29, 18, 8, 9, 22, 15, 10, 25, 5, 20, 0],
        [0, 14, 28, 11, 4, 23, 3, 2, 6, 0],
    ]
    state.route_nodes = routes
    state.route_active = [True] * len(routes)
    cost = compute_total_cost(state)
    assert int(cost) == 784


def test_a_n33_k5_cost_matches_bks():
    # Known BKS: 661; distance_type = 1 (EUC_2D rounded)
    state = read_vrp_dimacs("data/instances/A/A-n33-k5.vrp", distance_type=1)
    routes = [
        [0, 15, 17, 9, 3, 16, 29, 0],
        [0, 12, 5, 26, 7, 8, 13, 32, 2, 0],
        [0, 20, 4, 27, 25, 30, 10, 0],
        [0, 23, 28, 18, 22, 0],
        [0, 24, 6, 19, 14, 21, 1, 31, 11, 0],
    ]
    state.route_nodes = routes
    state.route_active = [True] * len(routes)
    cost = compute_total_cost(state)
    assert int(cost) == 661
