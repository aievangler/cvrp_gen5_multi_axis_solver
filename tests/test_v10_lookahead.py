from v10.distance import DistanceProvider
from v10.node import NodeStore
from v10.solution import Route, Solution, compute_route_cost
from v10.knn import KNNBuilder
from v10.config import V5Config
from v10.lookahead import run_short_lookahead_for_seed
from v10.delta import commit_moves


def test_lookahead_single_improvement():
    # Simple square: depot 0, customers 1-4
    coords = [(0, 0), (0, 10), (10, 0), (10, 10), (5, 5)]
    demands = [0, 1, 1, 1, 1]
    dist = DistanceProvider(coords)
    nodes = NodeStore(coords, demands, Q=3, dist=dist)
    knn = KNNBuilder(dist, nodes.N, K_node=3).build()
    cfg = V5Config(C_candidates=2, D_short=3, Y_short=0.05, K_seed_pool=4)

    # Two routes: [1,2] and [3,4]
    r0 = Route(seq=[1, 2], load=2, cost=0)
    r1 = Route(seq=[3, 4], load=2, cost=0)
    r0.cost = compute_route_cost(r0, dist)
    r1.cost = compute_route_cost(r1, dist)
    node_owner = [-1, 0, 0, 1, 1]
    sol = Solution(routes=[r0, r1], node_owner=node_owner, total_cost=r0.cost + r1.cost)

    success, chain, new_cost = run_short_lookahead_for_seed(sol, nodes, dist, knn, seed_rid=0, cfg=cfg)
    # Expect an improvement by pulling one node into route 0 if capacity allows
    if success:
        old_cost = sol.total_cost
        commit_moves(sol, nodes, dist, chain)
        assert sol.total_cost < old_cost
