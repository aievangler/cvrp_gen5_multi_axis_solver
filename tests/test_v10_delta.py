from v10.distance import DistanceProvider
from v10.node import NodeStore
from v10.solution import Route
from v10.delta import compute_delta_remove_for_route, best_insert_delta_for_route, clone_solution_to_temp, apply_move_in_temp, Move
from v10.solution import Solution, compute_route_cost


def test_delta_remove_and_insert():
    coords = [(0, 0), (0, 10), (10, 0), (10, 10)]
    demands = [0, 1, 1, 1]
    dist = DistanceProvider(coords)
    nodes = NodeStore(coords, demands, Q=5, dist=dist)
    route = Route(seq=[1, 2, 3], load=3, cost=0)
    route.cost = compute_route_cost(route, dist)

    delta_remove = compute_delta_remove_for_route(route, dist)
    # remove middle node 2 should shortcut 1->3
    assert delta_remove[2] == dist.dist(1, 3) - (dist.dist(1, 2) + dist.dist(2, 3))

    # best insert node 2 back
    delta_ins, pos = best_insert_delta_for_route(Route(seq=[1, 3], load=2, cost=0), 2, dist)
    # best insertion is either before first or after last depending on equal deltas; ensure delta matches expected improvement
    expected_delta = dist.dist(3, 2) + dist.dist(2, 0) - dist.dist(3, 0)
    assert delta_ins == expected_delta


def test_apply_move_in_temp_updates_cost():
    coords = [(0, 0), (0, 10), (10, 0)]
    demands = [0, 1, 1]
    dist = DistanceProvider(coords)
    nodes = NodeStore(coords, demands, Q=5, dist=dist)

    r0 = Route(seq=[1], load=1, cost=compute_route_cost(Route(seq=[1], load=1, cost=0), dist))
    r1 = Route(seq=[2], load=1, cost=compute_route_cost(Route(seq=[2], load=1, cost=0), dist))
    node_owner = [-1, 0, 1]
    sol = Solution(routes=[r0, r1], node_owner=node_owner, total_cost=r0.cost + r1.cost)

    ctx = clone_solution_to_temp(sol, dist)
    # move node 1 from route 0 to route 1 after existing node
    move = Move(node_id=1, src_rid=0, dst_rid=1, dst_pos=1, delta_cost=0)
    apply_move_in_temp(ctx, nodes, dist, move)
    # node 1 now owned by route 1
    assert ctx.node_owner[1] == 1
    # route 0 should be empty
    assert ctx.routes[0].seq == []
    # route 1 should have [2,1] or [1,2] depending on insertion; we set dst_pos=1 => [2,1]
    assert ctx.routes[1].seq == [2, 1]
