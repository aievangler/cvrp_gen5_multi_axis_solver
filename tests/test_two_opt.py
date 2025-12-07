from gen3.v1.core import CoreState, build_edge_index, compute_total_cost
from gen3.v1.ls import two_opt_cleanup, two_opt_route


def make_crossing_route_state() -> CoreState:
    # Square with a zig-zag crossing: route [0,1,3,2,4,0] crosses at center.
    # Coordinates scaled so 2-opt yields a clear cost improvement:
    # 1:(0,0), 2:(0,10), 3:(10,0), 4:(10,10)
    state = CoreState(
        N=4,
        Q=10,
        depot=0,
        x=[0.0, 0.0, 0.0, 10.0, 10.0],
        y=[0.0, 0.0, 10.0, 0.0, 10.0],
        r=[0.0] * 5,
        theta=[0.0] * 5,
        demand=[0, 1, 1, 1, 1],
        route_nodes=[[0, 1, 3, 2, 4, 0]],
        route_load=[4],
        route_mask=[(1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)],
        route_active=[True],
        node_route=[0, 0, 0, 0, 0],
        node_pos=[0, 1, 3, 2, 4],
        C0=0.0,
        cum_delta=0.0,
        current_cost=0.0,
    )
    state.edge_index = build_edge_index(state)
    state.current_cost = compute_total_cost(state)
    state.C0 = state.current_cost
    return state


def test_two_opt_route_improves_crossing():
    state = make_crossing_route_state()
    old_cost = compute_total_cost(state)
    improved = two_opt_route(state, 0)
    assert improved
    new_cost = compute_total_cost(state)
    assert new_cost < old_cost
    # Route should be rewired (not the original crossing order)
    assert state.route_nodes[0] != [0, 1, 3, 2, 4, 0]


def test_two_opt_cleanup_updates_cost():
    state = make_crossing_route_state()
    old_cost = compute_total_cost(state)
    improved = two_opt_cleanup(state)
    assert improved
    assert state.current_cost < old_cost
    assert state.cum_delta == 0.0
    assert state.C0 == state.current_cost
