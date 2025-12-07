from gen3.v1.core import CoreState
from gen3.v1.orchestrator import _pick_far_route


def build_state() -> CoreState:
    state = CoreState(
        N=3,
        Q=10,
        x=[0.0, 0.0, 100.0, -100.0],
        y=[0.0, 100.0, 0.0, 0.0],
        demand=[0, 1, 1, 1],
    )
    state.depot = 0
    state.route_nodes = [
        [0, 1, 0],  # north
        [0, 2, 0],  # east
        [0, 3, 0],  # west
    ]
    state.route_load = [1, 1, 1]
    state.route_mask = [1 << 1, 1 << 2, 1 << 3]
    state.route_active = [True, True, True]
    state.node_route = [None, 0, 1, 2]
    state.node_pos = [None, 1, 1, 1]
    return state


def test_pick_far_route_prefers_far_centroid():
    state = build_state()
    # From route 0 (north), farthest centroid should be route 2 (west) over route 1 (east),
    # since both are equidistant: tie can go either way but must not return the same route.
    choice = _pick_far_route(state, from_route=0)
    assert choice in {1, 2}
    assert choice != 0

    # If from_route is None, we simply pick the first active route.
    assert _pick_far_route(state, None) == 0
