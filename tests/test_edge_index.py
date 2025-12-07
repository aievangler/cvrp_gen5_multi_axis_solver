from gen3.v1.core import CoreState, build_edge_index
from gen3.v1.ls import apply_triad_move


def make_state_simple() -> CoreState:
    # Simple state: depot 0, customers 1 and 2 in route [0,1,2,0]
    state = CoreState(
        N=2,
        Q=10,
        depot=0,
        x=[0.0, 1.0, 2.0],
        y=[0.0, 0.0, 0.0],
        r=[0.0, 0.0, 0.0],
        theta=[0.0, 0.0, 0.0],
        demand=[0, 1, 1],
        route_nodes=[[0, 1, 2, 0]],
        route_load=[2],
        route_mask=[(1 << 1) | (1 << 2)],
        route_active=[True],
        node_route=[0, 0, 0],
        node_pos=[0, 1, 2],
        C0=0.0,
        cum_delta=0.0,
        current_cost=0.0,
    )
    state.edge_index = build_edge_index(state)
    return state


def test_build_edge_index_basic():
    state = make_state_simple()
    idx = state.edge_index
    assert idx is not None
    assert idx[(0, 1)] == (0, 0)
    assert idx[(1, 2)] == (0, 1)
    assert idx[(2, 0)] == (0, 2)


def test_apply_triad_move_rebuilds_edge_index():
    state = make_state_simple()
    # Move node 1 from (0,1,2) to between (2,0) -> route becomes [0,2,1,0]
    ok = apply_triad_move(state, j=1, p=0, s=2, i=2, k=0, rid_src=0, rid_tgt=0, delta_total=0)
    assert ok
    assert state.route_nodes[0] == [0, 2, 1, 0]
    idx = state.edge_index
    assert idx is not None
    assert (0, 1) not in idx
    assert idx[(0, 2)] == (0, 0)
    assert idx[(2, 1)] == (0, 1)
    assert idx[(1, 0)] == (0, 2)
