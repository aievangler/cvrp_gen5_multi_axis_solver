from gen3.v1.core import CoreState, build_edge_index, compute_total_cost
from gen3.v1.ls import (
    DDCPerturbConfig,
    apply_move_if_valid,
    compute_ddc_map,
    compute_eject_candidates,
    ddc_round_robin_perturb,
)
from gen3.v1.triad import TriadConfig


def build_simple_state() -> CoreState:
    # Geometry: depot at (0,0); customers at four cardinal points.
    state = CoreState(
        N=4,
        Q=100,
        x=[0.0, -10.0, 10.0, 0.0, 0.0],
        y=[0.0, 0.0, 0.0, 10.0, -10.0],
        demand=[0, 1, 1, 1, 1],
    )
    state.depot = 0
    state.route_nodes = [
        [0, 1, 2, 0],
        [0, 3, 4, 0],
    ]
    state.route_load = [2, 2]
    state.route_mask = [
        (1 << 1) | (1 << 2),
        (1 << 3) | (1 << 4),
    ]
    state.route_active = [True, True]
    state.node_route = [None, 0, 0, 1, 1]
    state.node_pos = [None, 1, 2, 1, 2]
    # Fully connected STN to allow all reloc candidates.
    stn = [[] for _ in range(state.N + 1)]
    for j in range(1, state.N + 1):
        stn[j] = [k for k in range(1, state.N + 1) if k != j]
    state.stn = stn
    state.edge_index = build_edge_index(state)
    state.C0 = compute_total_cost(state)
    state.current_cost = state.C0
    return state


def test_ddc_perturb_improves_cost():
    state = build_simple_state()
    base_cost = state.current_cost

    improved = ddc_round_robin_perturb(
        state,
        triad_cfg=TriadConfig(),
        cfg=DDCPerturbConfig(max_rounds=2, max_edges_per_route=5, max_neighbors_per_edge=10),
    )

    assert improved
    assert state.current_cost < base_cost
    active_nodes = set()
    for rid, route in enumerate(state.route_nodes):
        if state.route_active and rid < len(state.route_active) and not state.route_active[rid]:
            continue
        active_nodes.update(n for n in route if n != state.depot)
    assert active_nodes == {1, 2, 3, 4}


def test_apply_move_if_valid_rejects_uphill():
    state = build_simple_state()
    soft_Q = int(DDCPerturbConfig().soft_q_factor * state.Q)
    # Move node 1 within the same route between (0,2); delta_actual will be 0 -> reject.
    move = (1, 0, 0, state.depot, 2, 0)
    ok = apply_move_if_valid(state, move, soft_Q)
    assert not ok
    # Route remains unchanged
    assert state.route_nodes[0] == [0, 1, 2, 0]


def test_compute_eject_candidates_finds_move():
    state = build_simple_state()
    cfg = DDCPerturbConfig()
    ddc = compute_ddc_map(state)
    moves = compute_eject_candidates(
        state,
        ddc=ddc,
        soft_Q=int(cfg.soft_q_factor * state.Q),
        triad_cfg=TriadConfig(),
        cfg=cfg,
    )
    # Expect at least one route to have an improving eject move.
    assert moves


def test_ddc_round_robin_noop_single_route():
    state = CoreState(
        N=3,
        Q=10,
        x=[0.0, 1.0, 2.0, 3.0],
        y=[0.0, 0.0, 0.0, 0.0],
        demand=[0, 1, 1, 1],
    )
    state.depot = 0
    state.route_nodes = [[0, 1, 2, 3, 0]]
    state.route_load = [3]
    state.route_mask = [(1 << 1) | (1 << 2) | (1 << 3)]
    state.route_active = [True]
    state.node_route = [None, 0, 0, 0]
    state.node_pos = [None, 1, 2, 3]
    state.stn = [[], [2, 3], [1, 3], [1, 2]]
    state.edge_index = build_edge_index(state)
    state.C0 = compute_total_cost(state)
    state.current_cost = state.C0

    cfg = DDCPerturbConfig(max_rounds=2)
    improved = ddc_round_robin_perturb(state, TriadConfig(), cfg)
    assert not improved
    assert state.current_cost == state.C0
