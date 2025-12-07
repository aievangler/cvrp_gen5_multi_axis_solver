import pytest

from gen3.v1 import STNConfig, build_stn, read_vrp


def _write_simple_vrp(tmp_path):
    vrp = tmp_path / "grid.vrp"
    lines = [
        "NAME : grid",
        "TYPE : CVRP",
        "DIMENSION : 5",
        "CAPACITY : 10",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        "1 0 0",
        "2 10 0",
        "3 20 0",
        "4 0 10",
        "5 0 20",
        "DEMAND_SECTION",
        "1 0",
        "2 1",
        "3 1",
        "4 1",
        "5 1",
        "DEPOT_SECTION",
        "1",
        "-1",
        "EOF",
    ]
    vrp.write_text("\n".join(lines) + "\n")
    return vrp


def test_stn_symmetric_and_depot_excluded(tmp_path):
    state = read_vrp(_write_simple_vrp(tmp_path))
    cfg = STNConfig(stn_k1=1, stn_k2=1, stn_k3=0, stn_min=2)
    build_stn(state, cfg)
    for i in range(1, state.N + 1):
        for j in state.stn[i]:
            assert i in state.stn[j]
            assert j != state.depot
        assert state.depot not in state.stn[i]


def test_stn_min_size_enforced(tmp_path):
    state = read_vrp(_write_simple_vrp(tmp_path))
    cfg = STNConfig(stn_k1=1, stn_k2=0, stn_k3=0, stn_min=3)
    build_stn(state, cfg)
    for i in range(1, state.N + 1):
        # With 4 customers total, max neighbors is 3
        assert len(state.stn[i]) == 3
        assert len(set(state.stn[i])) == len(state.stn[i])


def test_stn_sorted_and_deduped(tmp_path):
    state = read_vrp(_write_simple_vrp(tmp_path))
    cfg = STNConfig(stn_k1=10, stn_k2=10, stn_k3=10, stn_min=2)
    build_stn(state, cfg)
    for neigh in state.stn[1:]:
        assert neigh == sorted(neigh)
        assert len(neigh) == len(set(neigh))
