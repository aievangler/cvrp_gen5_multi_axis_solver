from __future__ import annotations

import pytest

from gen3.v3 import STNConfig, build_stn, read_vrp_dimacs


@pytest.mark.parametrize("vrp_file,distance_type", [
    ("tests/tmp_tiny.vrp", 1),
    ("tests/tmp_tiny_explicit_v3.vrp", 2),
])
def test_stn_symmetric_and_min(vrp_file: str, distance_type: int) -> None:
    state = read_vrp_dimacs(vrp_file, distance_type=distance_type)
    cfg = STNConfig(stn_k1=1, stn_k2=0, stn_k3=0, stn_min=1)
    build_stn(state, cfg)

    # Depot should have empty STN list.
    assert state.stn[0] == []

    # With two customers, each should see the other.
    assert state.stn[1] == [2]
    assert state.stn[2] == [1]

    # Symmetry: neighbors mirror each other.
    for i in range(1, state.N + 1):
        for j in state.stn[i]:
            assert i in state.stn[j]
