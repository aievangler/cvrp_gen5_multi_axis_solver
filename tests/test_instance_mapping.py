from __future__ import annotations

from pathlib import Path

from v5_solver.core.instance_adapter import build_v2_preprocessed, load_instance, route_internal_to_raw


def test_internal_to_raw_mapping(tmp_path: Path):
    vrp = tmp_path / "map.vrp"
    vrp.write_text(
        "NAME: map\n"
        "TYPE: CVRP\n"
        "DIMENSION: 4\n"
        "EDGE_WEIGHT_TYPE: EUC_2D\n"
        "CAPACITY: 100\n"
        "NODE_COORD_SECTION\n"
        "2 0 0\n"  # depot is node 2
        "1 0 10\n"
        "3 10 0\n"
        "4 10 10\n"
        "DEMAND_SECTION\n"
        "1 10\n"
        "2 0\n"
        "3 10\n"
        "4 10\n"
        "DEPOT_SECTION\n"
        "2\n"
        "-1\n"
        "EOF\n"
    )
    inst = load_instance(vrp)
    pre, geom, qc, cand = build_v2_preprocessed(inst)
    # Internal index 0 == depot raw 2; customers mapped to original ids
    assert int(pre.base.original_ids[0]) == 2
    # Build a fake route with internal ids and map back
    internal_route = [0, 1, 2, 3, 0]  # depot + three customers
    mapped = route_internal_to_raw(internal_route, pre.base.original_ids)
    assert mapped == [1, 3, 4]
