import pytest

from gen3.v1 import (
    STNConfig,
    analyze,
    build_stn,
    edge_overlap_ratio,
    read_vrp,
    stn_coverage,
    triad_ddc,
)


def _vrp_analysis(tmp_path):
    vrp = tmp_path / "analysis.vrp"
    lines = [
        "NAME : analysis",
        "TYPE : CVRP",
        "DIMENSION : 4",
        "CAPACITY : 10",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
        "1 0 0",
        "2 0 3",
        "3 4 0",
        "4 4 3",
        "DEMAND_SECTION",
        "1 0",
        "2 1",
        "3 1",
        "4 1",
        "DEPOT_SECTION",
        "1",
        "-1",
        "EOF",
    ]
    vrp.write_text("\n".join(lines) + "\n")
    return vrp


def test_stn_coverage_and_ddc(tmp_path):
    state = read_vrp(_vrp_analysis(tmp_path))
    build_stn(state, STNConfig(stn_k1=3, stn_k2=0, stn_k3=0, stn_min=3))
    routes = [[0, 1, 2, 0], [0, 3, 0]]
    cov = stn_coverage(state, routes)
    assert cov.triads_total == 3
    assert cov.triads_covered == 3
    assert cov.coverage_pct == 100.0
    ddc = triad_ddc(state, routes)
    # Triads: (0,1,2), (1,2,0), (0,3,0)
    assert len(ddc) == 3
    assert all(isinstance(x, int) for x in ddc)


def test_edge_overlap_ratio(tmp_path):
    state = read_vrp(_vrp_analysis(tmp_path))
    build_stn(state, STNConfig(stn_k1=3, stn_k2=0, stn_k3=0, stn_min=3))
    bks_routes = [[0, 1, 2, 0]]
    solver_routes = [[0, 1, 3, 2, 0]]
    overlap = edge_overlap_ratio(state, bks_routes, solver_routes)
    shared, total, ratio = overlap[0]
    assert shared == 2  # edges (0,1) and (2,0)
    assert total == 3
    assert pytest.approx(ratio) == 2 / 3


def test_analyze_end_to_end(tmp_path):
    vrp = _vrp_analysis(tmp_path)
    bks_sol = tmp_path / "bks.sol"
    solver_sol = tmp_path / "solver.sol"
    bks_sol.write_text("Route #1: 2 3\nCost 0\n")
    solver_sol.write_text("Route #1: 2 4 3\nCost 0\n")
    result = analyze(str(vrp), str(bks_sol), str(solver_sol), STNConfig(stn_k1=3, stn_k2=0, stn_k3=0, stn_min=3))
    assert "coverage" in result and "ddc_bks" in result and "edge_overlap" in result
    cov = result["coverage"]
    assert cov.triads_total >= cov.triads_covered >= 0
