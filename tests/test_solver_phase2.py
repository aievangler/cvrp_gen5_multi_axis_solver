from __future__ import annotations

from pathlib import Path

from solver_phase2.runner import run_solver


def make_tiny_vrp(tmp_path: Path) -> Path:
    content = """NAME : tiny
TYPE : CVRP
DIMENSION : 4
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 10
NODE_COORD_SECTION
1 0 0
2 1 0
3 0 1
4 1 1
DEMAND_SECTION
1 0
2 2
3 3
4 4
DEPOT_SECTION
1
-1
EOF
"""
    vrp_path = tmp_path / "tiny2.vrp"
    vrp_path.write_text(content)
    return vrp_path


def test_solver_phase2_feasible(tmp_path):
    vrp_path = make_tiny_vrp(tmp_path)
    routes, cost = run_solver(str(vrp_path))
    seen = set()
    for r in routes:
        seen.update(n for n in r.nodes if n != 1)
        assert r.load <= 10
    assert seen == {2, 3, 4}
    assert cost > 0
