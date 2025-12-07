from __future__ import annotations

from pathlib import Path

from solver_phase5.runner import run_solver
from solver_phase5.config import SolverConfig


def make_tiny_vrp(tmp_path: Path) -> Path:
    content = """NAME : tiny5
TYPE : CVRP
DIMENSION : 5
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 10
NODE_COORD_SECTION
1 0 0
2 1 0
3 0 1
4 1 1
5 2 0
DEMAND_SECTION
1 0
2 2
3 3
4 4
5 2
DEPOT_SECTION
1
-1
EOF
"""
    vrp_path = tmp_path / "tiny5.vrp"
    vrp_path.write_text(content)
    return vrp_path


def test_solver_phase5_feasible(tmp_path):
    vrp_path = make_tiny_vrp(tmp_path)
    routes, cost = run_solver(str(vrp_path), cfg=SolverConfig(max_rounds_without_improve=2))
    seen = set()
    for r in routes:
        seen.update(n for n in r.nodes if n != 1)
        assert r.load <= 10
    assert seen == {2, 3, 4, 5}
    assert cost > 0
