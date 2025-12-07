from pathlib import Path

from moverx.solver import run_moverx
from solver_phase7.parser import parse_vrp


def test_moverx_small_instance_feasible():
    vrp_path = Path("data/instances/A/A-n32-k5.vrp")
    coords, demand, Q, depot = parse_vrp(vrp_path)
    routes, cost = run_moverx(str(vrp_path), neighbor_k=16, max_iters=500)
    # basic feasibility: all customers covered and loads within Q
    customers = [n for n in coords if n != depot]
    seen = set()
    for R in routes:
        seen.update(R.nodes[1:-1])
        assert R.load <= Q
    assert set(customers) == seen
    assert cost > 0
