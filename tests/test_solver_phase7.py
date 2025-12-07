from pathlib import Path

from solver_phase7.config import SolverConfig
from solver_phase7.runner import run_solver
from solver_phase7.validator import validate_solution
from solver_phase7.parser import parse_vrp


def test_phase7_small_instance_feasible():
    vrp_path = Path("data/instances/A/A-n32-k5.vrp")
    coords, demand, Q, depot = parse_vrp(vrp_path)
    cfg = SolverConfig(
        soft_moves=2000,
        max_ls_iters=500,
        max_capacity_iters=200,
        max_rounds_without_improve=1,
        neighbor_k=16,
        enable_swaps=False,
        ranking_iters=5,
    )
    routes, cost = run_solver(str(vrp_path), cfg)
    # ensure solver returns some routes and they are feasible
    customers = [n for n in coords if n != depot]
    validate_solution(routes, demand, Q, depot, customers)
    assert cost > 0
