from __future__ import annotations

from pathlib import Path

from analyzer.analyzer import analyze_run
from logger.logger_sqlite import Logger
from v5_solver.core.instance_adapter import load_instance
from v5_solver.core.config import V5Config
from v5_solver.core.orchestrator import V5Result


def make_inst(tmp_path: Path):
    vrp = tmp_path / "log.vrp"
    vrp.write_text(
        "NAME: log\n"
        "TYPE: CVRP\n"
        "DIMENSION: 3\n"
        "EDGE_WEIGHT_TYPE: EUC_2D\n"
        "CAPACITY: 100\n"
        "NODE_COORD_SECTION\n"
        "1 0 0\n"
        "2 0 10\n"
        "3 10 0\n"
        "DEMAND_SECTION\n"
        "1 0\n"
        "2 10\n"
        "3 10\n"
        "DEPOT_SECTION\n"
        "1\n"
        "-1\n"
        "EOF\n"
    )
    return load_instance(vrp)


def test_logger_analyzer_roundtrip(tmp_path: Path):
    inst = make_inst(tmp_path)
    # Simple one-route solution
    routes = [[2, 3]]
    cfg = V5Config()
    logger_path = tmp_path / "runs.sqlite"
    log = Logger(logger_path)
    run_id = log.start_run(inst.raw.name, cfg)
    # Fake stage and repo omitted; log solution directly
    log.log_solution(run_id, routes, cost=inst.route_cost(routes[0]))
    log.end_run(run_id)
    log.close()

    result = analyze_run(logger_path, tmp_path / "log.vrp", run_id)
    assert result["cost"] > 0
    assert result["routes"] == 1
