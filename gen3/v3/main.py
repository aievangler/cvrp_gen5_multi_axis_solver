from __future__ import annotations

import sys
import time

from .orchestrator import solve_with_state
from .solution import emit_solution_for_controller, validate_solution
from .io import read_vrp_dimacs
from .stn import STNConfig
from .constructor import InitConfig
from .soft import SoftConfig
from .cleanup import CleanupConfig
from .ls import LSConfig
from .two_opt import two_opt_cleanup


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    if len(argv) != 3:
        print("Usage: Solver instance.vrp distance_type time_limit", file=sys.stderr)
        return 1
    instance_path = argv[0]
    try:
        distance_type = int(argv[1])
    except ValueError:
        print("distance_type must be 0, 1, or 2", file=sys.stderr)
        return 1
    try:
        time_limit = float(argv[2])
    except ValueError:
        print("time_limit must be a number (seconds)", file=sys.stderr)
        return 1

    state = read_vrp_dimacs(instance_path, distance_type=distance_type)

    def on_improve(s):
        # Validate before emitting to ensure Controller gets feasible solutions
        validate_solution(s)
        emit_solution_for_controller(s)

    start = time.monotonic()
    solve_with_state(
        state=state,
        stn_config=STNConfig(),
        init_config=InitConfig(),
        soft_config=SoftConfig(),
        cleanup_config=CleanupConfig(),
        ls_config=LSConfig(),
        max_plateau=3,
        time_limit_sec=time_limit,
        on_improve=on_improve,
    )
    # Optionally emit best solution at end if none were emitted (unlikely)
    if time.monotonic() - start < time_limit:
        validate_solution(state)
        emit_solution_for_controller(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
