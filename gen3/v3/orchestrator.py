from __future__ import annotations

import time
from typing import Callable, Optional

from .cleanup import CleanupConfig, enforce_strict_capacity
from .constructor import InitConfig, build_initial_solution
from .core import CoreState, compute_total_cost, restore_snapshot, snapshot_state
from .ls import LSConfig, run_ls
from .soft import SoftConfig, soft_run
from .two_opt import two_opt_cleanup
from .solution import validate_solution
from .stn import STNConfig, build_stn


def solve_with_state(
    state: CoreState,
    stn_config: Optional[STNConfig] = None,
    init_config: Optional[InitConfig] = None,
    soft_config: Optional[SoftConfig] = None,
    cleanup_config: Optional[CleanupConfig] = None,
    ls_config: Optional[LSConfig] = None,
    two_opt_enabled: bool = True,
    two_opt_max_iters_per_route: Optional[int] = None,
    max_plateau: int = 3,
    time_limit_sec: Optional[float] = None,
    on_improve: Optional[Callable[[CoreState], None]] = None,
) -> CoreState:
    """
    Run the solver given a pre-loaded CoreState.
    Builds STN and initial solution, then iterates soft_run -> cleanup -> LS.
    on_improve is called with the current state whenever best_cost improves.
    Returns the state restored to the best solution found.
    """
    stn_config = stn_config or STNConfig()
    init_config = init_config or InitConfig()
    soft_config = soft_config or SoftConfig()
    cleanup_config = cleanup_config or CleanupConfig()
    ls_config = ls_config or LSConfig()

    build_stn(state, stn_config)
    build_initial_solution(state, init_config)
    validate_solution(state)

    state.current_cost = compute_total_cost(state)
    state.C0 = state.current_cost
    state.cum_delta = 0.0

    best_cost = state.current_cost
    best_solution = snapshot_state(state)
    best_feasible = snapshot_state(state)
    run_best_cost = float("inf")
    plateau = 0
    start = time.monotonic()

    # Emit initial solution if desired
    if on_improve is not None:
        on_improve(state)
        run_best_cost = state.current_cost

    while plateau < max_plateau:
        if time_limit_sec is not None and (time.monotonic() - start) >= time_limit_sec:
            break

        if soft_config.soft_run_enabled:
            soft_run(state, soft_config)

        ok = enforce_strict_capacity(state, cleanup_config)
        if not ok:
            # Roll back to last feasible, polish once with strict LS + 2-opt, then stop.
            restore_snapshot(state, best_feasible)
            run_ls(state, ls_config)
            if two_opt_enabled:
                two_opt_cleanup(state, max_iters_per_route=two_opt_max_iters_per_route)
            validate_solution(state)
            cur = state.current_cost
            if cur < best_cost - 1e-9:
                best_cost = cur
                best_solution = snapshot_state(state)
                best_feasible = snapshot_state(state)
                plateau = 0
                if cur < run_best_cost - 1e-9 and on_improve is not None:
                    on_improve(state)
                    run_best_cost = cur
            break

        # cleanup succeeded
        run_ls(state, ls_config)
        if two_opt_enabled:
            two_opt_cleanup(state, max_iters_per_route=two_opt_max_iters_per_route)
        validate_solution(state)
        cur = state.current_cost
        if cur < best_cost - 1e-9:
            best_cost = cur
            best_solution = snapshot_state(state)
            best_feasible = snapshot_state(state)
            plateau = 0
            if cur < run_best_cost - 1e-9 and on_improve is not None:
                on_improve(state)
                run_best_cost = cur
        else:
            plateau += 1

    # Restore best solution before returning
    restore_snapshot(state, best_solution)
    validate_solution(state)
    return state


def solve(
    instance_path: str,
    distance_type: int,
    stn_config: Optional[STNConfig] = None,
    init_config: Optional[InitConfig] = None,
    soft_config: Optional[SoftConfig] = None,
    cleanup_config: Optional[CleanupConfig] = None,
    ls_config: Optional[LSConfig] = None,
    max_plateau: int = 3,
    time_limit_sec: Optional[float] = None,
    on_improve: Optional[Callable[[CoreState], None]] = None,
) -> CoreState:
    """Convenience wrapper that loads an instance then solves."""
    from .io import read_vrp_dimacs

    state = read_vrp_dimacs(instance_path, distance_type=distance_type)
    return solve_with_state(
        state=state,
        stn_config=stn_config,
        init_config=init_config,
        soft_config=soft_config,
        cleanup_config=cleanup_config,
        ls_config=ls_config,
        max_plateau=max_plateau,
        time_limit_sec=time_limit_sec,
        on_improve=on_improve,
    )
