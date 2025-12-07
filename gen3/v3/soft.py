from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .core import CoreState
from .ls import run_ls, LSConfig


@dataclass
class SoftConfig:
    soft_run_enabled: bool = True
    soft_q_factor: float = 1.5
    soft_run_max_passes: int = 2
    max_candidates: int = 16


def soft_run(state: CoreState, cfg: Optional[SoftConfig] = None) -> bool:
    """
    Run LS with relaxed capacity (soft_Q) but still require delta<0 moves.
    """
    if cfg is None:
        cfg = SoftConfig()
    # Temporarily relax Q check inside run_ls by adjusting state.Q? Instead, we wrap.
    original_Q = state.Q
    soft_Q = int(cfg.soft_q_factor * original_Q)
    state.Q = soft_Q
    improved = run_ls(state, LSConfig(ls_max_passes=cfg.soft_run_max_passes, max_candidates=cfg.max_candidates))
    state.Q = original_Q
    return improved
