from __future__ import annotations

import math
from pathlib import Path

import pytest

from v5_solver.core.config import V5Config
from v5_solver.core.instance_adapter import V5Instance, load_instance
from v5_solver.core.repo_builder import build_repo, RepoRoute
from v5_solver.core.v2_wrapper import StageOut, RouteLite


def make_instance(tmp_path: Path) -> V5Instance:
    # Tiny synthetic EUC_2D instance with 1 depot + 4 customers
    vrp = tmp_path / "tiny.vrp"
    vrp.write_text(
        "NAME: tiny\n"
        "TYPE: CVRP\n"
        "DIMENSION: 5\n"
        "EDGE_WEIGHT_TYPE: EUC_2D\n"
        "CAPACITY: 100\n"
        "NODE_COORD_SECTION\n"
        "1 0 0\n"
        "2 0 10\n"
        "3 10 10\n"
        "4 10 0\n"
        "5 5 5\n"
        "DEMAND_SECTION\n"
        "1 0\n"
        "2 30\n"
        "3 30\n"
        "4 20\n"
        "5 20\n"
        "DEPOT_SECTION\n"
        "1\n"
        "-1\n"
        "EOF\n"
    )
    return load_instance(vrp)


def test_repo_extraction_fill_buckets(tmp_path: Path):
    inst = make_instance(tmp_path)
    # One route with load 100 -> should produce 50/70/100 candidates
    route = RouteLite(seq=[2, 3, 4, 5], cost=0.0, load=100.0)
    stage = StageOut(stage_name="test", routes=[route])
    cfg = V5Config(repo_min_seg_len=2, repo_max_seg_len=4, repo_max_routes=20, repo_k_per_bucket=5)
    repo = build_repo(inst, [stage], cfg)
    fills = {rr.fill_bucket for rr in repo.routes}
    assert {"50", "70", "100"} & fills  # at least some candidates from buckets
    # No duplicate node_sig within bucket
    sigs_by_bucket = {}
    for rr in repo.routes:
        sigs = sigs_by_bucket.setdefault(rr.bucket_key, set())
        assert rr.node_sig not in sigs
        sigs.add(rr.node_sig)


def test_repo_prunes_near_duplicates(tmp_path: Path):
    inst = make_instance(tmp_path)
    # Two routes with same node set but different cost
    r1 = RouteLite(seq=[2, 3, 4], cost=100.0, load=80.0)
    r2 = RouteLite(seq=[4, 3, 2], cost=50.0, load=80.0)
    stage = StageOut(stage_name="test", routes=[r1, r2])
    cfg = V5Config(repo_min_seg_len=2, repo_max_seg_len=4, repo_max_routes=10, repo_k_per_bucket=2)
    repo = build_repo(inst, [stage], cfg)
    # Keep the cheaper one
    node_sigs = [rr.node_sig for rr in repo.routes]
    assert len(node_sigs) == len(set(node_sigs))  # deduped
    kept = repo.routes[0]
    assert kept.cost <= min(r1.cost, r2.cost)
