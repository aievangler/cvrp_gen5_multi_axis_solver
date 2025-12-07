from __future__ import annotations

from pathlib import Path

from v5_solver.core.instance_adapter import load_instance
from v5_solver.core.macro_solver import SolutionRoute, macro_local_search, construct_from_repo
from v5_solver.core.config import V5Config
from v5_solver.core.repo_builder import Repo, RepoRoute


def make_inst(tmp_path: Path):
    vrp = tmp_path / "macro.vrp"
    vrp.write_text(
        "NAME: macro\n"
        "TYPE: CVRP\n"
        "DIMENSION: 4\n"
        "EDGE_WEIGHT_TYPE: EUC_2D\n"
        "CAPACITY: 100\n"
        "NODE_COORD_SECTION\n"
        "1 0 0\n"
        "2 0 10\n"
        "3 10 0\n"
        "4 10 10\n"
        "DEMAND_SECTION\n"
        "1 0\n"
        "2 10\n"
        "3 10\n"
        "4 10\n"
        "DEPOT_SECTION\n"
        "1\n"
        "-1\n"
        "EOF\n"
    )
    return load_instance(vrp)


def test_macro_local_search_replaces_cheaper(tmp_path: Path):
    inst = make_inst(tmp_path)
    # Current route covering nodes 2,3
    route = SolutionRoute(seq=[2, 3], load=20, cost=50, bitset=(1 << 2) | (1 << 3))
    # Repo macro with same node set but cheaper cost
    rr = RepoRoute(
        seq=[3, 2],
        load=20,
        cost=30,
        bitset=(1 << 2) | (1 << 3),
        prefix_load=[0, 10, 20],
        prefix_cost=[0, 10, 20],
        bucket_key=(0, 0, "50"),
        node_sig=(2, 3),
        fill_bucket="50",
    )
    repo = Repo(routes=[rr], index={rr.bucket_key: [rr]})
    improved = macro_local_search([route], repo, inst, V5Config())
    assert improved[0].cost == 30
    assert improved[0].seq == [3, 2]


def test_construct_from_repo_covers_all_nodes(tmp_path: Path):
    inst = make_inst(tmp_path)
    # Repo covers one node; fallback should cover the rest
    rr = RepoRoute(
        seq=[2],
        load=10,
        cost=inst.distance(1, 2) * 2,
        bitset=(1 << 2),
        prefix_load=[0, 10],
        prefix_cost=[0, inst.distance(1, 2), inst.distance(2, 1)],
        bucket_key=(0, 0, "50"),
        node_sig=(2,),
        fill_bucket="50",
    )
    repo = Repo(routes=[rr], index={rr.bucket_key: [rr]})
    from v5_solver.core.macro_solver import solve_with_repo

    routes = solve_with_repo(inst, repo, V5Config())
    all_nodes = {n for rt in routes for n in rt.seq}
    assert all_nodes >= {2, 3, 4}
