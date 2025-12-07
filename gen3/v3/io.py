from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

from .core import CoreState


def _parse_header_value(line: str) -> Tuple[str, str]:
    if ":" in line:
        key, val = line.split(":", 1)
        return key.strip().upper(), val.strip()
    parts = line.split()
    return (parts[0].strip().upper(), parts[1].strip() if len(parts) > 1 else "")


def _parse_explicit_matrix(lines: List[str], dimension: int, fmt: str) -> List[int]:
    """
    Parse EDGE_WEIGHT_SECTION lines into a full symmetric matrix (row-major),
    returned as a flat list of length dimension*dimension in TS indices (1-based in file).
    """
    nums: List[int] = []
    for ln in lines:
        for tok in ln.split():
            try:
                nums.append(int(tok))
            except ValueError:
                continue
    mat_ts = [[0] * dimension for _ in range(dimension)]
    fmt = fmt.upper()
    if fmt == "FULL_MATRIX":
        if len(nums) < dimension * dimension:
            raise ValueError("EDGE_WEIGHT_SECTION shorter than expected for FULL_MATRIX")
        it = iter(nums)
        for i in range(dimension):
            for j in range(dimension):
                mat_ts[i][j] = next(it)
    else:
        idx = 0
        if fmt in ("LOWER_ROW", "LOWER_DIAG_ROW"):
            include_diag = fmt == "LOWER_DIAG_ROW"
            for i in range(dimension):
                j_limit = i + 1 if include_diag else i
                for j in range(j_limit):
                    if idx >= len(nums):
                        raise ValueError("EDGE_WEIGHT_SECTION shorter than expected")
                    w = nums[idx]
                    idx += 1
                    mat_ts[i][j] = w
                    mat_ts[j][i] = w
        elif fmt in ("UPPER_ROW", "UPPER_DIAG_ROW"):
            include_diag = fmt == "UPPER_DIAG_ROW"
            for i in range(dimension):
                start = 0 if include_diag else i + 1
                for j in range(start, dimension):
                    if j <= i and not include_diag:
                        continue
                    if idx >= len(nums):
                        raise ValueError("EDGE_WEIGHT_SECTION shorter than expected")
                    w = nums[idx]
                    idx += 1
                    mat_ts[i][j] = w
                    mat_ts[j][i] = w
        else:
            raise ValueError(f"Unsupported EDGE_WEIGHT_FORMAT {fmt}")
    # flatten row-major
    flat: List[int] = []
    for i in range(dimension):
        for j in range(dimension):
            flat.append(mat_ts[i][j])
    return flat


def read_vrp_dimacs(path: str | Path, distance_type: int) -> CoreState:
    """
    Read a TSPLIB95/CVRP instance with DIMACS-compatible mapping.
    Internal indices: depot=0, customers=1..N, where N = DIMENSION-1.
    """
    path = Path(path)
    lines = [ln.rstrip() for ln in path.read_text().splitlines() if ln.strip()]

    dimension: int | None = None
    capacity: int | None = None
    coords: dict[int, Tuple[float, float]] = {}
    demands: dict[int, int] = {}
    depot_ts: int | None = None
    edge_weight_type: str | None = None
    edge_weight_format: str | None = None
    explicit_lines: List[str] = []

    section = None
    for raw in lines:
        upper = raw.upper()
        if upper.startswith("NODE_COORD_SECTION"):
            section = "COORD"
            continue
        if upper.startswith("DEMAND_SECTION"):
            section = "DEMAND"
            continue
        if upper.startswith("DEPOT_SECTION"):
            section = "DEPOT"
            continue
        if upper.startswith("EDGE_WEIGHT_SECTION"):
            section = "EDGE_WEIGHT"
            continue
        if upper.startswith("EOF"):
            break

        if section is None:
            key, val = _parse_header_value(raw)
            if key == "DIMENSION":
                dimension = int(val)
            elif key == "CAPACITY":
                capacity = int(val)
            elif key == "EDGE_WEIGHT_TYPE":
                edge_weight_type = val.upper()
            elif key == "EDGE_WEIGHT_FORMAT":
                edge_weight_format = val.upper()
            continue

        if section == "COORD":
            parts = raw.split()
            if len(parts) < 3:
                continue
            ts_id = int(parts[0])
            coords[ts_id] = (float(parts[1]), float(parts[2]))
        elif section == "DEMAND":
            parts = raw.split()
            if len(parts) < 2:
                continue
            ts_id = int(parts[0])
            demands[ts_id] = int(parts[1])
        elif section == "DEPOT":
            val = raw.split()[0]
            dep = int(val)
            if dep != -1 and depot_ts is None:
                depot_ts = dep
        elif section == "EDGE_WEIGHT":
            explicit_lines.append(raw)

    if dimension is None or capacity is None or depot_ts is None:
        raise ValueError("VRP file missing DIMENSION, CAPACITY, or DEPOT_SECTION")
    if depot_ts != 1:
        raise ValueError(f"Depot id must be 1, got {depot_ts}")

    size = dimension  # includes depot
    N = size - 1
    x = [0.0] * size
    y = [0.0] * size
    demand = [0] * size

    for ts_id, (cx, cy) in coords.items():
        internal = ts_id - 1
        if 0 <= internal < size:
            x[internal] = cx
            y[internal] = cy
    for ts_id, q in demands.items():
        internal = ts_id - 1
        if 0 <= internal < size:
            demand[internal] = q
    demand[0] = 0  # depot demand zero

    dist_matrix = None
    if edge_weight_type and edge_weight_type.startswith("EXPLICIT"):
        fmt = edge_weight_format or "LOWER_ROW"
        mat_ts = _parse_explicit_matrix(explicit_lines, size, fmt)
        # map ts indices (1-based) to internal (0-based)
        dist_matrix = [0] * (size * size)
        idx = 0
        for i in range(size):
            for j in range(size):
                dist_matrix[idx] = mat_ts[i * size + j]
                idx += 1
        distance_type = 2  # force explicit

    state = CoreState(
        N=N,
        Q=capacity,
        distance_type=distance_type,
        x=x,
        y=y,
        demand=demand,
        dist_matrix=dist_matrix,
    )
    # Initialize empty solution arrays sized to at least 1 route to allow tests to set routes later.
    state.route_nodes = []
    state.route_load = []
    state.route_mask = []
    state.route_active = []
    state.node_route = [-1] * size
    state.node_pos = [-1] * size
    return state
