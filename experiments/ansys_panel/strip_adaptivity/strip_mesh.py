"""Deterministic conforming tensor-product strips; no MAPDL or solve imports.

Every inserted line spans the panel. Marking is local, but its mesh effect is
global; the resulting cell/DoF cost includes all requested and closure lines.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


THETA = .5
ASPECT_LIMIT = 4.
DOF_CAP = 5382
MAX_ADAPTATION_STEPS = 3


def _axis(values, extent, name):
    axis = np.asarray(values, dtype=float).copy()
    if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
        raise ValueError(f"{name} must be a finite one-dimensional coordinate array")
    if not np.all(np.diff(axis) > 0):
        raise ValueError(f"{name} coordinates must be strictly increasing")
    if axis[0] != 0. or axis[-1] != float(extent):
        raise ValueError(f"{name} must cover exactly [0, {extent}]")
    return axis


def _ids(values, name):
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must be a one-dimensional integer sequence")
    if not np.isfinite(raw).all() or np.any(raw != np.floor(raw)) or np.any(raw <= 0):
        raise ValueError(f"{name} must contain positive integer IDs")
    ids = raw.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name} contains duplicate IDs")
    return ids


def _scores(values):
    scores = np.asarray(values, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("Scores must be a finite nonnegative one-dimensional vector")
    return scores.copy()


def classical_selector(observation):
    """Return eta_K squared aligned with observation element_ids, unmodified."""
    scores = _scores(observation["eta2"])
    if "element_ids" in observation and len(scores) != len(_ids(observation["element_ids"], "element_ids")):
        raise ValueError("eta2 and element_ids must have equal lengths")
    return scores


def dorfler_mark(element_ids, eta2, theta=THETA):
    """Minimal descending eta² prefix, exact-score ties resolved by element ID.

    A zero total is a reported failure; no arbitrary fallback marking is used.
    The experiment has a frozen theta=.5 and rejects another theta.
    """
    if float(theta) != THETA:
        raise ValueError("This experiment fixes Dorfler theta at 0.5")
    ids = _ids(element_ids, "element_ids")
    scores = _scores(eta2)
    if len(ids) != len(scores):
        raise ValueError("eta2 and element_ids must have equal lengths")
    try:
        total = math.fsum(map(float, scores))
    except OverflowError as error:
        raise ValueError("Dorfler requires a finite positive total eta2") from error
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Dorfler requires a finite positive total eta2; zero indicator cannot mark")
    order = np.lexsort((ids, -scores))
    target = THETA*total
    selected_count = 0
    for selected_count in range(1, len(order)+1):
        selected_sum = math.fsum(float(scores[i]) for i in order[:selected_count])
        if selected_sum >= target:
            break
    selected = ids[order[:selected_count]].tolist()
    selected_sum = math.fsum(float(scores[i]) for i in order[:selected_count])
    return selected, {"theta": THETA, "total_eta2": total, "target_eta2": target,
                      "selected_eta2": selected_sum, "selected_fraction": selected_sum/total,
                      "marked_element_ids": selected, "marked_count": selected_count,
                      "ordered_element_ids": ids[order].tolist(),
                      "ordered_eta2": scores[order].tolist(),
                      "tie_rule": "descending eta2, then ascending element ID"}


doerfler_mark = dorfler_mark


def counts(x, y):
    nx, ny = len(x)-1, len(y)-1
    return {"nx": nx, "ny": ny, "elements": nx*ny, "nodes": (nx+1)*(ny+1),
            "boundary_nodes": 2*(nx+ny), "expected_free_dof_topology": 6*(nx-1)*(ny-1),
            "solver_equation_count": None}


def maximum_aspect(x, y):
    dx, dy = np.diff(x), np.diff(y)
    return float(max(dx.max()/dy.min(), dy.max()/dx.min()))


def build_mesh(x, y, config):
    """Build CCW +Z rectangles; native_settings here are requested settings.

    A native settings/attribute query after model construction is still required.
    Predicted free DoF assume all six SHELL181 DoF fixed at every boundary node.
    """
    x, y = _axis(x, config["a"], "X"), _axis(y, config["b"], "Y")
    thickness = float(config.get("thickness", 8.))
    if not math.isfinite(thickness) or thickness <= 0:
        raise ValueError("Shell thickness must be finite and positive")
    nx, ny = len(x)-1, len(y)-1
    coordinates = [[float(xx), float(yy), 0.] for yy in y for xx in x]
    conn = []
    for j in range(ny):
        for i in range(nx):
            n = j*(nx+1)+i+1
            conn.append([n, n+1, n+nx+2, n+nx+1])
    cells = nx*ny
    settings = {"element_name": 181,
                "keyopts": {str(k): int(v) for k,v in config.get("keyopts", {"1":0,"3":2,"8":2}).items()},
                "thickness_mm": thickness, "layer_thickness_mm": thickness,
                "layer_material": 1, "layer_angle_degrees": 0.,
                "through_thickness_points": int(config.get("thickness_integration_points", 5)), "offset": "MID"}
    return {"node_ids": list(range(1, len(coordinates)+1)), "xyz_mm": coordinates,
            "element_ids": list(range(1, cells+1)), "connectivity_node_ids": conn,
            "attributes": {"TYPE": [1]*cells, "MAT": [1]*cells, "SECNUM": [1]*cells,
                           "THICKNESS": [thickness]*cells},
            "native_settings": settings,
            "native_settings_provenance": "Requested settings only; native query required before acceptance",
            "x_lines_mm": x.tolist(), "y_lines_mm": y.tolist(),
            "topology_prediction": counts(x,y),
            "dof_note": "6*(nx-1)*(ny-1) topology prediction for P1; no solver equation count"}


def _split_record(axis, index, coordinates):
    left, right = float(coordinates[index]), float(coordinates[index+1])
    midpoint = left+(right-left)/2
    if midpoint <= left or midpoint >= right:
        raise ValueError("Floating-point coordinates cannot represent requested interval bisection")
    return {"axis": axis, "interval_index_zero_based": int(index), "left_mm": left,
            "right_mm": right, "length_mm": right-left, "midpoint_mm": midpoint}


def remesh(x, y, marked_ids, config):
    """Propose all frozen-policy insertions, then report the full DoF budget.

    Marks are never dropped to fit the cap. Returned arrays describe the entire
    proposal even when rejected_before_solve=True. Callers must enforce the flag.
    """
    x, y = _axis(x, config["a"], "X"), _axis(y, config["b"], "Y")
    marked = _ids(marked_ids, "marked_ids")
    nx, ny = len(x)-1, len(y)-1
    if not len(marked) or np.any(marked > nx*ny):
        raise ValueError("Marked IDs must be a nonempty subset of current row-major element IDs")
    requested = {"X": defaultdict(list), "Y": defaultdict(list)}
    choices = []
    dx, dy = np.diff(x), np.diff(y)
    for eid in sorted(marked.tolist()):
        j, i = divmod(eid-1, nx)
        axis, index = ("X", i) if dx[i] >= dy[j] else ("Y", j)
        requested[axis][index].append(eid)
        choices.append({"element_id": eid, "x_interval_zero_based": i, "y_interval_zero_based": j,
                        "dx_mm": float(dx[i]), "dy_mm": float(dy[j]),
                        "requested_axis": axis, "requested_interval_zero_based": index})
    request_records = []
    for axis, coordinates in (("X", x), ("Y", y)):
        for index in sorted(requested[axis]):
            record = _split_record(axis, index, coordinates)
            record["source_marked_element_ids"] = requested[axis][index]
            request_records.append(record)
    new_x = np.sort(np.concatenate((x, [r["midpoint_mm"] for r in request_records if r["axis"] == "X"])))
    new_y = np.sort(np.concatenate((y, [r["midpoint_mm"] for r in request_records if r["axis"] == "Y"])))
    closure = []
    while True:
        widths, heights = np.diff(new_x), np.diff(new_y)
        candidates = [(float(-width), 0, i) for i,width in enumerate(widths) if width > ASPECT_LIMIT*heights.min()]
        candidates += [(float(-height), 1, j) for j,height in enumerate(heights) if height > ASPECT_LIMIT*widths.min()]
        if not candidates:
            break
        _, axis_code, index = min(candidates)
        axis, coordinates = ("X", new_x) if axis_code == 0 else ("Y", new_y)
        record = _split_record(axis, index, coordinates)
        record["maximum_aspect_before"] = maximum_aspect(new_x, new_y)
        closure.append(record)
        if axis_code == 0:
            new_x = np.insert(new_x, index+1, record["midpoint_mm"])
        else:
            new_y = np.insert(new_y, index+1, record["midpoint_mm"])
    before, after = counts(x,y), counts(new_x,new_y)
    within = after["expected_free_dof_topology"] <= DOF_CAP
    metadata = {"marked_element_ids": sorted(marked.tolist()), "cell_split_choices": choices,
                "mark_to_interval_rule": "split longer original physical interval; equal lengths choose X",
                "requested_interval_splits": request_records, "closure_splits": closure,
                "closure_rule": "largest offending current interval; equal lengths choose X then lowest index",
                "requested_new_line_count": len(request_records), "closure_new_line_count": len(closure),
                "total_new_line_count": len(request_records)+len(closure),
                "before": before, "after": after, "maximum_aspect_before": maximum_aspect(x,y),
                "maximum_aspect_after": maximum_aspect(new_x,new_y), "aspect_limit": ASPECT_LIMIT,
                "dof_cap": DOF_CAP, "within_dof_budget": within, "rejected_before_solve": not within,
                "solve_performed": False, "dof_note": "Topology prediction only; no solver equation count",
                "x_lines_mm": new_x.tolist(), "y_lines_mm": new_y.tolist()}
    return new_x, new_y, metadata
