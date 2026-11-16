"""One fixed, bounded minimax coordinate repair for saved planar Q4 meshes.

This module imports no ANSYS interface. Connectivity, IDs, attributes and Z are
immutable. The independent validator and subsequent native checks remain gates;
SLSQP's success flag alone is never mesh acceptance.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import time

import numpy as np
from scipy.optimize import minimize

from validate_mesh import validate_mesh


MAXITER = 200
FTOL = 1e-9
FINITE_DIFFERENCE_EPS = 1e-7
FEASIBILITY_TOLERANCE = 1e-7
MINIMUM_OBJECTIVE_IMPROVEMENT = 1e-8
DISPLACEMENT_FRACTION = .20
NEIGHBOR_RINGS = 2
GAUSS = 1 / np.sqrt(3.)
SAMPLES = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1],
                    [-GAUSS, -GAUSS], [GAUSS, -GAUSS],
                    [GAUSS, GAUSS], [-GAUSS, GAUSS]])
DERIVATIVES = np.array([
    [[-(1-y), -(1-x)], [(1-y), -(1+x)], [(1+y), (1+x)], [-(1+y), (1-x)]]
    for x, y in SAMPLES]) / 4


def _cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def geometry(xy, indices):
    """Exactly the original eight-point planar Q4 condition-number metric."""
    points = xy[indices]
    jac = np.einsum("eni,snj->esij", points, DERIVATIVES)
    determinant = np.linalg.det(jac)
    singular = np.linalg.svd(jac, compute_uv=False)
    condition = singular[..., 0] / np.maximum(singular[..., 1], 1e-30)
    scale = determinant / np.maximum(np.linalg.norm(jac[..., :, 0], axis=-1) *
                                      np.linalg.norm(jac[..., :, 1], axis=-1), 1e-30)
    prev = np.roll(points, 1, axis=1) - points
    following = np.roll(points, -1, axis=1) - points
    lengths = np.linalg.norm(following, axis=-1)
    cosine = np.sum(prev*following, axis=-1) / np.maximum(
        np.linalg.norm(prev, axis=-1)*lengths, 1e-30)
    area = np.sum(_cross(points, np.roll(points, -1, axis=1)), axis=1)/2
    return {"determinant": determinant, "condition": condition,
            "scaled_jacobian": scale, "cosine_angles": cosine,
            "area": area, "edge_lengths": lengths}


def _indices(mesh):
    nodes = np.asarray(mesh["node_ids"], dtype=np.int64)
    xyz = np.asarray(mesh["xyz_mm"], dtype=float)
    element_ids = np.asarray(mesh["element_ids"], dtype=np.int64)
    conn = np.asarray(mesh["connectivity_node_ids"], dtype=np.int64)
    if xyz.shape != (len(nodes), 3) or conn.shape != (len(element_ids), 4):
        raise ValueError("Expected original N x 3 coordinates and four-node connectivity")
    lookup = {int(n): i for i, n in enumerate(nodes)}
    if len(lookup) != len(nodes):
        raise ValueError("Duplicate node IDs")
    indices = np.array([[lookup[int(n)] for n in row] for row in conn])
    if any(len(set(row)) != 4 for row in indices):
        raise ValueError("Repeated Q4 corner")
    return nodes, xyz, element_ids, indices


def _inside_union(points, base_quads, tolerance):
    inside = np.zeros(len(points), dtype=bool)
    for polygon in base_quads:
        edge = np.roll(polygon, -1, axis=0)-polygon
        values = _cross(edge[None, :, :], points[:, None, :]-polygon[None, :, :])
        limits = tolerance*np.maximum(np.linalg.norm(edge, axis=1), 1.)
        inside |= np.all(values >= -limits, axis=1) | np.all(values <= limits, axis=1)
    return inside


def prepare_problem(mesh, config, marked_base_quads, *, extra_frozen_node_ids=()):
    """Record the active set once, entirely from original coordinates/topology.

    A refined element is protected when all four ORIGINAL vertices lie in the
    union of the supplied marked BASE quadrilaterals. All its nodes are frozen.
    """
    nodes, xyz, element_ids, indices = _indices(mesh)
    xy = xyz[:, :2].copy()
    a, b = float(config["a"]), float(config["b"])
    tolerance = float(config.get("coordinate_tolerance_mm", 1e-8))
    quality = config.get("quality_limits", {})
    values = (quality.get("aspect_ratio_max", 4.), quality.get("angle_min_degrees", 30.),
              quality.get("angle_max_degrees", 150.), quality.get("scaled_jacobian_min", .2))
    if tuple(map(float, values)) != (4., 30., 150., .2):
        raise ValueError("Repair requires the original 4 / 30..150 / 0.2 thresholds")
    original = geometry(xy, indices)
    if not np.isfinite(xyz).all() or np.any(original["determinant"] <= 0):
        raise ValueError("Original mesh must have finite coordinates and positive Jacobians")
    base_quads = np.asarray(marked_base_quads, dtype=float)
    if base_quads.size == 0:
        base_quads = np.empty((0, 4, 2))
    if base_quads.ndim != 3 or base_quads.shape[1:] != (4, 2):
        raise ValueError("marked_base_quads must have shape K x 4 x 2")
    inside = _inside_union(xy, base_quads, tolerance)
    protected_elements = np.all(inside[indices], axis=1)
    protected = np.zeros(len(nodes), dtype=bool)
    protected[indices[protected_elements].ravel()] = True
    boundary = (np.abs(xy[:, 0]) <= tolerance) | (np.abs(xy[:, 0]-a) <= tolerance) | \
               (np.abs(xy[:, 1]) <= tolerance) | (np.abs(xy[:, 1]-b) <= tolerance)
    bad = np.max(original["condition"], axis=1) > 4.
    edge_owners = defaultdict(list)
    hi = np.full(len(nodes), np.inf)
    for e, row in enumerate(indices):
        for k, (n, m) in enumerate(zip(row, np.roll(row, -1))):
            edge_owners[tuple(sorted((int(n), int(m))))].append(e)
            hi[n] = min(hi[n], original["edge_lengths"][e, k])
            hi[m] = min(hi[m], original["edge_lengths"][e, k])
    if not np.isfinite(hi).all() or np.any(hi <= 0):
        raise ValueError("Every original node must have a positive incident edge")
    neighbors = [set() for _ in element_ids]
    for owners in edge_owners.values():
        for i in owners:
            neighbors[i].update(set(owners)-{i})
    active_elements = set(np.flatnonzero(bad).tolist())
    for _ in range(NEIGHBOR_RINGS):
        active_elements |= {n for e in tuple(active_elements) for n in neighbors[e]}
    active = np.zeros(len(nodes), dtype=bool)
    if active_elements:
        active[indices[sorted(active_elements)].ravel()] = True
    frozen = boundary | protected | np.isin(nodes, list(extra_frozen_node_ids))
    active &= ~frozen
    active_indices = np.flatnonzero(active)
    affected = np.any(active[indices], axis=1)
    return {"nodes": nodes, "xyz": xyz.copy(), "xy": xy, "element_ids": element_ids,
            "indices": indices, "original": original, "hi": hi,
            "boundary": boundary, "protected": protected, "frozen": ~active,
            "protected_elements": protected_elements, "bad": bad,
            "active_elements": np.array(sorted(active_elements), dtype=int),
            "active_indices": active_indices, "affected": affected,
            "a": a, "b": b, "tolerance": tolerance}


def _unpack(vector, problem):
    xy = problem["xy"].copy()
    active = problem["active_indices"]
    xy[active] += vector[:-1].reshape(-1, 2)*problem["hi"][active, None]
    return xy


def _constraint_values(vector, problem):
    """Dimensionless >=0 inequalities, omitting constant feasible cell rows."""
    xy = _unpack(vector, problem)
    active = problem["active_indices"]
    affected = problem["affected"]
    geom = geometry(xy, problem["indices"][affected])
    original = {key: value[affected] for key, value in problem["original"].items()}
    fixed = problem["original"]["condition"][~affected]
    fixed_max = float(fixed.max()) if fixed.size else 1.
    ratios_area = geom["area"]/original["area"]
    ratios_edge = geom["edge_lengths"]/original["edge_lengths"]
    cosine = np.cos(np.deg2rad(30.))
    d = vector[:-1].reshape(-1, 2)
    h = problem["hi"][active]
    rows = [np.array([vector[-1]-fixed_max]),
            vector[-1]-geom["condition"].ravel(),
            (geom["determinant"]/original["determinant"]-.25).ravel(),
            (geom["scaled_jacobian"]-.2).ravel(),
            (cosine-geom["cosine_angles"]).ravel(),
            (cosine+geom["cosine_angles"]).ravel(),
            ratios_area-.5, 2.-ratios_area,
            (ratios_edge-.7).ravel(), (1.3-ratios_edge).ravel(),
            DISPLACEMENT_FRACTION**2-np.sum(d*d, axis=1),
            xy[active, 0]/h, (problem["a"]-xy[active, 0])/h,
            xy[active, 1]/h, (problem["b"]-xy[active, 1])/h]
    return np.concatenate(rows)


def evaluate_candidate(problem, xy):
    """Independent direct recomputation of every original hard restriction."""
    xy = np.asarray(xy, dtype=float)
    original = problem["original"]
    geom = geometry(xy, problem["indices"])
    displacement = np.linalg.norm(xy-problem["xy"], axis=1)/problem["hi"]
    determinant_ratio = geom["determinant"]/original["determinant"]
    area_ratio = geom["area"]/original["area"]
    edge_ratio = geom["edge_lengths"]/original["edge_lengths"]
    angle = np.rad2deg(np.arccos(np.clip(geom["cosine_angles"], -1, 1)))
    eps = FEASIBILITY_TOLERANCE
    checks = {
        "finite_coordinates": bool(np.isfinite(xy).all()),
        "fixed_nodes_unchanged": bool(np.array_equal(xy[problem["frozen"]], problem["xy"][problem["frozen"]])),
        "radial_displacement_cap": bool(np.max(displacement) <= .2+eps),
        "original_jacobian_fraction": bool(np.min(determinant_ratio) >= .25-eps),
        "positive_jacobians": bool(np.min(geom["determinant"]) > 0),
        "scaled_jacobian": bool(np.min(geom["scaled_jacobian"]) >= .2-eps),
        "angle_bounds": bool(angle.min() >= 30.-eps and angle.max() <= 150.+eps),
        "area_ratio_bounds": bool(area_ratio.min() >= .5-eps and area_ratio.max() <= 2.+eps),
        "edge_length_ratio_bounds": bool(edge_ratio.min() >= .7-eps and edge_ratio.max() <= 1.3+eps),
        "inside_rectangle": bool(xy[:, 0].min() >= -eps and xy[:, 0].max() <= problem["a"]+eps and
                                  xy[:, 1].min() >= -eps and xy[:, 1].max() <= problem["b"]+eps),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "maximum_condition": float(geom["condition"].max()),
            "maximum_displacement_over_hi": float(displacement.max()),
            "minimum_determinant_ratio": float(determinant_ratio.min()),
            "minimum_scaled_jacobian": float(geom["scaled_jacobian"].min()),
            "minimum_angle_degrees": float(angle.min()), "maximum_angle_degrees": float(angle.max()),
            "minimum_area_ratio": float(area_ratio.min()), "maximum_area_ratio": float(area_ratio.max()),
            "minimum_edge_ratio": float(edge_ratio.min()), "maximum_edge_ratio": float(edge_ratio.max())}


def optimize_coordinates(mesh, config, marked_base_quads, *, extra_frozen_node_ids=()):
    """Return a new mesh plus JSON audit; run exactly one fixed SLSQP attempt.

    A feasible improving iterate can be retained even if SLSQP does not declare
    convergence; optimizer termination and mesh-gate acceptance stay separate.
    No input dictionary or coordinate array is modified.
    """
    start = time.perf_counter()
    problem = prepare_problem(mesh, config, marked_base_quads,
                              extra_frozen_node_ids=extra_frozen_node_ids)
    initial = evaluate_candidate(problem, problem["xy"])
    active = problem["active_indices"]
    audit = {"algorithm": "one SLSQP epigraph minimax attempt",
             "maxiter": MAXITER, "ftol": FTOL, "neighbor_rings": NEIGHBOR_RINGS,
             "displacement_fraction_of_original_hi": DISPLACEMENT_FRACTION,
             "feasibility_tolerance": FEASIBILITY_TOLERANCE,
             "finite_difference_eps": FINITE_DIFFERENCE_EPS,
             "minimum_objective_improvement": MINIMUM_OBJECTIVE_IMPROVEMENT,
             "bad_original_element_ids": problem["element_ids"][problem["bad"]].tolist(),
             "active_region_element_ids": problem["element_ids"][problem["active_elements"]].tolist(),
             "affected_element_ids": problem["element_ids"][problem["affected"]].tolist(),
             "movable_node_ids": problem["nodes"][active].tolist(),
             "protected_element_ids": problem["element_ids"][problem["protected_elements"]].tolist(),
             "protected_node_ids": problem["nodes"][problem["protected"]].tolist(),
             "boundary_node_ids": problem["nodes"][problem["boundary"]].tolist(),
             "original_hi_mm": {str(int(problem["nodes"][i])): float(problem["hi"][i]) for i in active},
             "before": initial, "accepted_coordinate_change": False,
             "optimizer_performed": False, "solve_performed": False}
    output = copy.deepcopy(mesh)
    if not initial["passed"] or not len(active):
        audit.update({"reason": "initial hard restrictions fail" if not initial["passed"] else "no movable nodes",
                      "after": initial, "optimizer": None, "elapsed_seconds": time.perf_counter()-start})
        return output, audit
    x0 = np.zeros(2*len(active)+1)
    x0[-1] = initial["maximum_condition"]
    best = {"xy": problem["xy"].copy(), "evaluation": initial, "iteration": 0}
    iteration = 0
    history = []

    def retain(vector):
        nonlocal iteration
        iteration += 1
        xy = _unpack(vector, problem)
        evaluation = evaluate_candidate(problem, xy)
        history.append({"iteration": iteration, "tau": float(vector[-1]),
                        "maximum_condition": evaluation["maximum_condition"],
                        "hard_constraints_passed": evaluation["passed"],
                        "maximum_displacement_over_hi": evaluation["maximum_displacement_over_hi"]})
        if evaluation["passed"] and evaluation["maximum_condition"] < best["evaluation"]["maximum_condition"]:
            best.update(xy=xy, evaluation=evaluation, iteration=iteration)

    objective_jac = np.zeros(len(x0)); objective_jac[-1] = 1.
    optimum = minimize(lambda vector: vector[-1], x0, method="SLSQP",
                       jac=lambda vector: objective_jac,
                       bounds=[(-.2, .2)]*(len(x0)-1)+[(1., None)],
                       constraints={"type": "ineq", "fun": lambda vector: _constraint_values(vector, problem)},
                       callback=retain, options={"maxiter": MAXITER, "ftol": FTOL,
                                                 "eps": FINITE_DIFFERENCE_EPS, "disp": False})
    retain(optimum.x)
    audit["optimizer_performed"] = True
    audit["optimizer"] = {"success": bool(optimum.success), "status": int(optimum.status),
                          "message": str(optimum.message), "iterations": int(optimum.nit),
                          "function_evaluations": int(optimum.nfev),
                          "last_tau": float(optimum.x[-1]), "selected_iteration": int(best["iteration"]),
                          "last_minimum_inequality": float(_constraint_values(optimum.x, problem).min())}
    improved = best["evaluation"]["maximum_condition"] < initial["maximum_condition"]-MINIMUM_OBJECTIVE_IMPROVEMENT
    candidate_xyz = problem["xyz"].copy(); candidate_xyz[:, :2] = best["xy"]
    validation = validate_mesh(mesh["node_ids"], candidate_xyz, mesh["element_ids"],
                               mesh["connectivity_node_ids"], mesh["attributes"], config)
    nonaspect_pass = all(passed for key, passed in validation["checks"].items() if key != "aspect_ratio_threshold")
    accepted = improved and best["evaluation"]["passed"] and nonaspect_pass
    if accepted:
        output["xyz_mm"] = candidate_xyz.tolist()
    audit.update({"accepted_coordinate_change": bool(accepted), "selected_candidate": best["evaluation"],
                  "after": best["evaluation"] if accepted else initial,
                  "independent_candidate_validation": validation,
                  "iteration_history": history,
                  "reason": "feasible strict improvement" if accepted else "no independently valid strict improvement",
                  "elapsed_seconds": time.perf_counter()-start})
    return output, audit
