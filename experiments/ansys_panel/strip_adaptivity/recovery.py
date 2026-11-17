"""Shell-resultant recovery indicator and element graph, without ANSYS calls.

Inputs are unsmoothed centroid resultants, in GLOBAL XY, ordered as
Nxx,Nyy,Nxy,Mxx,Myy,Mxy,Qx,Qy. q_h is their piecewise constant projection;
this is a recovery heuristic, not a guaranteed FE-error bound. Engineering
strain/curvature conventions are [exx,eyy,gxy] and [kxx,kyy,2*kxy], whereas
Nxy and Mxy are physical tensor entries. No fitted component weights occur.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np


COMPONENTS = ("Nxx", "Nyy", "Nxy", "Mxx", "Myy", "Mxy", "Qx", "Qy")
PARTS = ("membrane", "bending", "shear")


class RecoveryError(ValueError):
    """An explicit geometry, recovery, or quadrature acceptance failure."""


def _ids(value, name):
    raw = np.asarray(value)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.number):
        raise RecoveryError(f"{name} must be a one-dimensional integer sequence")
    if not np.isfinite(raw).all() or np.any(raw <= 0) or np.any(raw != np.floor(raw)):
        raise RecoveryError(f"{name} must contain positive integer IDs")
    ids = raw.astype(np.int64)
    if len(np.unique(ids)) != len(ids):
        raise RecoveryError(f"{name} contains duplicate IDs")
    return ids


def _shape(xi, eta):
    signs = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    n = .25*(1.+signs[:, 0]*xi)*(1.+signs[:, 1]*eta)
    deriv = .25*np.column_stack((signs[:, 0]*(1.+signs[:, 1]*eta),
                                  signs[:, 1]*(1.+signs[:, 0]*xi)))
    return n, deriv


def _geometry(mesh):
    node_ids = _ids(mesh["node_ids"], "node_ids")
    element_ids = _ids(mesh["element_ids"], "element_ids")
    xyz = np.asarray(mesh["xyz_mm"], dtype=float)
    if xyz.shape != (len(node_ids), 3) or not np.isfinite(xyz).all():
        raise RecoveryError("xyz_mm must be finite, with shape (number of nodes, 3)")
    if np.any(np.abs(xyz[:, 2]) > 1e-10):
        raise RecoveryError("This recovery implementation requires the flat XY mid-surface z=0")
    raw = np.asarray(mesh["connectivity_node_ids"])
    if raw.shape != (len(element_ids), 4) or not np.issubdtype(raw.dtype, np.number):
        raise RecoveryError("connectivity_node_ids must have shape (number of elements, 4)")
    if not np.isfinite(raw).all() or np.any(raw != np.floor(raw)):
        raise RecoveryError("Connectivity must contain actual integer node IDs")
    mapping = {int(n): i for i, n in enumerate(node_ids)}
    try:
        conn = np.array([[mapping[int(n)] for n in row] for row in raw], dtype=int)
    except KeyError as error:
        raise RecoveryError(f"Connectivity references missing node {error.args[0]}") from error
    if any(len(set(row)) != 4 for row in conn):
        raise RecoveryError("Each element must have four distinct vertices")
    if len(set(conn.ravel())) != len(node_ids):
        raise RecoveryError("Unused mesh nodes are not accepted")
    xy = xyz[:, :2]
    corners = xy[conn]
    center = np.mean(corners, axis=1)  # natural-coordinate centroid of Q4
    areas = np.zeros(len(conn))
    gauss, weight = np.polynomial.legendre.leggauss(2)
    locations = [(float(x), float(y), 0.) for x in (-1., 1.) for y in (-1., 1.)]
    locations += [(float(x), float(y), float(wx*wy))
                  for x, wx in zip(gauss, weight) for y, wy in zip(gauss, weight)]
    for xi, eta, w in locations:
        _, deriv = _shape(xi, eta)
        jac = np.einsum("eia,ib->eab", corners, deriv)
        det = np.linalg.det(jac)
        if np.any(det <= 0.) or not np.isfinite(det).all():
            bad = element_ids[np.where((det <= 0.) | ~np.isfinite(det))[0]].tolist()
            raise RecoveryError(f"Nonpositive Q4 Jacobian for elements {bad}")
        areas += w*det
    _, deriv0 = _shape(0., 0.)
    jac0 = np.einsum("eia,ib->eab", corners, deriv0)
    return {"node_ids": node_ids, "element_ids": element_ids, "xy": xy,
            "conn": conn, "corners": corners, "centers": center, "areas": areas,
            "dx": np.ptp(corners[:, :, 0], axis=1),
            "dy": np.ptp(corners[:, :, 1], axis=1),
            "jacobian_condition_center": np.linalg.cond(jac0)}


def _resultants(value, count=None):
    q = np.asarray(value, dtype=float)
    if q.ndim != 2 or q.shape[1] != 8 or not np.isfinite(q).all():
        raise RecoveryError("Resultants must be a finite (number of elements, 8) array")
    if count is not None and q.shape[0] != count:
        raise RecoveryError("Resultant rows must align with mesh element_ids")
    return q


def rotate_resultants(local, bases):
    """Rotate physical tensors/vectors into GLOBAL XY.

    Columns of each basis are local unit axes expressed in global axes. A
    single 2x2 or 3x3 basis can be broadcast; a 3x3 basis must have normal +Z.
    This function does not infer or silently replace a native element basis.
    """
    raw = np.asarray(local, dtype=float)
    single = raw.ndim == 1
    q = _resultants(raw[None, :] if single else raw)
    basis = np.asarray(bases, dtype=float)
    if basis.shape in ((2, 2), (3, 3)):
        basis = np.broadcast_to(basis, (len(q),)+basis.shape)
    if basis.shape not in ((len(q), 2, 2), (len(q), 3, 3)) or not np.isfinite(basis).all():
        raise RecoveryError("bases must be finite 2x2 or flat 3x3 rotations aligned with resultants")
    if not np.allclose(np.swapaxes(basis, 1, 2)@basis, np.eye(basis.shape[1]), rtol=0., atol=1e-10):
        raise RecoveryError("Element bases must be orthonormal")
    if not np.allclose(np.linalg.det(basis), 1., rtol=0., atol=1e-10):
        raise RecoveryError("Element bases must be proper rotations")
    if basis.shape[1] == 3:
        if (not np.allclose(basis[:, 2, :], [0., 0., 1.], rtol=0., atol=1e-10)
                or not np.allclose(basis[:, :, 2], [0., 0., 1.], rtol=0., atol=1e-10)):
            raise RecoveryError("Element normals must be +Z for this flat-shell implementation")
    r = basis[:, :2, :2]
    out = np.empty_like(q)
    for offset in (0, 3):
        tensor = np.empty((len(q), 2, 2))
        tensor[:, 0, 0], tensor[:, 1, 1] = q[:, offset], q[:, offset+1]
        tensor[:, 0, 1] = tensor[:, 1, 0] = q[:, offset+2]
        rotated = r@tensor@np.swapaxes(r, 1, 2)
        out[:, offset] = rotated[:, 0, 0]
        out[:, offset+1] = rotated[:, 1, 1]
        out[:, offset+2] = rotated[:, 0, 1]
    out[:, 6:8] = np.einsum("eij,ej->ei", r, q[:, 6:8])
    return out[0] if single else out


def constitutive_matrices(young, poisson, thickness, shear_stiffness):
    """Return A [N/mm], D [N mm], explicit S [N/mm] for an isotropic MID section."""
    e, nu, t = float(young), float(poisson), float(thickness)
    if not all(map(math.isfinite, (e, nu, t))) or e <= 0. or t <= 0. or not -1. < nu < .5:
        raise RecoveryError("Require finite E>0, t>0 and -1<nu<0.5")
    cp = e/(1.-nu*nu)*np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1.-nu)/2.]])
    s = np.asarray(shear_stiffness, dtype=float)
    if s.ndim == 0:
        s = float(s)*np.eye(2)
    if s.shape != (2, 2) or not np.isfinite(s).all() or not np.allclose(s, s.T, atol=1e-12, rtol=1e-12):
        raise RecoveryError("Explicit shear_stiffness must be a finite symmetric 2x2 matrix or positive scalar")
    if np.any(np.linalg.eigvalsh(s) <= 0.):
        raise RecoveryError("Explicit shear_stiffness must be positive definite")
    return t*cp, t**3/12.*cp, s.copy()


def _recover(g, q, condition_limit):
    count_nodes = len(g["node_ids"])
    incident = [set() for _ in range(count_nodes)]
    for element, nodes in enumerate(g["conn"]):
        for node in nodes:
            incident[node].add(element)
    nodal = np.empty((count_nodes, 8))
    audits = []
    for node, origin in enumerate(g["xy"]):
        patch = set(incident[node])
        rings = 0
        while True:
            index = np.array(sorted(patch), dtype=int)
            relative = g["centers"][index]-origin
            scale = float(np.max(np.linalg.norm(relative, axis=1)))
            if scale > 0.:
                design = np.column_stack((np.ones(len(index)), relative/scale))
                # Normalizing all area weights by the same constant changes no LS solution.
                weights = np.sqrt(g["areas"][index]/np.max(g["areas"][index]))
                x = design*weights[:, None]
                coefficients, _, rank, singular = np.linalg.lstsq(x, q[index]*weights[:, None], rcond=None)
                condition = float(singular[0]/singular[-1]) if rank == 3 else math.inf
            else:
                rank, condition = 1, math.inf
            if rank == 3 and condition <= condition_limit:
                nodal[node] = coefficients[0]
                audits.append({"node_id": int(g["node_ids"][node]), "element_count": len(patch),
                               "rings_added": rings, "rank": int(rank), "condition": condition,
                               "scale_mm": scale})
                break
            # Element adjacency by a shared vertex; this differs from graph edges.
            expanded = set(patch)
            for element in patch:
                for vertex in g["conn"][element]:
                    expanded.update(incident[vertex])
            if expanded == patch:
                raise RecoveryError(f"Patch recovery failed at node {g['node_ids'][node]}: "
                                    f"elements={len(patch)}, rank={rank}, condition={condition}")
            patch = expanded
            rings += 1
    return nodal, audits


def _integrate(g, q, recovered, inverses, order):
    points, weights = np.polynomial.legendre.leggauss(order)
    result = np.zeros((len(q), 3))
    fields = recovered[g["conn"]]
    for xi, wx in zip(points, weights):
        for eta, wy in zip(points, weights):
            n, deriv = _shape(float(xi), float(eta))
            det = np.linalg.det(np.einsum("eia,ib->eab", g["corners"], deriv))
            if np.any(det <= 0.):
                raise RecoveryError("Nonpositive Jacobian at indicator integration point")
            delta = np.einsum("i,eij->ej", n, fields)-q
            for part, (section, inverse) in enumerate(zip((slice(0, 3), slice(3, 6), slice(6, 8)), inverses)):
                d = delta[:, section]
                result[:, part] += wx*wy*det*np.einsum("ei,ij,ej->e", d, inverse, d)
    if not np.isfinite(result).all() or np.any(result < 0.):
        raise RecoveryError("Indicator integration produced an invalid energy")
    return result


def compute_indicator(mesh, resultants, *, young, poisson, thickness,
                      shear_stiffness, strain_energy, condition_limit=1e8,
                      quadrature_rtol=.001):
    """Recover resultants and integrate eta_K squared, in N mm.

    Initial patches contain incident cells and expand by shared-vertex element
    rings. Rank/conditioning failures raise RecoveryError without substitution.
    Integration compares orders 2/3, then 3/5 if needed and fails if unstable.
    shear_stiffness is mandatory: the caller must establish native section data.
    """
    if not math.isfinite(float(condition_limit)) or condition_limit < 1.:
        raise RecoveryError("condition_limit must be finite and >=1")
    if not math.isfinite(float(quadrature_rtol)) or not 0. < quadrature_rtol <= .001:
        raise RecoveryError("quadrature_rtol must be positive and <=0.001")
    energy = float(strain_energy)
    if not math.isfinite(energy) or energy <= 0.:
        raise RecoveryError("Positive finite strain_energy in N mm is required")
    g = _geometry(mesh)
    q = _resultants(resultants, len(g["element_ids"]))
    matrices = constitutive_matrices(young, poisson, thickness, shear_stiffness)
    inverses = tuple(np.linalg.inv(matrix) for matrix in matrices)
    recovered, patches = _recover(g, q, float(condition_limit))
    # Floating-point floor is only for comparing near-zero quadrature results;
    # it never truncates or changes an element indicator or a selector score.
    projected_energy = 0.
    for section, inverse in zip((slice(0, 3), slice(3, 6), slice(6, 8)), inverses):
        d = q[:, section]
        projected_energy += float(np.sum(g["areas"]*np.einsum("ei,ij,ej->e", d, inverse, d)))
    roundoff_floor = 1024.*np.finfo(float).eps**2*max(2.*energy, projected_energy)
    attempted, previous = [], None
    stable, accepted = False, None
    relative_change = None
    for order in (2, 3, 5):
        parts = _integrate(g, q, recovered, inverses, order)
        total = math.fsum(float(v) for v in parts.ravel())
        attempted.append({"order": order, "sum_eta2_N_mm": total,
                          "sum_parts_N_mm": dict(zip(PARTS, np.sum(parts, axis=0).tolist()))})
        if previous is not None:
            relative_change = abs(total-previous)/max(abs(total), abs(previous), roundoff_floor)
            if abs(total-previous) <= quadrature_rtol*max(abs(total), abs(previous))+roundoff_floor:
                stable, accepted = True, order
                break
        previous = total
    if not stable:
        raise RecoveryError(f"Indicator quadrature did not stabilize at 2/3/5: {attempted}")
    per_element = np.sum(parts, axis=1)
    return {"schema_version": 1, "indicator_kind": "centroid-projection shell-resultant recovery heuristic",
            "units_eta2": "N mm", "resultant_components": list(COMPONENTS),
            "element_ids": g["element_ids"].tolist(), "eta2": per_element.tolist(),
            "parts_per_element": {name: parts[:, j].tolist() for j, name in enumerate(PARTS)},
            "sum_eta2": float(total), "sum_parts": dict(zip(PARTS, np.sum(parts, axis=0).tolist())),
            "eta_rel": math.sqrt(total/(2.*energy)), "strain_energy_N_mm": energy,
            "recovered_node_ids": g["node_ids"].tolist(),
            "recovered_nodal_resultants": recovered.tolist(),
            "patches": {"adjacency": "elements sharing any vertex", "condition_limit": condition_limit,
                        "max_condition": max(p["condition"] for p in patches),
                        "max_rings_added": max(p["rings_added"] for p in patches), "nodes": patches},
            "quadrature": {"attempts": attempted, "accepted_order": accepted,
                           "relative_change": relative_change, "rtol": quadrature_rtol,
                           "roundoff_floor_N_mm": roundoff_floor, "passed": True},
            "section_stiffness": {"A_N_per_mm": matrices[0].tolist(),
                                  "D_N_mm": matrices[1].tolist(), "S_N_per_mm": matrices[2].tolist()},
            "assumptions": ["flat isotropic symmetric MID section; B=0",
                            "q_h is unsmoothed centroid field, constant on each element",
                            "no equilibrium constraints or artificial edge moments imposed",
                            "not a guaranteed bound or exact SHELL181 discrete energy norm"]}


def build_observation(mesh, resultants, indicator, config):
    """Build a serializable element graph; normalization uses fixed physics only.

    config requires a_mm,b_mm,pressure_N_per_mm2,young_N_per_mm2,poisson,
    thickness_mm. Global coordinates assume panel origin (0,0). Resultants and
    indicator rows must align exactly with mesh element_ids. No ML is run.
    """
    g = _geometry(mesh)
    q = _resultants(resultants, len(g["element_ids"]))
    if indicator["element_ids"] != g["element_ids"].tolist():
        raise RecoveryError("Indicator element_ids must match mesh row order exactly")
    eta2 = np.asarray(indicator["eta2"], dtype=float)
    parts = np.column_stack([indicator["parts_per_element"][name] for name in PARTS])
    if eta2.shape != (len(q),) or parts.shape != (len(q), 3) or not np.isfinite(parts).all():
        raise RecoveryError("Invalid per-element indicator in observation")
    if not np.isfinite(eta2).all() or np.any(eta2 < 0.) or np.any(parts < 0.):
        raise RecoveryError("Observation indicator must be finite and nonnegative")
    if not np.allclose(np.sum(parts, axis=1), eta2, rtol=1e-12, atol=0.):
        raise RecoveryError("Observation eta2 does not equal its mechanical parts")
    a, b = float(config["a_mm"]), float(config["b_mm"])
    pressure = float(config["pressure_N_per_mm2"])
    if not all(map(math.isfinite, (a, b, pressure))) or min(a, b) <= 0. or pressure == 0.:
        raise RecoveryError("Fixed physical panel dimensions and nonzero pressure are required")
    _, d, _ = constitutive_matrices(config["young_N_per_mm2"], config["poisson"],
                                   config["thickness_mm"], np.eye(2))
    nscale = abs(pressure)*b
    mscale = abs(pressure)*b*b
    escale = pressure**2*b**6/d[0, 0]
    normalizers = {"length_x_mm": a, "length_y_mm": b, "area_mm2": a*b,
                   "N_and_Q_N_per_mm": nscale, "M_N": mscale, "eta2_N_mm": float(escale),
                   "definition": "N,Q=|p|*b; M=|p|*b^2; eta2=p^2*b^6/D11; fixed input physics"}
    names = ["center_x/a", "center_y/b", "area/(a*b)", "dx/a", "dy/b", "cond_J_center"]
    names += [f"{name}/{'M' if 3 <= i < 6 else 'NQ'}_scale" for i, name in enumerate(COMPONENTS)]
    names += [f"eta2_{part}/energy_scale" for part in PARTS]+["eta2/energy_scale"]
    features = np.column_stack((g["centers"]/np.array([a, b]), g["areas"]/(a*b),
                                g["dx"]/a, g["dy"]/b, g["jacobian_condition_center"],
                                q/np.array([nscale]*3+[mscale]*3+[nscale]*2),
                                parts/escale, eta2/escale))
    owners = defaultdict(list)
    for element, vertices in enumerate(g["conn"]):
        for i in range(4):
            edge = tuple(sorted((int(vertices[i]), int(vertices[(i+1) % 4]))))
            owners[edge].append(element)
    directed, edge_features = [], []
    neighbors = [[] for _ in q]
    for edge, elements in sorted(owners.items()):
        if len(elements) > 2:
            raise RecoveryError("Nonmanifold shared edge in element graph")
        if len(elements) == 2:
            i, j = sorted(elements, key=lambda k: int(g["element_ids"][k]))
            length = float(np.linalg.norm(g["xy"][edge[1]]-g["xy"][edge[0]]))
            for source, target in ((i, j), (j, i)):
                directed.append((source, target))
                delta = (g["centers"][target]-g["centers"][source])/np.array([a, b])
                edge_features.append([float(delta[0]), float(delta[1]), length/b])
                neighbors[source].append(int(g["element_ids"][target]))
    order = sorted(range(len(directed)), key=lambda k: directed[k])
    directed = [directed[k] for k in order]
    return {"schema_version": 1, "node_semantics": "finite element, not mesh vertex",
            "element_ids": g["element_ids"].tolist(), "eta2": eta2.tolist(),
            "feature_names": names, "node_features": features.tolist(),
            "edge_index": [[pair[i] for pair in directed] for i in (0, 1)],
            "edge_features": [edge_features[k] for k in order],
            "edge_feature_names": ["delta_center_x/a", "delta_center_y/b", "shared_edge_length/b"],
            "edge_definition": "bidirectional shared complete mesh edge; no diagonal vertex edges",
            "normalization": normalizers,
            "geometry": {"center_xy_mm": g["centers"].tolist(), "area_mm2": g["areas"].tolist(),
                         "dx_mm": g["dx"].tolist(), "dy_mm": g["dy"].tolist(),
                         "neighbor_element_ids": [sorted(row) for row in neighbors]},
            "native_resultants_global": q.tolist(), "resultant_components": list(COMPONENTS),
            "indicator_parts_N_mm": {name: parts[:, j].tolist() for j, name in enumerate(PARTS)},
            "training_or_inference_executed": False}
