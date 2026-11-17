"""Native SHELL181 centroid resultants for the planar isotropic strip pilot.

The caller owns the MAPDL session and must enter POST1 and load its result set.
This module never creates a session, changes a mesh, or requests a solution.
SMISC sequence items retain the element coordinate system, independently of
RSYS. No nodal averaging or stress extrapolation is used by this exporter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ELEMENT_SOURCE = "https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/ans_elem/Hlp_E_SHELL181.html"
THEORY_SOURCE = "https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/ans_thry/thy_el181.html"
GET_SOURCE = "https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/ans_cmd/Hlp_C_GET.html"
VGET_SOURCE = "https://ansyshelp.ansys.com/public/views/secured/corp/v261/en/ans_cmd/Hlp_C_VGET_st.html"
LABELS = ("N11", "N22", "N12", "M11", "M22", "M12", "Q13", "Q23")
SEQUENCES = (*range(1, 9), 15, 16)


def identify_centroid_shear_stiffness(ids, gamma, q, tolerance=1e-6):
    """Identify a symmetric centroid relation using even IDs; test on odd IDs.

    This is an operational constitutive identification from native solved Q and
    gamma, without reference errors, mesh marking or fitted error weights. It
    does not establish equality to the assembled shell's full energy operator.
    The fixed original residual tolerance also applies to the holdout and all
    cells. A mesh with spatially varying effective stiffness may fail this test.
    """
    ids = np.asarray(ids, dtype=np.int64)
    gamma, q = np.asarray(gamma, dtype=float), np.asarray(q, dtype=float)
    train, test = ids % 2 == 0, ids % 2 != 0
    evidence = {"method": "symmetric linear least squares Q=S*gamma",
        "split": "even actual element IDs fit; odd actual element IDs independent holdout",
        "training_element_ids": ids[train].tolist(), "holdout_element_ids": ids[test].tolist(),
        "relative_tolerance": tolerance, "reference_data_used": False,
        "fitted_error_weights": False, "assembled_energy_operator_verified": False,
        "scope": "Effective native centroid constitutive relation; not an exact FE energy-norm identity",
        "accepted": False, "failure_reasons": []}
    failures = evidence["failure_reasons"]
    if not np.any(train) or not np.any(test):
        failures.append("both_parity_groups_required")
        return None, evidence
    design = np.zeros((2 * len(ids), 3))
    design[0::2, :2] = gamma
    design[1::2, 1:] = gamma
    row_train = np.repeat(train, 2)
    coefficients, _, rank, singular = np.linalg.lstsq(design[row_train], q.ravel()[row_train], rcond=None)
    evidence["symmetric_design_rank"] = int(rank)
    evidence["training_design_condition_number"] = (float(singular[0] / singular[-1])
        if len(singular) and singular[-1] > 0 else None)
    if rank != 3:
        failures.append("symmetric_constitutive_design_not_full_rank_three")
        return None, evidence
    a, b, c = coefficients
    matrix = np.array([[a, b], [b, c]])
    evidence["identified_matrix"] = matrix.tolist()
    positive = bool(np.isfinite(matrix).all() and np.linalg.eigvalsh(matrix).min() > 0)
    evidence["symmetric_positive_definite"] = positive
    if not positive:
        failures.append("identified_matrix_not_SPD")
        return None, evidence
    scale = float((a + c) / 2.)
    evidence["diagonal_relative_difference"] = float(abs(a - c) / scale)
    evidence["off_diagonal_relative_magnitude"] = float(abs(b) / scale)
    if max(evidence["diagonal_relative_difference"], evidence["off_diagonal_relative_magnitude"]) > tolerance:
        failures.append("identified_relation_not_isotropic_within_original_tolerance")
    error = q - gamma @ matrix.T
    for label, mask in (("training", train), ("holdout", test), ("all_cells", np.ones(len(ids), dtype=bool))):
        scale_q = float(np.linalg.norm(q[mask]))
        residual = float(np.linalg.norm(error[mask]) / max(scale_q, 1e-30))
        evidence[label + "_relative_residual"] = residual
        if scale_q <= 1e-12 or residual > tolerance:
            failures.append(label + "_constitutive_residual_exceeds_original_tolerance")
    # Also retain and test each cell, with only a numerical near-zero denominator
    # floor (1e-12 times RMS native Q); this is not a fitted physical parameter.
    rms_q = float(np.linalg.norm(q) / np.sqrt(len(ids)))
    cell_relative = np.linalg.norm(error, axis=1) / np.maximum(np.linalg.norm(q, axis=1), max(1e-30, 1e-12 * rms_q))
    evidence["maximum_cell_relative_residual"] = float(cell_relative.max())
    evidence["near_zero_Q_denominator_floor"] = max(1e-30, 1e-12 * rms_q)
    if cell_relative.max() > tolerance:
        failures.append("individual_cell_constitutive_residual_exceeds_original_tolerance")
    evidence["accepted"] = not failures
    return (matrix if not failures else None), evidence


def _native_column(mapdl, ids, label):
    """VGET starts at actual ID 1, so row i belongs to element i+1.

    Native VGET loops IDs even through holes. A short/packed gRPC response is
    never zipped with a noncontiguous ID list: explicit scalar GET is used.
    """
    try:
        dense = np.asarray(mapdl.get_array("ELEM", 1, item1="ETAB", it1num=label), dtype=float).ravel()
        if len(dense) >= int(ids.max()):
            values = dense[ids - 1]
            method = "native VGET ELEM,1,ETAB,label; indexed by actual element_id-1"
        else:
            values = np.array([mapdl.get_value("ELEM", int(e), "ETAB", label) for e in ids], dtype=float)
            method = "native scalar GET ELEM,actual_id,ETAB,label; short VGET fallback"
    except Exception:
        values = np.array([mapdl.get_value("ELEM", int(e), "ETAB", label) for e in ids], dtype=float)
        method = "native scalar GET ELEM,actual_id,ETAB,label; VGET unavailable"
    if values.shape != ids.shape or not np.isfinite(values).all():
        raise ValueError(f"Invalid native ETABLE column {label}")
    # Independent endpoint probes detect an unexpected gRPC vector convention.
    for index in sorted({0, len(ids) - 1}):
        scalar = float(mapdl.get_value("ELEM", int(ids[index]), "ETAB", label))
        if not np.isclose(values[index], scalar, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Native ETABLE ID alignment check failed for {label}")
    return values, method


def export_resultants(mapdl, mesh, work, config):
    """Return JSON-serializable local resultants, bases and shear stiffness.

    Root-runner aliases are ``local_resultants`` (n,8), ``bases_xy`` (n,2,2;
    local axes are columns in global XY), and ``shear_stiffness_matrix`` (2,2).
    The mesh must contain actual positive IDs and global undeformed coordinates.
    Only the preserved one-layer isotropic, MID-offset SHELL181 model is accepted.
    """
    work = Path(work)
    target = work / "native_resultants_section_export.json"
    if target.exists():
        raise FileExistsError("Native resultant evidence already exists; use a new runtime")
    ids = np.asarray(mesh["element_ids"], dtype=np.int64)
    if ids.ndim != 1 or not len(ids) or np.any(ids <= 0) or len(np.unique(ids)) != len(ids):
        raise ValueError("Expected unique positive actual element IDs")
    mapdl.allsel()
    raw = {int(row[8]): np.asarray(row, dtype=np.int64) for row in mapdl.mesh.elem}
    if set(raw) != set(ids.tolist()):
        raise ValueError("Native and supplied element IDs differ")
    native_nodes = dict(zip(np.asarray(mapdl.mesh.nnum, dtype=np.int64).tolist(),
                            np.asarray(mapdl.mesh.nodes, dtype=float)))
    supplied_nodes = dict(zip(mesh["node_ids"], mesh["xyz_mm"]))
    if set(native_nodes) != set(supplied_nodes):
        raise ValueError("Native and supplied node IDs differ")
    if any(not np.allclose(native_nodes[n], supplied_nodes[n], rtol=0, atol=1e-8) for n in native_nodes):
        raise ValueError("Native undeformed coordinates differ from the accepted mesh")
    type_ids = {int(raw[int(e)][1]) for e in ids}
    if type_ids != {1} or any(int(raw[int(e)][0]) != 1 or int(raw[int(e)][3]) != 1 for e in ids):
        raise ValueError("The pilot requires native TYPE=1, MAT=1, SECNUM=1")
    if any(int(raw[int(e)][4]) != 0 for e in ids):
        raise ValueError("Nonzero native ESYS requires an explicit orientation implementation")
    element_name = int(round(mapdl.get_value("ETYP", 1, "ATTR", "ENAM")))
    keyopts = {str(k): int(round(mapdl.get_value("ETYP", 1, "ATTR", f"KOP{k}" if k < 10 else f"KO{k}")))
               for k in (1, 3, 8, 11)}
    if element_name != 181 or keyopts != {"1": 0, "3": 2, "8": 2, "11": 0}:
        raise ValueError("Native SHELL181 formulation/orientation differs from the preserved model")
    bases, centroids = [], []
    for index, e in enumerate(ids):
        conn = raw[int(e)][10:14].tolist()
        if conn != mesh["connectivity_node_ids"][index] or len(set(conn)) != 4:
            raise ValueError("Native connectivity differs or is not four distinct vertices")
        p = np.array([native_nodes[n] for n in conn])
        # Q4 derivatives at xi=eta=0; the documented default SHELL181 axis is S1.
        s1 = (-p[0] + p[1] + p[2] - p[3]) / 4.
        s2 = (-p[0] - p[1] + p[2] + p[3]) / 4.
        e1 = s1 / np.linalg.norm(s1)
        e3 = np.cross(s1, s2); e3 /= np.linalg.norm(e3)
        e2 = np.cross(e3, e1)
        basis = np.column_stack((e1, e2, e3))
        if (not np.isfinite(basis).all() or not np.allclose(basis, np.eye(3), atol=1e-10, rtol=0)
                or not np.allclose(p[1] - p[0], p[2] - p[3], atol=1e-8, rtol=0)
                or not np.allclose(p[3] - p[0], p[2] - p[1], atol=1e-8, rtol=0)
                or not np.allclose(p[:, 2], 0., atol=1e-8, rtol=0)):
            raise ValueError("The exporter accepts only CCW axis-aligned planar strip rectangles")
        bases.append(basis)
        centroids.append(p.mean(axis=0))
    columns, retrieval = [], {}
    for name, sequence in zip((*LABELS, "G13", "G23"), SEQUENCES):
        label = "SR" + name
        mapdl.etable(label, "SMISC", sequence)
        values, method = _native_column(mapdl, ids, label)
        columns.append(values); retrieval[name] = method
    values = np.column_stack(columns)
    prop = lambda item: float(mapdl.get_value("SHEL", 1, "PROP", item))
    thickness, nlay, position = prop("TTHK"), int(round(prop("NLAY"))), int(round(prop("POS")))
    layer_mat = int(round(mapdl.get_value("SHEL", 1, "LAYD", 1, "MAT")))
    layer_angle = float(mapdl.get_value("SHEL", 1, "LAYD", 1, "ANGL"))
    overrides = {name: prop(name) for name in ("UT11", "UT22", "UT12")}
    young = float(mapdl.get_value("EX", 1, "TEMP", 0))
    poisson = float(mapdl.get_value("PRXY", 1, "TEMP", 0))
    if (nlay != 1 or position != 1 or layer_mat != 1 or layer_angle != 0
            or any(value != 0 for value in overrides.values())
            or not np.isclose(thickness, config["thickness"], rtol=0, atol=1e-10)
            or not np.isclose(young, config["young"], rtol=1e-12, atol=0)
            or not np.isclose(poisson, config["poisson"], rtol=0, atol=1e-12)):
        raise ValueError("Native section/material properties do not justify the documented isotropic default")
    default = np.eye(2) * ((5. / 6.) * young / (2. * (1. + poisson)) * thickness)
    native_matrix, native_query_error = None, None
    try:
        candidate = np.array([[mapdl.get_value("SHEL", 1, "PROP", "E", i, j)
                               for j in (1, 2)] for i in (1, 2)], dtype=float)
        native_matrix = candidate.tolist() if np.isfinite(candidate).all() else None
        valid = (np.isfinite(candidate).all() and np.allclose(candidate, candidate.T, atol=1e-8, rtol=1e-10)
                 and np.linalg.eigvalsh(candidate).min() > 0)
    except Exception as error:
        valid = False
        native_query_error = type(error).__name__
    q, gamma = values[:, 6:8], values[:, 8:10]
    try:
        native_ts_factors={label:prop(label) for label in ("TS11","TS22","TS12")}
    except Exception as error:
        native_ts_factors={"query_error_type":type(error).__name__}
    # PROP,E is the uncorrected section shear matrix in this preserved isotropic
    # case (Gt I). PROP,TS11/22 supply the native shear-correction factors. Do not
    # infer these section properties by fitting independently exported centroid
    # resultants and generalized strains from an assumed-strain shell element.
    ts_available = (set(native_ts_factors) == {"TS11", "TS22", "TS12"}
                    and all(np.isfinite(v) for v in native_ts_factors.values()))
    gt = young / (2. * (1. + poisson)) * thickness
    section_checks = {
        "native_uncorrected_E_finite_symmetric_positive_definite": bool(valid),
        "native_TS_factors_available": bool(ts_available),
        "native_uncorrected_E_matches_Gt_identity": bool(valid and np.allclose(candidate, np.eye(2) * gt, rtol=1e-6, atol=1e-8)),
        "native_TS12_zero": bool(ts_available and abs(native_ts_factors["TS12"]) <= 1e-12),
        "native_TS_diagonals_match_documented_5_over_6": bool(ts_available and np.allclose(
            [native_ts_factors["TS11"], native_ts_factors["TS22"]], 5. / 6., rtol=1e-6, atol=0)),
    }
    corrected = (np.diag([candidate[i, i] * native_ts_factors[f"TS{i+1}{i+1}"] for i in range(2)])
                 if valid and ts_available else None)
    section_checks["corrected_section_S_SPD"] = bool(corrected is not None and np.linalg.eigvalsh(corrected).min() > 0)
    section_checks["corrected_section_S_isotropic"] = bool(corrected is not None and np.allclose(
        corrected, np.eye(2) * np.trace(corrected) / 2., rtol=1e-6, atol=1e-8))
    section_verified = all(section_checks.values())
    q_scale = float(np.linalg.norm(q))
    gamma_scale = float(np.linalg.norm(gamma))
    informative = q_scale > 1e-12 and gamma_scale > 1e-20
    residual_default = float(np.linalg.norm(q - gamma @ default.T) / max(q_scale, 1e-30))
    residual_native = (float(np.linalg.norm(q - gamma @ candidate.T) / max(q_scale, 1e-30))
                       if native_matrix is not None else None)
    ratios = []
    for component in range(2):
        mask = np.abs(gamma[:, component]) > 1e-18
        values_q_gamma = q[mask, component] / gamma[mask, component]
        ratios.append({"nonzero_gamma_count": int(mask.sum()),
                       "minimum_Q_over_gamma": float(values_q_gamma.min()) if len(values_q_gamma) else None,
                       "median_Q_over_gamma": float(np.median(values_q_gamma)) if len(values_q_gamma) else None,
                       "maximum_Q_over_gamma": float(values_q_gamma.max()) if len(values_q_gamma) else None})
    native_verified = bool(valid and informative and residual_native <= 1e-6)
    default_verified = bool(informative and residual_default <= 1e-6)
    residual_section = (float(np.linalg.norm(q - gamma @ corrected.T) / max(q_scale, 1e-30))
                        if corrected is not None else None)
    diagnostic = {"native_section_E_matrix": native_matrix, "native_TS_factors":native_ts_factors,"native_E_symmetric_positive_definite": bool(valid),
        "native_corrected_section_S_matrix": corrected.tolist() if corrected is not None else None,
        "native_section_checks": section_checks, "native_section_verified": section_verified,
        "residual_vs_corrected_native_section": residual_section,
        "centroid_Q_S_gamma_identity_is_acceptance_gate": False,
        "centroid_identity_limitation": "Independently exported SHELL181 assumed-strain/incompatible-mode centroid Q and gamma need not reproduce the section constitutive identity; their geometry-dependent ratios are diagnostic, not section stiffness",
        "native_query_error_type": native_query_error, "documented_default_matrix": default.tolist(),
        "native_diagonal_over_documented_default": [native_matrix[i][i] / default[i, i] for i in range(2)] if native_matrix is not None else None,
        "Q_norm": q_scale, "gamma_norm": gamma_scale, "element_count": len(ids),
        "residual_vs_default": residual_default, "residual_vs_native": residual_native,
        "relative_residual_tolerance": 1e-6, "componentwise_Q_over_gamma": ratios,
        "native_matrix_verified_by_Q_gamma": native_verified, "documented_default_verified_by_Q_gamma": default_verified,
        "native_material_and_section": {"young_MPa": young, "poisson": poisson, "thickness_mm": thickness,
            "layers": nlay, "offset_position": position, "layer_material": layer_mat,
            "layer_angle_degrees": layer_angle, "user_transverse_shear_overrides": overrides,
            "element_name": element_name, "keyopts": keyopts}}
    diagnostic_index = 1
    while (work / f"native_shear_diagnostic_{diagnostic_index:04d}.json").exists():
        diagnostic_index += 1
    diagnostic_path = work / f"native_shear_diagnostic_{diagnostic_index:04d}.json"
    with diagnostic_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(diagnostic, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if not section_verified:
        raise ValueError("Native E and TS section properties did not verify the preserved isotropic shear stiffness")
    chosen, residual, verified = corrected, residual_section, True
    identification = {"used": False, "status": "rejected_as_section_stiffness_source",
        "reason": "A constant centroid fit can pass on one uniform geometry and fail after geometry changes; it does not establish the native section stiffness",
        "old_fit_residual_tolerance_unchanged": 1e-6,
        "new_fit_or_per_element_identification_performed": False}
    basis_array = np.asarray(bases)
    result = {"element_ids": ids.tolist(), "labels": list(LABELS),
        "local_resultants": values[:, :8].tolist(), "values_local": values[:, :8].tolist(),
        "bases_xy": basis_array[:, :2, :2].tolist(), "basis_local_to_global": basis_array.tolist(),
        "centroid_xyz_mm": np.asarray(centroids).tolist(), "gamma_local": gamma.tolist(),
        "shear_stiffness_matrix": chosen.tolist(), "shear_stiffness_N_per_mm": chosen.tolist(),
        "native_shear_stiffness_verified": verified,
        "units": {"N11_N22_N12_Q13_Q23": "N/mm", "M11_M22_M12": "N", "shear_stiffness": "N/mm", "gamma": "dimensionless"},
        "native_evidence": {"element_name": element_name, "keyopts": keyopts,
            "all_element_ESYS_zero": True, "element_ids_conn_coordinates_checked": True,
            "basis_source": "Native undeformed coordinates/connectivity; Q4 centroid S1 and cross(S1,S2); default KEYOPT(11)=0, ESYS=0",
            "element_angle_query": "No documented ELEM ANG centroid query used; basis derived from verified native geometry and formulation",
            "retrieval": retrieval, "smisc_sequences": dict(zip((*LABELS, "gamma13", "gamma23"), SEQUENCES)),
            "location": "Element centroid SMISC sequence values, no nodal averaging or recovery", "RSYS_rotation_used": False},
        "shear_stiffness_evidence": {"source": "native_section_E_diagonal_times_native_TS_correction_factors",
            "native_section_E_matrix": native_matrix,"native_TS_factors":native_ts_factors,"native_query_error_type": native_query_error,
            "native_corrected_section_S_matrix": chosen.tolist(), "native_section_checks": section_checks,
            "native_section_constitutive_stiffness_verified": True,
            "formula_scope": "Preserved isotropic diagonal section only: S11=E11*TS11, S22=E22*TS22, S12=0; native E12 and TS12 checked zero",
            "native_GET_limitation": "GET preprocessing section quantities may not reflect calculated section data; zero/non-SPD E is not treated as actual stiffness",
            "documented_default_matrix": default.tolist(), "k": 5. / 6., "young_MPa": young,
            "poisson": poisson, "thickness_mm": thickness, "layers": nlay, "offset_position": position,
            "user_transverse_shear_overrides": overrides, "Q_S_gamma_relative_residual": residual,
            "residual_vs_default": residual_default, "residual_vs_native": residual_native,
            "Q_S_gamma_identity_is_acceptance_gate": False,
            "Q_S_gamma_identity_within_old_tolerance": bool(informative and residual <= 1e-6),
            "centroid_identity_limitation": diagnostic["centroid_identity_limitation"],
            "centroid_constitutive_identification": identification,
            "assembled_energy_operator_verified": False,
            "native_matrix_matches_documented_default": bool(valid and np.allclose(candidate, default, rtol=1e-6, atol=1e-8)),
            "diagnostic_sha256": hashlib.sha256(diagnostic_path.read_bytes()).hexdigest(),
            "relative_residual_tolerance": 1e-6, "informative_nonzero_shear": informative},
        "sources": {"element": ELEMENT_SOURCE, "theory": THEORY_SOURCE, "GET": GET_SOURCE, "VGET": VGET_SOURCE}}
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    result["private_export_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    return result
