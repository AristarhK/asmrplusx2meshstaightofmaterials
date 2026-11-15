"""Independently audit saved panel exports without launching or importing ANSYS.

Reads raw files in each measurement's runtime_directory and writes a NEW JSON.
The checks establish internal export consistency, not a discretization-error
bound or independent validation of SHELL181's formulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

import numpy as np


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
COMPONENTS = ["sx", "sy", "sz", "txy", "tyz", "txz"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_finite(value):
    """Check nested JSON numbers, ignoring strings, booleans and nulls."""
    if isinstance(value, dict):
        return all(numeric_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(numeric_finite(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(np.isfinite(value))
    return True


def stress_row(text, node):
    """Parse one six-component PRNSOL row, including wrapped/adjacent fields."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"\s*(\d+)\s+", line)
        if match and int(match.group(1)) == node:
            values = re.findall(NUMBER, line[match.end():])
            following = index + 1
            while len(values) < 6 and following < len(lines):
                continuation = lines[following].strip()
                if not continuation or re.search(r"[A-CF-Za-cf-z]", continuation):
                    break
                values.extend(re.findall(NUMBER, continuation))
                following += 1
            if len(values) == 6:
                return np.array([float(v.replace("D", "E").replace("d", "e")) for v in values])
    raise ValueError(f"No unambiguous six-component PRNSOL row for node {node}")


def equivalent(stress):
    """Independent 3-D von Mises implementation, MPa."""
    sx, sy, sz, txy, tyz, txz = np.moveaxis(stress, -1, 0)
    return np.sqrt(((sx-sy)**2+(sy-sz)**2+(sz-sx)**2)/2 + 3*(txy*txy+tyz*tyz+txz*txz))


def audit_case(record, config):
    checks, details, diagnostics = {}, {}, {}
    runtime = Path(record["runtime_directory"])

    def check(name, passed, **information):
        checks[name] = bool(passed)
        if information:
            details[name] = information

    try:
        fields_path = runtime / "fields.npz"
        with np.load(fields_path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        xyz, conn = arrays["xyz"], arrays["connectivity"]
        xs, ys, uz = arrays["xs"], arrays["ys"], arrays["uz"]
        top, bottom = arrays["top"], arrays["bottom"]
        force, moment = arrays["reactions"], arrays["reaction_moments"]
        nx, ny = int(record["nx"]), int(record["ny"])
        check("all_exported_values_finite", numeric_finite(record) and all(np.isfinite(a).all() for a in arrays.values()))
        check("array_dimensions", xyz.shape == ((nx+1)*(ny+1), 3) and conn.shape == (nx*ny, 4)
              and uz.shape == (len(xyz),) and top.shape == bottom.shape == (ny+1, nx+1, 6)
              and force.shape == moment.shape == xyz.shape and xs.shape == (nx+1,) and ys.shape == (ny+1,))
        check("counts_match_record", len(xyz) == record["nodes"] and len(conn) == record["elements"])
        check("field_hash", digest(fields_path) == record["fields_sha256"])
        model_path = runtime / "model.apdl"
        model = model_path.read_text(encoding="ascii")
        check("model_hash", digest(model_path) == record["apdl_sha256"])
        normals = np.cross(xyz[conn[:,1]]-xyz[conn[:,0]], xyz[conn[:,3]]-xyz[conn[:,0]])
        check("positive_z_normals", np.all(normals[:,2] > 0) and np.allclose(normals[:,:2], 0, rtol=0, atol=1e-12),
              minimum_cross_product_z=float(normals[:,2].min()))
        check("planar_unrotated_panel", np.allclose(xyz[:,2], 0, atol=1e-12) and not re.search(r"^\s*NROTAT|^\s*NMODIF", model, re.M|re.I))
        check("positive_deflection", uz.max() > 0 and record["central_deflection"] > 0)
        center_node = (ny//2)*(nx+1)+nx//2+1
        center_index = (ny//2, nx//2)
        check("displacement_measurements", np.isclose(uz[center_node-1], record["central_deflection"], rtol=1e-12, atol=1e-12)
              and np.isclose(np.abs(uz).max(), record["max_abs_uz"], rtol=1e-12, atol=1e-12))
        text_hashes = {}
        for surface, tensor, label in (("top", top, "TOP"), ("bottom", bottom, "BOTTOM")):
            text_path = runtime / f"center_stress_{surface}.txt"
            text = text_path.read_text(encoding="utf-8")
            printed = stress_row(text, center_node)
            exported = tensor[center_index]
            check(f"center_{surface}_explicit_text_matches", np.allclose(printed, exported, rtol=1e-9, atol=1e-9),
                  component_order=COMPONENTS, printed_mpa=printed.tolist(), exported_mpa=exported.tolist(),
                  maximum_absolute_difference_mpa=float(np.max(np.abs(printed-exported))))
            check(f"center_{surface}_location_label", bool(re.search(r"SHELL NODAL RESULTS ARE AT\s+"+label, text)))
            check(f"center_{surface}_global_axes", "ARE IN GLOBAL COORDINATES" in text)
            # Absence of a NAR warning is recorded, not proof of the selected source.
            diagnostics[f"center_{surface}_nar_indicator_present"] = bool(re.search(r"NODAL[- ]AVERAGED|\bNAR\b", text, re.I))
            text_hashes[text_path.name] = digest(text_path)
        if not record["nonlinear"]:
            check("linear_top_bottom_antisymmetry", np.allclose(top, -bottom, rtol=1e-9, atol=1e-9),
                  maximum_absolute_surface_sum_mpa=float(np.max(np.abs(top+bottom))),
                  rationale="Flat symmetric homogeneous section, linear pure transverse bending, zero membrane loading")
        else:
            diagnostics["surface_antisymmetry"] = "Not required: geometrically nonlinear membrane stresses are permitted"

        solve_path = runtime / "solve_response.txt"
        solve = solve_path.read_text(encoding="utf-8")
        final_times = [float(value.replace("D", "E")) for value in re.findall(r"\*\*\*\s+TIME\s*=\s*("+NUMBER+r")", solve, re.I)]
        completed = bool(re.search(r"LOAD STEP\s+1\s+SUBSTEP\s+\d+\s+COMPLETED", solve, re.I))
        check("completed_at_final_time", completed and bool(final_times) and np.isclose(final_times[-1], 1, rtol=0, atol=1e-9),
              final_listing_time=final_times[-1] if final_times else None)
        errors = [line.strip() for line in solve.splitlines() if re.search(r"\*\*\* ERROR \*\*\*|SOLUTION NOT CONVERGED|DOES NOT CONVERGE", line, re.I)]
        check("no_solver_error_markers", not errors, markers=errors)
        check("verification_mode_disabled", "VERIFICATION RUN ONLY" not in solve)
        equations = [int(value) for value in re.findall(r"Number of equations\s*=\s*(\d+)", solve, re.I)]
        fixed_pairs = re.findall(r"^\s*D,(\d+),(UX|UY|UZ|ROTX|ROTY|ROTZ),0\s*$", model, re.M|re.I)
        independently_expected = 6*len(xyz)-len(set(fixed_pairs))
        check("actual_equation_count", bool(equations) and equations[-1] == independently_expected == record["active_dof"] == record["expected_free_dof"],
              from_listing=equations[-1] if equations else None, from_nodes_and_constraints=independently_expected)
        check("record_source_status", record["status"] == "completed" and bool(record["converged"]))
        diagnostics["raw_solu_cnvg"] = record.get("raw_solu_cnvg")
        diagnostics["cnvg_interpretation"] = "Raw value preserved; acceptance uses completed final-time listing, finite readable exports and independent balances. It is not rewritten as a successful raw flag."

        resultant = force.sum(axis=0)
        resultant_moment = (np.cross(xyz, force)+moment).sum(axis=0)
        external = np.asarray(record["external_force"], dtype=float)
        external_moment = np.asarray(record["external_moment_origin"], dtype=float)
        load = config["pressure"]*config["a"]*config["b"]
        check("prescribed_load_resultant", np.allclose(external, [0,0,load], rtol=1e-12, atol=1e-9)
              and np.allclose(external_moment, [load*config["b"]/2, -load*config["a"]/2, 0], rtol=1e-12, atol=1e-9))
        force_relative = float(np.linalg.norm(resultant+external)/abs(load))
        moment_relative = float(np.linalg.norm(resultant_moment+external_moment)/np.linalg.norm(external_moment))
        tolerance = config["criteria"]["force_balance"]
        check("force_balance", force_relative <= tolerance, relative_residual=force_relative, tolerance=tolerance)
        check("moment_balance", moment_relative <= tolerance, relative_residual=moment_relative, tolerance=tolerance)
        check("reactions_match_measurement", np.allclose(resultant, record["reaction_force"], rtol=1e-12, atol=1e-9)
              and np.allclose(resultant_moment, record["reaction_moment_origin"], rtol=1e-12, atol=1e-7))
        element_areas = normals[:,2]
        nodal_load = np.zeros(len(xyz))
        np.add.at(nodal_load, conn.ravel(), np.repeat(config["pressure"]*element_areas/4, 4))
        half_work = float(nodal_load@uz/2)
        check("half_work_export_consistency", np.isclose(half_work, record["half_external_work"], rtol=1e-12, atol=1e-9), recomputed_half_work_n_mm=half_work)
        if not record["nonlinear"]:
            energy_relative = abs(record["strain_energy"]-half_work)/abs(record["strain_energy"])
            check("linear_energy_work_identity", energy_relative <= 1e-5, relative_residual=energy_relative, tolerance=1e-5)
        else:
            diagnostics["energy_work_identity"] = "Half final-load work is not an exact identity for geometrically nonlinear response; no linear identity gate applied"
        maximum_vm = max(float(equivalent(top).max()), float(equivalent(bottom).max()))
        diagnostics["yield_check"] = {"maximum_recovered_nodal_von_mises_mpa": maximum_vm,
            "yield_diagnostic_mpa": config["yield_diagnostic"], "below_yield_diagnostic": maximum_vm <= config["yield_diagnostic"],
            "interpretation": "Applicability diagnostic for recovered shell-surface stress; not a normative allowable or a proof about unexported integration-point stresses"}
        diagnostics["audit_input_hashes"] = {**text_hashes, solve_path.name:digest(solve_path), fields_path.name:digest(fields_path), model_path.name:digest(model_path)}
    except Exception as error:
        checks["audit_completed"] = False
        details["audit_error"] = {"type":type(error).__name__, "message":str(error)}
    return {"case":record.get("case"), "runtime_directory":str(runtime), "passed":all(checks.values()),
            "checks":checks, "details":details, "diagnostics":diagnostics}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Audit output already exists: {args.output}")
    manifest_path = args.measurements.with_name("manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = json.loads(args.measurements.read_text(encoding="utf-8"))
    results = [audit_case(record, manifest["config"]) for record in records]
    report = {"audited_utc":datetime.now(timezone.utc).isoformat(), "passed":bool(results) and all(row["passed"] for row in results),
        "case_count":len(results), "measurements_sha256":digest(args.measurements), "manifest_sha256":digest(manifest_path),
        "audit_source_sha256":digest(__file__), "cases":results,
        "scope":"Read-only checks of saved fields and listings; no ANSYS launch and no claim of a proven reference-solution error bound"}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Audited {len(results)} cases; passed={report['passed']}")
    for row in results:
        failed = [name for name, value in row["checks"].items() if not value]
        print(f"{row['case']}: {'PASS' if row['passed'] else 'FAIL'}" + (f" ({', '.join(failed)})" if failed else ""))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
