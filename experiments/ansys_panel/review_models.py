"""Read-only review of actual saved APDL models and solve evidence.

This checks the requested research panel against its input commands, rather
than treating a published configuration or constraint count as proof of BCs.
It imports no ANSYS package and never executes an APDL command.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re

import numpy as np


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"


def _hash(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(text):
    return float(text.replace("D", "E").replace("d", "e"))


def inspect_model(record, config):
    """Return JSON-compatible checks/details/hashes for one saved solve.

    ``record`` is a measurements.json entry; ``config`` is panel.json data.
    No files are written. A failed check or read/parse error makes passed False.
    Positive pivots and these checks do not replace a mechanism eigenanalysis
    or prove a discretization-error bound.
    """
    runtime = Path(record["runtime_directory"])
    checks, details, hashes = {}, {}, {}

    def check(name, passed, **values):
        checks[name] = bool(passed)
        if values:
            details[name] = values

    try:
        paths = {name: runtime / name for name in ("model.apdl", "solve_response.txt", "build.log", "fields.npz")}
        hashes = {name: _hash(path) for name, path in paths.items()}
        model = paths["model.apdl"].read_text(encoding="ascii")
        solve = paths["solve_response.txt"].read_text(encoding="utf-8")
        build = paths["build.log"].read_text(encoding="utf-8")
        lines = [line.strip() for line in model.splitlines() if line.strip() and not line.lstrip().startswith("!")]
        commands = [line.split(",") for line in lines]
        nx, ny, h = int(record["nx"]), int(record["ny"]), float(record["h"])
        check("grid_dimensions", np.isclose(nx*h, config["a"], rtol=0, atol=1e-12)
              and np.isclose(ny*h, config["b"], rtol=0, atol=1e-12))
        xx, yy = np.meshgrid(np.arange(nx+1)*h, np.arange(ny+1)*h)
        expected_xyz = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
        node_commands = [c for c in commands if c[0] == "N"]
        ids = np.array([int(c[1]) for c in node_commands])
        xyz = np.array([[_number(v) for v in c[2:]] for c in node_commands])
        check("actual_node_ids", np.array_equal(ids, np.arange(1, len(expected_xyz)+1)), count=len(ids))
        check("actual_node_coordinates", xyz.shape == expected_xyz.shape and np.allclose(xyz, expected_xyz, rtol=0, atol=1e-12),
              requested_bounds_mm=[0, config["a"], 0, config["b"], 0, 0])
        base = np.arange((ny+1)*(nx+1)).reshape(ny+1, nx+1)[:-1,:-1].ravel()+1
        expected_connectivity = np.column_stack((base, base+1, base+nx+2, base+nx+1))
        connectivity = np.array([[int(v) for v in c[1:]] for c in commands if c[0] == "E"])
        check("actual_element_connectivity", np.array_equal(connectivity, expected_connectivity), count=len(connectivity))
        normals = np.cross(xyz[connectivity[:,1]-1]-xyz[connectivity[:,0]-1], xyz[connectivity[:,3]-1]-xyz[connectivity[:,0]-1])
        check("actual_positive_z_normals", np.all(normals[:,2] > 0) and np.allclose(normals[:,:2], 0, rtol=0, atol=1e-12))
        boundary = np.flatnonzero((xx.ravel() == 0) | (xx.ravel() == config["a"]) | (yy.ravel() == 0) | (yy.ravel() == config["b"]))+1
        dofs = ("UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ")
        expected_d = {(int(node), dof, 0.0) for node in boundary for dof in (dofs if record["support"] == "P1" else ("UZ",))}
        if record["support"] == "P0":
            expected_d.update({(1,"UX",0.0), (1,"UY",0.0), (nx+1,"UY",0.0)})
        actual_d = [(int(c[1]), c[2], _number(c[3])) for c in commands if c[0] == "D"]
        check("exact_boundary_dof_set", len(actual_d) == len(set(actual_d)) and set(actual_d) == expected_d,
              actual_constraint_count=len(actual_d), expected_constraint_count=len(expected_d),
              boundary_nodes=len(boundary), counts_by_dof={dof:sum(item[1] == dof for item in actual_d) for dof in dofs},
              p0_inplane_anchors_mm=[[0,0,"UX=UY=0"],[config["a"],0,"UY=0"]] if record["support"] == "P0" else [])
        required = ["FINISH", "/CLEAR,NOSTART", "/PREP7", "ET,1,SHELL181", "ETCONTROL,OFF",
                    *[f"KEYOPT,1,{key},{value}" for key,value in config["keyopts"].items()],
                    f"MP,EX,1,{config['young']}", f"MP,PRXY,1,{config['poisson']}", "SECTYPE,1,SHELL",
                    f"SECDATA,{config['thickness']},1,0,{config['thickness_integration_points']}", "SECOFFSET,MID",
                    "TYPE,1", "MAT,1", "SECNUM,1", f"SFE,ALL,1,PRES,,{config['pressure']}", "ALLSEL,ALL", "CHECK",
                    "/SOLU", "ANTYPE,STATIC", "EQSLV,SPARSE", f"NLGEOM,{'ON' if record['nonlinear'] else 'OFF'}",
                    "OUTRES,ALL,ALL", "TIME,1", "AUTOTS,ON", "NSUBST,5,100,1" if record["nonlinear"] else "NSUBST,1", "NEQIT,50", "RESCONTROL,NONE"]
        missing = [line for line in required if line not in lines]
        # Fixed definitions/load commands occur exactly once; FINISH may repeat.
        unique_prefixes = {line.split(",")[0] for line in required} - {"FINISH", "MP", "KEYOPT"}
        multiplicity = {prefix:sum(c[0] == prefix for c in commands) for prefix in unique_prefixes}
        allowed = {line.split(",")[0] for line in required} | {"N","E","D","/NOPR","/GOPR"}
        unexpected = [line for line in lines if line.split(",")[0] not in allowed]
        check("fixed_settings_and_load_commands", not missing and all(value == 1 for value in multiplicity.values())
              and sum(c[0] == "MP" for c in commands) == 2 and sum(c[0] == "KEYOPT" for c in commands) == len(config["keyopts"]), missing=missing)
        check("no_unexpected_commands", not unexpected, unexpected=unexpected)
        equations = [int(v) for v in re.findall(r"Number of equations\s*=\s*(\d+)", solve, re.I)]
        expected_equations = 6*len(xyz)-len(expected_d)
        check("actual_solver_equations", bool(equations) and equations[-1] == expected_equations == record["active_dof"],
              solver_equations=equations[-1] if equations else None, from_exact_dofs=expected_equations)
        completions = re.findall(r"LOAD STEP\s+(\d+)\s+SUBSTEP\s+(\d+)\s+COMPLETED\.\s+CUM ITER\s*=\s*(\d+)", solve, re.I)
        times = [_number(v) for v in re.findall(r"\*\*\*\s+TIME\s*=\s*("+NUMBER+r")", solve, re.I)]
        check("completed_final_time", bool(completions) and bool(times) and np.isclose(times[-1], 1, rtol=0, atol=1e-9),
              completed_substeps=len(completions), final_substep=int(completions[-1][1]) if completions else None,
              cumulative_iterations=int(completions[-1][2]) if completions else None, final_time=times[-1] if times else None)
        marker = r"(?mi)^\s*\*\*\*\s+(?:WARNING|ERROR)\s+\*\*\*[^\r\n]*"
        check("no_build_solve_warning_error_markers", not re.findall(marker, solve+"\n"+build), markers=re.findall(marker, solve+"\n"+build))
        check("no_solve_failure_marker", not re.search(r"SOLUTION NOT CONVERGED|DOES NOT CONVERGE|VERIFICATION RUN ONLY", solve, re.I))
        pivots = [_number(v) for v in re.findall(r"Sparse solver minimum pivot=\s*("+NUMBER+r")", solve, re.I)]
        check("printed_minimum_pivots_positive", bool(pivots) and all(v > 0 for v in pivots), printed_minimum_pivots=pivots)
        check("nonlinear_record_cnvg", not record["nonlinear"] or record.get("raw_solu_cnvg") == 1,
              raw_record_solu_cnvg=record.get("raw_solu_cnvg"), linear_flag_not_assumed_to_be_one=True)
        with np.load(paths["fields.npz"], allow_pickle=False) as fields:
            center = (ny//2)*(nx+1)+nx//2
            field_xyz, field_uz = fields["xyz"], fields["uz"]
            check("field_coordinates_equal_actual_model", field_xyz.shape == xyz.shape and np.allclose(field_xyz, xyz, rtol=0, atol=1e-12))
            check("physical_center_and_displacement", np.allclose(xyz[center], [config["a"]/2,config["b"]/2,0], rtol=0, atol=1e-12)
                  and np.isfinite(field_uz[center]) and np.isclose(field_uz[center], record["central_deflection"], rtol=1e-12, atol=1e-12),
                  node_number=int(ids[center]), coordinates_mm=xyz[center].tolist(), uz_mm=float(field_uz[center]))
        check("record_input_hashes", hashes["model.apdl"] == record["apdl_sha256"] and hashes["fields.npz"] == record["fields_sha256"])
    except Exception as error:
        checks["inspection_completed"] = False
        details["error"] = {"type":type(error).__name__, "message":str(error)}
    return {"case":record.get("case"), "passed":all(checks.values()), "checks":checks, "details":details, "input_hashes":hashes,
            "limitation":"Saved-input and listing audit; no new solve, no modal mechanism test, no proof of the quantitative cause of the P0/Navier discrepancy"}
