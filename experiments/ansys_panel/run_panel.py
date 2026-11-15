"""Run isolated, reproducible SHELL181 panel studies with an existing PyMAPDL.

No machine paths in this module. Every invocation creates a new output and
runtime directory; --previous imports immutable measurements from a prior stage.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
from datetime import datetime, timezone

import numpy as np
from analysis import navier_convergence, stress_metrics, transition_convergence


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def grid(config, h):
    nx, ny = round(config["a"] / h), round(config["b"] / h)
    xs, ys = np.linspace(0, config["a"], nx + 1), np.linspace(0, config["b"], ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    xyz = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    ij = np.arange((ny + 1) * (nx + 1)).reshape(ny + 1, nx + 1)[:-1, :-1].ravel()
    conn = np.column_stack((ij, ij + 1, ij + nx + 2, ij + nx + 1))
    normals = np.cross(xyz[conn[:, 1]] - xyz[conn[:, 0]], xyz[conn[:, 3]] - xyz[conn[:, 0]])
    assert np.allclose(normals[:, :2], 0) and np.all(normals[:, 2] > 0)
    edges = np.flatnonzero((xyz[:, 0] == 0) | (xyz[:, 0] == config["a"]) | (xyz[:, 1] == 0) | (xyz[:, 1] == config["b"]))
    return nx, ny, xs, ys, xyz, conn, edges


def model_text(c, h, support, nonlinear):
    nx, ny, xs, ys, xyz, conn, edges = grid(c, h)
    constraints = {int(n + 1): (["UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ"] if support == "P1" else ["UZ"]) for n in edges}
    if support == "P0":
        constraints[1] += ["UX", "UY"]
        constraints[nx + 1] += ["UY"]
    lines = ["FINISH", "/CLEAR,NOSTART", "/PREP7", "/NOPR", "ET,1,SHELL181", "ETCONTROL,OFF"]
    lines += [f"KEYOPT,1,{k},{v}" for k, v in c["keyopts"].items()]
    lines += [f"MP,EX,1,{c['young']}", f"MP,PRXY,1,{c['poisson']}", "SECTYPE,1,SHELL", f"SECDATA,{c['thickness']},1,0,{c['thickness_integration_points']}", "SECOFFSET,MID", "TYPE,1", "MAT,1", "SECNUM,1"]
    lines += [f"N,{i+1},{x:.12g},{y:.12g},0" for i, (x, y, _) in enumerate(xyz)]
    lines += ["E," + ",".join(str(int(n + 1)) for n in nodes) for nodes in conn]
    lines += [f"D,{node},{dof},0" for node, dofs in constraints.items() for dof in dofs]
    lines += [f"SFE,ALL,1,PRES,,{c['pressure']}", "ALLSEL,ALL", "/GOPR", "CHECK", "FINISH", "/SOLU", "ANTYPE,STATIC", "EQSLV,SPARSE", f"NLGEOM,{'ON' if nonlinear else 'OFF'}", "OUTRES,ALL,ALL", "TIME,1", "AUTOTS,ON", "NSUBST,5,100,1" if nonlinear else "NSUBST,1", "NEQIT,50", "RESCONTROL,NONE"]
    return "\n".join(lines) + "\n", constraints, (nx, ny, xs, ys, xyz, conn, edges)


def export_reactions(mapdl, path, constraints, xyz):
    # Query only constrained DoFs; no undefined free-DoF reactions or FSUM.
    lines = ["*CFOPEN,reactions,csv", "*VWRITE,'node,dof,value'", "(A)"]
    for node, dofs in constraints.items():
        for dof in dofs:
            label = {"UX":"FX", "UY":"FY", "UZ":"FZ", "ROTX":"MX", "ROTY":"MY", "ROTZ":"MZ"}[dof]
            lines += [f"*GET,PANRF,NODE,{node},RF,{label}", f"*VWRITE,{node},{['UX','UY','UZ','ROTX','ROTY','ROTZ'].index(dof)+1},PANRF", "(F12.0,',',F3.0,',',E25.16)"]
    lines += ["*CFCLOS"]
    script = path / "reactions.apdl"
    script.write_text("\n".join(lines) + "\n", encoding="ascii")
    mapdl.input(str(script))
    data = np.loadtxt(path / "reactions.csv", delimiter=",", skiprows=1)
    forces, moments = np.zeros_like(xyz), np.zeros_like(xyz)
    for node, dof, value in data:
        if int(dof) <= 3:
            forces[int(node)-1, int(dof)-1] = value
        else:
            moments[int(node)-1, int(dof)-4] = value
    return forces, moments


def run_case(mapdl, c, h, support, nonlinear, runtime):
    name = f"{support}_h{h:g}_{'nonlinear' if nonlinear else 'linear'}"
    work = runtime / name
    work.mkdir()
    text, constraints, arrays = model_text(c, h, support, nonlinear)
    nx, ny, xs, ys, xyz, conn, edges = arrays
    mapdl.finish()
    mapdl.cwd(str(work))
    mapdl.filname("panel")
    model = work / "model.apdl"
    start = time.perf_counter()
    model.write_text(text, encoding="ascii")
    build_response = mapdl.input(str(model))
    (work / "build.log").write_text(str(build_response), encoding="utf-8")
    mesh_time = time.perf_counter() - start
    mapdl.run("/OUTPUT,solve,out")
    start = time.perf_counter()
    try:
        response = mapdl.solve()
    finally:
        solve_time = time.perf_counter() - start
        mapdl.run("/OUTPUT")
    (work / "solve_response.txt").write_text(str(response), encoding="utf-8")
    # gRPC returns SOLVE listing directly; /OUTPUT can remain empty on this API.
    solve_log = str(response) + "\n" + (work / "solve.out").read_text(errors="replace")
    equations = re.findall(r"number of equations\s*=\s*(\d+)", solve_log, re.I)
    if not equations:
        raise RuntimeError("Actual equation count absent in solver output")
    raw_cnvg = int(mapdl.get_value("ACTIVE", 0, "SOLU", "CNVG"))
    completed = bool(re.search(r"LOAD STEP\s+1\s+SUBSTEP\s+\d+\s+COMPLETED", solve_log, re.I))
    if not completed or re.search(r"\*\*\* ERROR \*\*\*|SOLUTION NOT CONVERGED", solve_log, re.I):
        raise RuntimeError("No completed solution substep or solver reports failure")
    cnvg = completed
    start = time.perf_counter()
    mapdl.finish()
    mapdl.post1()
    mapdl.set("LAST")
    result_time = float(mapdl.get_value("ACTIVE", 0, "SET", "TIME"))
    if not np.isclose(result_time, 1.0, atol=1e-10, rtol=0):
        raise RuntimeError(f"Last result time is {result_time}, expected full load TIME=1")
    mapdl.allsel()
    mapdl.run("/GRAPHICS,FULL")
    mapdl.rsys(0)
    mapdl.run("LAYER,1")
    mapdl.run("AVPRIN,0")
    ids = np.asarray(mapdl.mesh.nnum)
    actual_xyz = np.asarray(mapdl.mesh.nodes)
    assert np.array_equal(ids, np.arange(1, len(xyz)+1)) and np.allclose(actual_xyz, xyz)
    assert int(mapdl.get_value("ELEM", 0, "COUNT")) == len(conn)
    uz = np.asarray(mapdl.post_processing.nodal_displacement("Z"))
    if not np.isfinite(uz).all() or uz.max() <= 0:
        raise RuntimeError("Displacements nonfinite or pressure direction incorrect")
    surfaces, metrics = {}, {}
    for surface, option in (("top", "TOP"), ("bottom", "BOT")):
        mapdl.shell(option)
        components = [np.asarray(mapdl.get_array("NODE", item1="S", it1num=k, item2="ESOL")) for k in ("X", "Y", "Z", "XY", "YZ", "XZ")]
        stress = np.column_stack(components).reshape(ny+1, nx+1, 6)
        if not np.isfinite(stress).all():
            raise RuntimeError("Nonfinite shell surface stresses")
        surfaces[surface] = stress
        metrics[surface] = stress_metrics(xs, ys, stress, margin=c["interior_margin"], radius=c["disk_radius"], centers=c["disk_centers"])
        # Independent textual check at the center with explicit element source.
        center = int((ny//2)*(nx+1)+nx//2+1)
        mapdl.nsel("S", "NODE", "", center)
        mapdl.run("/FORMAT,9,E,22,14")
        (work / f"center_stress_{surface}.txt").write_text(mapdl.run("PRNSOL,S,COMP,,ESOL"), encoding="utf-8")
        mapdl.allsel()
    mapdl.etable("ESTRAIN", "SENE")
    mapdl.etable("EART", "AENE")
    mapdl.ssum()
    energy = float(mapdl.get_value("SSUM", 0, "ITEM", "ESTRAIN"))
    artificial = float(mapdl.get_value("SSUM", 0, "ITEM", "EART"))
    forces, moments = export_reactions(mapdl, work, constraints, xyz)
    reaction_force = forces.sum(axis=0)
    reaction_moment = (np.cross(xyz, forces) + moments).sum(axis=0)
    load = c["pressure"] * c["a"] * c["b"]
    external_moment = np.array([load*c["b"]/2, -load*c["a"]/2, 0])
    nodal_load = np.zeros(len(xyz))
    np.add.at(nodal_load, conn.ravel(), c["pressure"]*h*h/4)
    half_work = float(nodal_load @ uz / 2)
    np.savez_compressed(work / "fields.npz", xs=xs, ys=ys, xyz=xyz, connectivity=conn, uz=uz, top=surfaces["top"], bottom=surfaces["bottom"], reactions=forces, reaction_moments=moments)
    warnings = [line.strip() for line in (solve_log + "\n" + str(build_response)).splitlines() if re.search(r"\*\*\*\s+(WARNING|ERROR)\s+\*\*\*", line, re.I)]
    expected_equations = 6*len(xyz) - sum(map(len, constraints.values()))
    balance = abs(float(reaction_force[2])+load)/load
    record = {"case": name, "support": support, "nonlinear": nonlinear, "h": h, "nx": nx, "ny": ny, "elements": len(conn), "nodes": len(xyz), "active_dof": int(equations[-1]), "dof_source": "MAPDL sparse solver Number of equations", "expected_free_dof": expected_equations, "converged": cnvg, "raw_solu_cnvg": raw_cnvg, "central_deflection": float(uz.reshape(ny+1,nx+1)[ny//2,nx//2]), "max_abs_uz": float(np.abs(uz).max()), "strain_energy": energy, "artificial_energy": artificial, "half_external_work": half_work, "linear_energy_relative_check": abs(energy-half_work)/energy, "reaction_force": reaction_force.tolist(), "reaction_moment_origin": reaction_moment.tolist(), "external_force": [0,0,load], "external_moment_origin": external_moment.tolist(), "force_balance_relative": balance, "moment_balance_relative": float(np.linalg.norm(reaction_moment+external_moment)/np.linalg.norm(external_moment)), "disks": {s:m["disks"] for s,m in metrics.items()}, "stress_extrema": {s:{k:v for k,v in m.items() if k != "disks"} for s,m in metrics.items()}, "timing_seconds": {"mesh": mesh_time, "solve": solve_time, "export": time.perf_counter()-start}, "mesh_quality": {"aspect_ratio": 1.0, "minimum_angle_degrees": 90.0, "warpage_degrees": 0.0, "jacobian_determinant": h*h/4, "normal_z_min": 1.0, "normal_z_max": 1.0, "source": "all generated Q4 connectivity and coordinates, uniform planar squares"}, "solver_warning_lines": warnings, "runtime_directory": str(work), "apdl_sha256": sha(model), "fields_sha256": sha(work/"fields.npz"), "status": "completed"}
    record["result_time"] = result_time
    record["solver_pivot_lines"] = [line.strip() for line in solve_log.splitlines() if "pivot=" in line]
    record["acceptance_passed"] = balance <= c["criteria"]["force_balance"] and int(equations[-1]) == expected_equations
    if not record["acceptance_passed"]:
        record["status"] = "failed_validation"
    write_json(work / "measurement.json", record)
    if balance > c["criteria"]["force_balance"] or int(equations[-1]) != expected_equations:
        raise RuntimeError(f"Force/DoF gate failed; preserved measurement in {work}")
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exec-file", default=os.getenv("PYMAPDL_MAPDL_EXEC"), required=not os.getenv("PYMAPDL_MAPDL_EXEC"))
    ap.add_argument("--runtime-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("panel.json"))
    ap.add_argument("--levels", nargs="+", type=int, default=list(range(5)))
    ap.add_argument("--previous", type=Path)
    ap.add_argument("--nonlinear", action="store_true")
    args = ap.parse_args()
    c = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    runtime = args.runtime_root.resolve() / run_id
    runtime.mkdir(parents=True, exist_ok=False)
    records = []
    if args.previous:
        prior = json.loads((args.previous / "manifest.json").read_text(encoding="utf-8"))
        if prior["config_sha256"] != sha(args.config):
            raise ValueError("Previous stage uses different physical config")
        records = json.loads((args.previous / "measurements.json").read_text(encoding="utf-8"))
    provenance = {"started_utc": run_id, "config_sha256": sha(args.config), "source_sha256": {p.name:sha(p) for p in Path(__file__).parent.glob("*.py")}, "python": sys.version, "python_executable": sys.executable, "packages": {p:importlib.metadata.version(p) for p in ["ansys-mapdl-core","ansys-mapdl-reader","numpy","scipy","matplotlib"]}, "solver_executable": str(Path(args.exec_file).resolve()), "runtime_directory": str(runtime), "nproc":2, "additional_switches":"-smp", "previous_stage": str(args.previous) if args.previous else None, "config":c}
    write_json(args.output / "manifest.json", provenance)
    write_json(args.output / "navier.json", navier_convergence(**{k:c[k] for k in ("a","b","thickness","young","poisson","pressure")}))
    from ansys.mapdl.core import launch_mapdl
    mapdl = None
    failure = None
    try:
        mapdl = launch_mapdl(exec_file=args.exec_file, run_location=str(runtime), nproc=2, additional_switches="-smp", start_timeout=120, start_instance=True, mode="grpc", remove_temp_dir_on_exit=False, log_apdl=str(runtime/"session.apdl"))
        mapdl.finish()
        mapdl.verify("OFF")
        write_json(args.output / "session.json", {"description":str(mapdl), "version":str(mapdl.version), "ip":str(mapdl.ip), "port":int(mapdl.port)})
        for level in args.levels:
            h = c["mesh_sizes"][level]
            for support in ("P0", "P1"):
                if any(r["support"] == support and r["h"] == h and not r["nonlinear"] for r in records):
                    continue
                print(f"START {support} h={h:g}", flush=True)
                record = run_case(mapdl,c,h,support,False,runtime)
                records.append(record)
                write_json(args.output/f"{record['case']}.json", record)
                print(f"DONE {record['case']} elements={record['elements']} dof={record['active_dof']} w={record['central_deflection']:.10g} balance={record['force_balance_relative']:.3g}", flush=True)
            passed = []
            for support in ("P0", "P1"):
                seq = sorted((r for r in records if r["support"] == support and not r["nonlinear"]), key=lambda r:-r["h"])
                tests = [transition_convergence(a,b) for a,b in zip(seq,seq[1:])]
                passed.append(len(tests)>=2 and all(t["passed"] for t in tests[-2:]))
            if all(passed):
                print("Both supports have two consecutive passing refinement transitions", flush=True)
                break
        if args.nonlinear:
            h = min(r["h"] for r in records if r["support"] == "P1" and not r["nonlinear"])
            print(f"START P1 nonlinear h={h:g}", flush=True)
            record = run_case(mapdl,c,h,"P1",True,runtime)
            records.append(record)
            write_json(args.output/f"{record['case']}.json", record)
            print(f"DONE {record['case']} w={record['central_deflection']:.10g}", flush=True)
    except Exception as error:
        failure = {"type":type(error).__name__, "message":str(error), "traceback":traceback.format_exc()}
        write_json(args.output / "failure.json", failure)
        print(failure["traceback"], flush=True)
    finally:
        if mapdl is not None:
            try:
                mapdl.exit()
            except Exception as error:
                write_json(args.output / "exit_warning.json", {"message":str(error)})
        write_json(args.output / "measurements.json", records)
        fields = ["case","support","nonlinear","h","elements","nodes","active_dof","central_deflection","max_abs_uz","strain_energy","artificial_energy","force_balance_relative","moment_balance_relative"]
        with (args.output / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
            writer=csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key:r[key] for key in fields} for r in records)
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
