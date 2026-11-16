"""Separate immutable localization and coordinate-optimization stages, no MAPDL."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np

from coordinate_diagnostics import diagnose, plot_diagnostics
import optimize_coordinates as optimizer
from publish_mesh_gate import check_hash, quality, read_json
from run_mesh_gate import digest, save_json

CASES=("interior_single","edge_patch","corner_patch")


def verify_artifacts(root):
    root=root.resolve()
    for name,sha in read_json(root/"artifact_manifest.json")["files"].items():
        path=(root/name).resolve()
        if root not in path.parents:raise ValueError("Artifact escapes input directory")
        check_hash(path,sha)


def check_protocol(protocol):
    expected={"maxiter":optimizer.MAXITER,"ftol":optimizer.FTOL,
              "finite_difference_eps":optimizer.FINITE_DIFFERENCE_EPS,
              "acceptance_constraint_tolerance":optimizer.FEASIBILITY_TOLERANCE,
              "minimum_objective_improvement":optimizer.MINIMUM_OBJECTIVE_IMPROVEMENT,
              "max_radial_displacement_over_h_i":optimizer.DISPLACEMENT_FRACTION,
              "attempts_per_case":1,"cases":list(CASES)}
    for key,value in expected.items():
        if protocol[key]!=value:raise ValueError(f"Algorithm differs from frozen protocol: {key}")
    if optimizer.NEIGHBOR_RINGS!=2:raise ValueError("Active-region policy changed")


def manifest(output,source,protocol_path,stage):
    save_json(output/"manifest.json",{"stage":stage,"created_utc":datetime.now(timezone.utc).isoformat(),
        "source_commit":"24bba5fd0e1c2cf7d9da25a77607b5c12fd7625b",
        "source_artifact_manifest_sha256":digest(source/"artifact_manifest.json"),
        "protocol_sha256":digest(protocol_path),
        "packages":{p:importlib.metadata.version(p) for p in ("numpy","scipy","matplotlib")},
        "source_sha256":{p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},
        "solve_performed":False,"indicator_computed":False,"training_performed":False,
        "primary_goal_met":None,"strict_goal_met":None,"accuracy_note":"Not assessed: no solve"})
    save_json(output/"artifact_manifest.json",{"files":{p.relative_to(output).as_posix():digest(p) for p in output.rglob("*") if p.is_file()}})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=("localize","optimize"))
    ap.add_argument("--source",type=Path,default=Path(__file__).parent/"runs/20261116_mesh_gate01")
    ap.add_argument("--protocol",type=Path,default=Path(__file__).parent/"coordinate_protocol.json")
    ap.add_argument("--diagnostics",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    verify_artifacts(args.source)
    protocol=read_json(args.protocol);check_protocol(protocol)
    if args.stage=="optimize":
        if args.diagnostics is None:raise ValueError("Saved prior localization is mandatory")
        verify_artifacts(args.diagnostics)
        if digest(args.protocol)!=read_json(args.diagnostics/"manifest.json")["protocol_sha256"]:
            raise ValueError("Protocol changed after localization")
    args.output.mkdir(parents=True,exist_ok=False)
    figures=args.output/"figures";figures.mkdir()
    shutil.copyfile(args.protocol,args.output/"protocol.json")
    config=read_json(args.source/"execution_manifest.json")["config"]
    save_json(args.output/"config.json",config)
    base=read_json(args.source/"base_mesh.json")
    coordinates={int(n):np.array(p[:2]) for n,p in zip(base["node_ids"],base["xyz_mm"])}
    base_cells={int(e):np.array([coordinates[n] for n in ns]) for e,ns in zip(base["element_ids"],base["connectivity_node_ids"])}
    records=[]
    for name in CASES:
        original=read_json(args.source/f"{name}_mesh.json")
        source_record=read_json(args.source/f"{name}.json")
        marked=source_record["marked_element_ids"]
        if args.stage=="localize":
            start=time.perf_counter();diag=diagnose(original,base,marked,config)
            save_json(args.output/f"{name}_diagnostics.json",diag)
            plot_diagnostics(original,diag,figures/name,f"{name}: original failed EREFINE")
            records.append({"case":name,"bad_element_ids":diag["bad_element_ids"],"elapsed_diagnostics_and_figures_seconds":time.perf_counter()-start})
            print(f"LOCALIZED {name}: {diag['bad_element_ids']}",flush=True)
            continue
        before=read_json(args.diagnostics/f"{name}_diagnostics.json")
        print(f"OPTIMIZE {name}: single fixed attempt",flush=True)
        start=time.perf_counter()
        improved,audit=optimizer.optimize_coordinates(original,config,np.array([base_cells[e] for e in marked]))
        optimization_wall=time.perf_counter()-start
        start=time.perf_counter();q=quality(improved,config);validation_wall=time.perf_counter()-start
        start=time.perf_counter();after=diagnose(improved,base,marked,config,reference_mesh=original)
        save_json(args.output/f"{name}_diagnostics.json",after)
        plot_diagnostics(improved,after,figures/name,f"{name}: coordinate candidate",bounds=before["transition_bounds_mm"])
        plotting_wall=time.perf_counter()-start
        # Machine-independent evidence of every moved node and original local h_i.
        old=np.array(original["xyz_mm"]);new=np.array(improved["xyz_mm"])
        moved=[]
        for i,n in enumerate(original["node_ids"]):
            distance=float(np.linalg.norm(new[i]-old[i]))
            if distance>0:
                hi=audit["original_hi_mm"][str(n)]
                moved.append({"node_id":n,"before_xyz_mm":old[i].tolist(),"after_xyz_mm":new[i].tolist(),"displacement_mm":distance,"original_hi_mm":hi,"displacement_over_hi":distance/hi})
        structural={k:improved[k]==original[k] for k in ("node_ids","element_ids","connectivity_node_ids","attributes","native_settings")}
        protected=before["marked_region"]["protected_node_ids"]
        fixed_lookup={n:i for i,n in enumerate(original["node_ids"])}
        protected_unchanged=all(np.array_equal(old[fixed_lookup[n]],new[fixed_lookup[n]]) for n in protected)
        record={"case":name,"quality":q,"coordinate_optimization":audit,"unchanged_structure":structural,
            "marked_nodes_fixed_exactly":protected_unchanged,"moved_nodes":moved,
            "marked_size_before":before["marked_region"]["size_statistics"],"marked_size_after":after["marked_region"]["size_statistics"],
            "original_mesh_sha256":digest(args.source/f"{name}_mesh.json"),
            "native_before_parallel_deviation_warnings":source_record["native_parallel_deviation_warnings"],
            "native_check_performed":False,"native_warning_free":None,
            "timing_seconds":{"optimization_and_internal_validation":optimization_wall,"independent_geometry_validation":validation_wall,"diagnostics_and_figures":plotting_wall},
            "solve_performed":False,"indicator_computed":False,"training_performed":False,
            "primary_goal_met":None,"strict_goal_met":None}
        if not all(structural.values()) or not protected_unchanged:raise ValueError("Immutable mesh property changed")
        save_json(args.output/f"{name}_mesh.json",improved)
        save_json(args.output/f"{name}.json",record);records.append(record)
        print(f"DONE {name}: kappa={q['metrics']['maximum_aspect_ratio']:.8f} geometry_pass={q['passed']}",flush=True)
    save_json(args.output/"summary.json",{"stage":args.stage,"cases":records,"no_solve":True})
    if args.stage=="optimize":
        fields=("case","condition_before","condition_after","scaled_jacobian_after","angle_min_after","angle_max_after","max_displacement_over_hi","geometry_passed","optimizer_success","optimization_seconds","validation_seconds")
        with (args.output/"comparison.csv").open("x",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader()
            for r in records:
                a=r["coordinate_optimization"];m=r["quality"]["metrics"];t=r["timing_seconds"]
                writer.writerow({"case":r["case"],"condition_before":a["before"]["maximum_condition"],"condition_after":m["maximum_aspect_ratio"],"scaled_jacobian_after":m["minimum_scaled_jacobian"],"angle_min_after":m["minimum_angle_degrees"],"angle_max_after":m["maximum_angle_degrees"],"max_displacement_over_hi":a["after"]["maximum_displacement_over_hi"],"geometry_passed":r["quality"]["passed"],"optimizer_success":a["optimizer"]["success"] if a["optimizer"] else None,"optimization_seconds":t["optimization_and_internal_validation"],"validation_seconds":t["independent_geometry_validation"]})
    manifest(args.output,args.source,args.protocol,args.stage)


if __name__=="__main__":main()
