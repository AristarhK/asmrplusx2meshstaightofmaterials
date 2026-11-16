"""Independent saved-coordinate audit, no optimizer execution and no MAPDL."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

from check_optimized_native import compare_mesh
from coordinate_diagnostics import diagnose
from optimize_coordinates import evaluate_candidate, prepare_problem
from publish_mesh_gate import expand_loads, quality, read_json
from run_coordinate_study import CASES, check_protocol, verify_artifacts
from run_mesh_gate import digest, load_check


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,default=Path(__file__).parent/"runs/20261116_mesh_gate01")
    ap.add_argument("--localization",type=Path,required=True)
    ap.add_argument("--optimized",type=Path,required=True)
    ap.add_argument("--native",type=Path)
    args=ap.parse_args()
    for root in (args.source,args.localization,args.optimized):verify_artifacts(root)
    config=read_json(args.source/"execution_manifest.json")["config"]
    protocol=read_json(args.localization/"protocol.json");check_protocol(protocol)
    protocol_sha=digest(args.localization/"protocol.json")
    if protocol_sha!=digest(args.optimized/"protocol.json"):raise ValueError("Protocol changed between stages")
    for root in (args.localization,args.optimized):
        m=read_json(root/"manifest.json")
        if m["protocol_sha256"]!=protocol_sha or m["solve_performed"] or m["indicator_computed"] or m["training_performed"]:
            raise ValueError("Unexpected stage provenance")
    base=read_json(args.source/"base_mesh.json")
    coordinates={int(n):np.array(p[:2]) for n,p in zip(base["node_ids"],base["xyz_mm"])}
    cells={int(e):np.array([coordinates[n] for n in ns]) for e,ns in zip(base["element_ids"],base["connectivity_node_ids"])}
    statuses={}
    for name in CASES:
        original=read_json(args.source/f"{name}_mesh.json")
        data=read_json(args.optimized/f"{name}_mesh.json")
        record=read_json(args.optimized/f"{name}.json")
        source_record=read_json(args.source/f"{name}.json")
        marked=source_record["marked_element_ids"]
        problem=prepare_problem(original,config,np.array([cells[e] for e in marked]))
        evaluation=evaluate_candidate(problem,np.array(data["xyz_mm"])[:,:2])
        if evaluation!=record["coordinate_optimization"]["after"] or not evaluation["passed"]:
            raise ValueError(f"Frozen restrictions not reproduced: {name}")
        for key in ("node_ids","element_ids","connectivity_node_ids","attributes","native_settings"):
            if data[key]!=original[key]:raise ValueError(f"Mesh topology/settings changed: {key}")
        if not np.array_equal(np.array(data["xyz_mm"])[:,2],np.array(original["xyz_mm"])[:,2]):
            raise ValueError("Z coordinates changed")
        q=quality(data,config)
        if q!=record["quality"]:raise ValueError("Full independent geometry gate disagrees")
        a=record["coordinate_optimization"]
        if a["optimizer"] and a["optimizer"]["iterations"]>protocol["maxiter"]:
            raise ValueError("Iteration cap exceeded")
        before=diagnose(original,base,marked,config)
        after=diagnose(data,base,marked,config,reference_mesh=original)
        if before!=read_json(args.localization/f"{name}_diagnostics.json") or after!=read_json(args.optimized/f"{name}_diagnostics.json"):
            raise ValueError("Element diagnostics do not reproduce")
        if before["marked_region"]["size_statistics"]!=after["marked_region"]["size_statistics"]:
            raise ValueError("Protected-region resolution changed")
        statuses[name]={"geometry_passed":q["passed"],"maximum_condition":q["metrics"]["maximum_aspect_ratio"],"max_displacement_over_hi":evaluation["maximum_displacement_over_hi"]}
    if args.native:
        verify_artifacts(args.native)
        native=read_json(args.native/"summary.json")
        if native["solve_performed"] or native["solve_commands"] or native["solution_files"]:
            raise ValueError("Unexpected native solve evidence")
        for record in native["cases"]:
            name=record["case"]
            if record.get("status")=="skipped_offline_gate_failed" or record.get("skipped") or record.get("skipped_offline_geometry"):
                if statuses[name]["geometry_passed"]:raise ValueError("Passing candidate skipped")
                continue
            if not statuses[name]["geometry_passed"]:raise ValueError("Failed candidate sent to MAPDL")
            if "native_quality" not in record:
                statuses[name]["native_passed"]=False
                continue
            data=read_json(args.native/f"{name}_mesh.json")
            expected=read_json(args.optimized/f"{name}_mesh.json")
            if quality(data,config)!=record["native_quality"] or compare_mesh(expected,data)!=record["round_trip"]:
                raise ValueError("Native mesh evidence disagrees")
            loads=load_check(expand_loads(read_json(args.native/f"{name}_loads.json")),data,config)
            if loads!=record["boundary_and_pressure"]:raise ValueError("Native load evidence disagrees")
            table=record["native_shape_tests"]
            shape_ok=all(row["warnings"]==row["errors"]==0 for row in table.values())
            if "native_shape_warning_free" in record and record["native_shape_warning_free"]!=shape_ok:
                raise ValueError("Shape warning flag disagrees with native table")
            if any(row["tested"]!=len(data["element_ids"]) for row in table.values()):
                raise ValueError("Incomplete native shape coverage")
            if record["passed"] and (not shape_ok or not loads["passed"] or not record["round_trip"]["passed"]):
                raise ValueError("Native acceptance contradicts required gates")
            statuses[name]["native_passed"]=record["passed"]
            statuses[name]["native_parallel_warnings"]=record["native_shape_tests"]["Parallel Deviation"]["warnings"]
    print(json.dumps({"saved_audit_consistent":True,"optimizer_rerun":False,"solve_performed":False,"cases":statuses},indent=2))


if __name__=="__main__":main()
