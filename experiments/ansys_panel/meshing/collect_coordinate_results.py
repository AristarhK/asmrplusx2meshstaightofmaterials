"""Collect checked coordinate/native comparisons using the standard library only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def nonshape_warning_count(messages):
    count=0
    for item in messages:
        if item["severity"]!="WARNING":continue
        body=" ".join(item["message"].split())
        if item["stage"]=="check_unloaded" and body=="No constraints have been defined using the D command.":continue
        shape=re.fullmatch(r"Quadrilateral element \d+ has a pair of opposite edges that are [0-9.]+ degrees away from being parallel\. This exceeds the warning limit of [0-9.]+ degrees\.",body,re.I)
        if not shape:count+=1
    return count


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ("source","optimized","native","earlier-native","output"):
        ap.add_argument("--"+name,type=Path,required=True)
    args=ap.parse_args(); provenance={}
    for key in ("source","optimized","native","earlier_native"):
        root=getattr(args,key).resolve()
        for name,value in read(root/"artifact_manifest.json")["files"].items():
            path=(root/name).resolve()
            if root not in path.parents or sha(path)!=value:raise ValueError("Input integrity failure")
        provenance[key+"_artifact_manifest_sha256"]=sha(root/"artifact_manifest.json")
    rows=[]; records=[]
    for case in ("interior_single","edge_patch","corner_patch"):
        b=read(args.source/f"{case}.json"); o=read(args.optimized/f"{case}.json")
        n=read(args.native/f"{case}.json"); early=read(args.earlier_native/f"{case}.json")
        bm,om=b["quality"]["metrics"],o["quality"]["metrics"]
        loads=n["boundary_and_pressure"]; audit=o["coordinate_optimization"]
        warning_ids=sorted({int(m) for message in n["native_messages"] if message["stage"]=="check_loaded" for m in re.findall(r"Quadrilateral element\s+(\d+)",message["message"])})
        row={"case":case,"elements":om["elements"],"nodes":om["nodes"],
            "condition_before":bm["maximum_aspect_ratio"],"condition_after":om["maximum_aspect_ratio"],
            "scaled_jacobian_before":bm["minimum_scaled_jacobian"],"scaled_jacobian_after":om["minimum_scaled_jacobian"],
            "angle_min_before_deg":bm["minimum_angle_degrees"],"angle_max_before_deg":bm["maximum_angle_degrees"],
            "angle_min_after_deg":om["minimum_angle_degrees"],"angle_max_after_deg":om["maximum_angle_degrees"],
            "max_displacement_over_original_hi":audit["after"]["maximum_displacement_over_hi"],
            "max_displacement_mm":max(x["displacement_mm"] for x in o["moved_nodes"]),
            "marked_min_edge_before_mm":o["marked_size_before"]["edge_length_min_mm"],
            "marked_max_edge_before_mm":o["marked_size_before"]["edge_length_max_mm"],
            "marked_min_edge_after_mm":o["marked_size_after"]["edge_length_min_mm"],
            "marked_max_edge_after_mm":o["marked_size_after"]["edge_length_max_mm"],
            "native_parallel_warnings_before":b["native_parallel_deviation_warnings"],
            "native_parallel_warnings_after":n["native_shape_tests"]["Parallel Deviation"]["warnings"],
            "geometry_passed":o["quality"]["passed"],"native_shape_warning_free":n["native_shape_warning_free"],
            "native_nonshape_unexpected_warning_count":nonshape_warning_count(n["native_messages"]),
            "native_total_unexpected_warning_messages":n["native_unexpected_warning_count"],
            "native_passed":n["passed"],"load_check_passed":loads["passed"],
            "loaded_area_mm2":loads["loaded_area_mm2"],"force_z_N":loads["pressure_resultant_N"][2],
            "expected_free_dof_topology":loads["expected_free_dof_topology"],
            "optimization_including_internal_validation_seconds":o["timing_seconds"]["optimization_and_internal_validation"],
            "external_offline_validation_seconds":o["timing_seconds"]["independent_geometry_validation"],
            "native01_case_seconds":early["wall_seconds"]["case_total"],
            "native02_case_seconds":n["wall_seconds"]["case_total"],
            "native_validation_accumulated_case_seconds":early["wall_seconds"]["case_total"]+n["wall_seconds"]["case_total"]}
        rows.append(row);records.append({**row,"native_warning_element_ids_after":warning_ids,"native02_stage_seconds":n["wall_seconds"],"primary_goal_met":None,"strict_goal_met":None})
    args.output.mkdir(parents=True,exist_ok=False)
    with (args.output/"comparison.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    save(args.output/"comparison.json",{"cases":records,"accuracy_goals":"Not assessed; no solve","eligible_for_adaptive_solve":False})
    save(args.output/"manifest.json",{**provenance,"collector_sha256":sha(Path(__file__)),"native01_total_wall_seconds":read(args.earlier_native/"summary.json")["total_wall_seconds"],"native02_total_wall_seconds":read(args.native/"summary.json")["total_wall_seconds"],"classification_note":"Recomputed nonshape warnings from original messages: native02's legacy nonshape counter also counted repeated Parallel Deviation messages. Native shape tables and all acceptance outcomes are unchanged.","no_optimization_or_native_rerun":True})
    save(args.output/"artifact_manifest.json",{"files":{p.name:sha(p) for p in args.output.iterdir() if p.is_file()}})
    print(json.dumps({"rows":len(rows),"all_geometries_passed":all(r["geometry_passed"] for r in rows),"all_native_checks_passed":all(r["native_passed"] for r in rows)}))


if __name__=="__main__":main()
