"""Verify portable mesh-only evidence without MAPDL; failed cases stay failed."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from publish_mesh_gate import CASES, check_hash, expand_loads, quality, read_json
from run_mesh_gate import digest, load_check


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_directory",type=Path)
    args=ap.parse_args(); root=args.run_directory.resolve()
    manifest=read_json(root/"artifact_manifest.json")
    for name,sha in manifest["files"].items():
        path=(root/name).resolve()
        if root not in path.parents:raise ValueError("Artifact path escapes run directory")
        check_hash(path,sha)
    execution=read_json(root/"execution_manifest.json")
    config=execution["config"]
    base=read_json(root/"base_mesh.json")
    if quality(base,config)!=read_json(root/"base_quality.json"):
        raise ValueError("Base geometry evidence does not reproduce")
    summary=read_json(root/"summary.json")
    if summary["solve_performed"] or summary["solve_commands"] or summary["solution_files"]:
        raise ValueError("Unexpected solution evidence")
    if summary["indicator_computed"] or summary["training_performed"]:
        raise ValueError("Unexpected indicator/training")
    statuses={}
    for name,expected in zip(CASES,summary["cases"]):
        record=read_json(root/f"{name}.json")
        if record!=expected:raise ValueError("Summary disagrees with individual record")
        data=read_json(root/f"{name}_mesh.json")
        q=quality(data,config)
        loads=load_check(expand_loads(read_json(root/f"{name}_loads.json")),data,config)
        if q!=record["quality"] or loads!=record["boundary_and_pressure"]:
            raise ValueError(f"Saved geometry/load validation disagrees: {name}")
        settings_ok=data["native_settings"]==base["native_settings"]
        if settings_ok!=record["element_section_settings_preserved"]:
            raise ValueError("Native settings preservation disagrees")
        passed=bool(q["passed"] and loads["passed"] and settings_ok and record["unloaded_before_refine"] and record["native_refinement_error"] is None and record["element_increase"]>0 and not record["native_warning_or_error_lines"])
        if passed!=record["passed"]:raise ValueError("Saved acceptance status does not follow checks")
        statuses[name]=passed
    if summary["eligible_for_adaptive_solves"]!=all(statuses.values()):
        raise ValueError("Adaptive eligibility disagrees")
    print(json.dumps({"portable_audit_consistent":True,"artifact_hashes":len(manifest["files"]),"case_acceptance":statuses,"eligible_for_adaptive_solves":summary["eligible_for_adaptive_solves"]},indent=2))


if __name__=="__main__":main()
