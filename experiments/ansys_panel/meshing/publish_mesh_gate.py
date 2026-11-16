"""Read-only native export audit and compact publication; never starts MAPDL.

All source outputs remain immutable. A new output directory is mandatory.
The source execution's comma-only CDB reader missed native DBLOCK/SFEBLOCK;
this program corrects only that postprocessing, using original hashed exports.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import re
import shutil

from cdb_loads import parse_cdb_loads
from run_mesh_gate import digest, load_check, plot_mesh, save_json
from validate_mesh import validate_mesh

CASES=("interior_single","edge_patch","corner_patch","all_elements")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_hash(path, expected):
    if digest(path)!=expected:
        raise ValueError(f"Source hash mismatch: {Path(path).name}")


def compact_loads(commands):
    """Lossless groups for all accepted real D/SFE records, preserving multiplicity."""
    displacement={}; pressure={}; other=[]
    for f in commands:
        if f[0]=="D":
            key=(f[2],float(f[3]),float(f[4] or 0) if len(f)>4 else 0.)
            displacement.setdefault(key,[]).append(int(f[1]))
        elif f[0]=="SFE":
            key=(int(f[2]),f[3],f[4],tuple(float(v) for v in f[5:]))
            pressure.setdefault(key,[]).append(int(f[1]))
        else:other.append(f)
    return {"displacement_groups":[{"dof":k[0],"value":k[1],"imaginary_value":k[2],"node_ids":v} for k,v in sorted(displacement.items())],
            "pressure_groups":[{"face":k[0],"label":k[1],"kval":k[2],"four_values_N_per_mm2":list(k[3]),"element_ids":v} for k,v in sorted(pressure.items())],"other_records":other,
            "note":"Native DBLOCK/SFEBLOCK decoded; imaginary pressure records required zero by parser"}


def expand_loads(evidence):
    result=[]
    for g in evidence["displacement_groups"]:
        result.extend([["D",str(n),g["dof"],str(g["value"]),str(g["imaginary_value"])] for n in g["node_ids"]])
    for g in evidence["pressure_groups"]:
        result.extend([["SFE",str(e),str(g["face"]),g["label"],g["kval"],*[str(v) for v in g["four_values_N_per_mm2"]]] for e in g["element_ids"]])
    return result+evidence["other_records"]


def portable_response(text):
    return re.sub(r"(\*\*\*\s+(?:NOTE|WARNING|ERROR)\s+\*\*\*)[^\r\n]*",r"\1",text)


def quality(data, config):
    return validate_mesh(data["node_ids"],data["xyz_mm"],data["element_ids"],data["connectivity_node_ids"],data["attributes"],config)


def cell_signatures(data):
    points={int(n):tuple(round(float(v),8) for v in p) for n,p in zip(data["node_ids"],data["xyz_mm"])}
    return {tuple(sorted(points[n] for n in ns)) for ns in data["connectivity_node_ids"]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--runtime",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    original_manifest=read_json(args.source/"manifest.json")
    for name,sha in read_json(args.source/"artifact_manifest.json")["files"].items():
        check_hash(args.source/name,sha)
    config=original_manifest["config"]
    # Quality limits are not parameters of postprocessing and cannot be loosened.
    required={"aspect_ratio_max":4.,"angle_min_degrees":30.,"angle_max_degrees":150.,"scaled_jacobian_min":.2}
    if config["quality_limits"]!=required:raise ValueError("Unexpected quality thresholds")
    base=read_json(args.source/"base_mesh.json")
    base_cdb=args.runtime/"base/unloaded.cdb"
    check_hash(base_cdb,original_manifest["base_unloaded_cdb_sha256"])
    if parse_cdb_loads(base_cdb):raise ValueError("Base is not unloaded")
    base_quality=quality(base,config)
    if not base_quality["passed"]:raise ValueError("Base failed independent geometry audit")
    args.output.mkdir(parents=True,exist_ok=False)
    figures=args.output/"figures";figures.mkdir()
    inputs=args.output/"inputs";inputs.mkdir()
    shutil.copyfile(args.runtime/"base/base_model.apdl",inputs/"base_model.apdl")
    save_json(args.output/"execution_manifest.json",original_manifest)
    save_json(args.output/"base_mesh.json",base)
    save_json(args.output/"base_quality.json",base_quality)
    plot_mesh(base,set(),figures/"base.png","Base: 240 SHELL181; h=50 mm; unloaded")
    records=[]
    for name in CASES:
        original=read_json(args.source/f"{name}.json")
        work=args.runtime/name
        for filename,sha in original["raw_input_sha256"].items():
            check_hash(work/filename,sha)
        data=read_json(args.source/f"{name}_mesh.json")
        before=parse_cdb_loads(work/"before_refine.cdb")
        loaded=parse_cdb_loads(work/"after_loading.cdb")
        audit=load_check(loaded,data,config)
        q=quality(data,config)
        if q!=original["quality"]:raise ValueError("Geometry audit changed since execution")
        response=(work/"refine_response.txt").read_text(encoding="utf-8")
        sanitized=portable_response(response)
        record=dict(original)
        record["quality"]=q
        unchanged=len(cell_signatures(base)&cell_signatures(data))
        record["original_elements_unchanged_geometry"]=unchanged
        record["original_elements_replaced_geometry"]=len(base["element_ids"])-unchanged
        record["unloaded_before_refine"]=not before
        record["before_load_commands"]=before
        record["boundary_and_pressure"]=audit
        record["native_warning_or_error_lines"]=[s.strip() for s in sanitized.splitlines() if re.search(r"\*\*\*\s+(?:ERROR|WARNING)\s+\*\*\*",s,re.I)]
        parallel=re.search(r"Parallel Deviation\s+\d+\s+(\d+)\s+(\d+)",response)
        record["native_parallel_deviation_warnings"]=int(parallel[1]) if parallel else 0
        record["native_parallel_deviation_errors"]=int(parallel[2]) if parallel else 0
        record["passed"]=bool(q["passed"] and not before and record["native_refinement_error"] is None and record["element_increase"]>0 and record["element_section_settings_preserved"] and audit["passed"] and not record["native_warning_or_error_lines"])
        record["failure_reasons"]=q["failures"]+(["native_shape_warning"] if record["native_warning_or_error_lines"] else [])+([] if audit["passed"] else ["load_validation"])
        record["native_refinement_response_sha256"]=digest(work/"refine_response.txt")
        evidence=compact_loads(loaded)
        if load_check(expand_loads(evidence),data,config)!=audit:
            raise ValueError("Compact load evidence round trip changed validation")
        save_json(args.output/f"{name}_loads.json",evidence)
        save_json(args.output/f"{name}_mesh.json",data)
        save_json(args.output/f"{name}.json",record)
        (args.output/f"{name}_native_response.txt").write_text(sanitized,encoding="utf-8",newline="\n")
        shutil.copyfile(work/"mark.apdl",inputs/f"{name}_mark.apdl")
        shutil.copyfile(work/"reapply_p1.apdl",inputs/f"{name}_reapply_p1.apdl")
        plot_mesh(base,set(record["marked_element_ids"]),figures/f"{name}_before.png",f"{name}: before; orange=marked ({record['marked_count']})")
        plot_mesh(data,set(),figures/f"{name}_after.png",f"{name}: after; {record['elements_after']} elements; {'PASS' if record['passed'] else 'FAIL'}")
        records.append(record)
    forbidden=[]
    for log in args.runtime.rglob("session.apdl"):
        for line in log.read_text(encoding="utf-8",errors="replace").splitlines():
            if re.match(r"\s*(SOLVE|LSSOLVE|MFSOLVE|PSOLVE)\s*(,|$)",line,re.I):forbidden.append(line)
    solutions=[p.name for p in args.runtime.rglob("*") if p.is_file() and p.suffix.lower() in (".rst",".rth",".full",".mntr")]
    summary={"case_count":len(records),"all_cases_passed":all(r["passed"] for r in records),"eligible_for_adaptive_solves":all(r["passed"] for r in records),"solve_commands":forbidden,"solution_files":solutions,"solve_performed":bool(forbidden or solutions),"indicator_computed":False,"training_performed":False,"cases":records}
    save_json(args.output/"summary.json",summary)
    save_json(args.output/"publication_manifest.json",{"source_run_id_utc":original_manifest["run_id_utc"],"execution_manifest_sha256":digest(args.source/"manifest.json"),"postprocessing":"Correct native DBLOCK/SFEBLOCK decoding of original immutable CDB; no model or mesh rerun", "source_execution_artifact_manifest_sha256":digest(args.source/"artifact_manifest.json"),"postprocessor_source_sha256":{p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},"private_runtime":"Original CDB, DB, APDL session logs remain local; only hashes and selected portable evidence published"})
    fields=["case","marked_count","elements_before","elements_after","nodes_after","element_increase","element_growth_factor","max_aspect","min_scaled_jacobian","native_shape_warnings","boundary_nodes","expected_free_dof_topology","load_check_passed","force_z_N","passed"]
    with (args.output/"comparison.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in records:
            q,b=r["quality"]["metrics"],r["boundary_and_pressure"]
            writer.writerow({**{k:r[k] for k in fields if k in r},"max_aspect":q["maximum_aspect_ratio"],"min_scaled_jacobian":q["minimum_scaled_jacobian"],"native_shape_warnings":r["native_parallel_deviation_warnings"],"boundary_nodes":b["boundary_nodes"],"expected_free_dof_topology":b["expected_free_dof_topology"],"load_check_passed":b["passed"],"force_z_N":b["pressure_resultant_N"][2]})
    save_json(args.output/"artifact_manifest.json",{"files":{p.relative_to(args.output).as_posix():digest(p) for p in args.output.rglob("*") if p.is_file()}})
    print(json.dumps({k:v for k,v in summary.items() if k!="cases"},indent=2))
    return 1 if summary["solve_performed"] else 0


if __name__=="__main__":raise SystemExit(main())
