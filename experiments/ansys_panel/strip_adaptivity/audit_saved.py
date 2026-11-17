"""Read-only audit and terminal-cost table; preserve original solved-state table."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def read(p):return json.loads(p.read_text(encoding="utf-8"))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    with p.open("x",encoding="utf-8",newline="\n") as f:json.dump(d,f,indent=2,ensure_ascii=False,allow_nan=False);f.write("\n")


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("run",type=Path);ap.add_argument("--output",type=Path,required=True);a=ap.parse_args()
    result=subprocess.run([sys.executable,"-B",str(Path(__file__).with_name("verify_run.py")),str(a.run)],capture_output=True,text=True,encoding="utf-8")
    if result.returncode:raise ValueError(result.stdout+result.stderr)
    numerical=json.loads(result.stdout)
    s=read(a.run/"summary.json");rows=[]
    with (a.run/"comparison.csv").open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
    rejection_seconds=sum(sum(p["timing_seconds"].values()) for p in s["rejected_proposals"])
    terminal=[]
    for method in ("adaptive","uniform"):
        row=max((r for r in rows if r["method"]==method),key=lambda r:int(r["step"]))
        extra=rejection_seconds if method=="adaptive" else 0.
        terminal.append({"method":method,"last_completed_case":row["case"],"last_actual_dof":int(row["active_dof"]),"primary_goal_met":row["primary_goal_met"]=="True","strict_goal_met":row["strict_goal_met"]=="True","completed_stage_prefix_seconds":float(row["cumulative_stages_seconds"]),"subsequent_rejected_proposal_seconds":extra,"terminal_stage_seconds":float(row["cumulative_stages_seconds"])+extra,"terminal_including_operational_overhead_seconds":float(row["cumulative_with_terminal_overhead_seconds"])+extra})
    a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/"numerical_audit.json",numerical)
    save(a.output/"terminal_costs.json",{"methods":terminal,"note":"Original comparison.csv rows end at each completed solve; final rejection is additional work, never represented as an extra solve; final reporting cost is separate wall_measurement.json"})
    with (a.output/"terminal_costs.csv").open("x",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(terminal[0]),lineterminator="\n");writer.writeheader();writer.writerows(terminal)
    save(a.output/"audit_provenance.json",{"source_artifact_manifest_sha256":sha(a.run/"artifact_manifest.json"),"source_commit":"4582fefd4e66804e866d7ae9432472588d344f6e","audit_source_sha256":{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},"note":"Producer manifest source hashes are preserved, including the initial verifier; current verifier fixes only equation-line parsing to exclude Maximum wavefront. Historical produced files and manifests unchanged.","solve_performed":False,"neural_training_or_inference":False})
    save(a.output/"artifact_manifest.json",{"files":{p.name:sha(p) for p in a.output.iterdir() if p.is_file()}})
    print(json.dumps({"numerical_audit_passed":True,"terminal":terminal},indent=2))


if __name__=="__main__":main()
