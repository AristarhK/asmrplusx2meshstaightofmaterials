"""Portable tables and figures from completed measured solves; no ANSYS calls."""
from __future__ import annotations
import csv
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,data):
    with path.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(data,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write("\n")


def publish_results(output,physical,method_rows,rejections,overhead,current_wall,runtime):
    byname={r["case"]:r for r in physical}
    proposals=[json.loads(p.read_text(encoding="utf-8")) for p in sorted(output.glob("proposal_*.json"))]
    extra=sum(sum(p["timing_seconds"].values()) for p in proposals)
    actual_wall=current_wall+sum(a.get("attempt_wall_seconds",a["elapsed_attempt_seconds"]) for a in overhead["failed_attempts"])
    common_session=overhead["launch"]+overhead["close"]
    assigned=sum(r["measured_case_work_seconds"] for r in physical)+extra+common_session
    residual=max(0.,actual_wall-assigned)
    # All observed unassigned validation/API/recovery cost is shown and charged
    # equally, making the terminal standalone-method comparison conservative.
    common_overhead=common_session+residual
    rows=[]
    for row in method_rows:
        case=byname[row["case"]]
        rows.append({"method":row["method"],"step":row["step"],"case":row["case"],
            "elements":case["elements"],"active_dof":case["active_dof"],
            **{"error_"+k+"_percent":100*v for k,v in row["relative_errors"].items()},
            "primary_goal_met":row["primary_goal_met"],"strict_goal_met":row["strict_goal_met"],
            "cumulative_stages_seconds":row["cumulative_case_seconds"],
            "cumulative_with_terminal_overhead_seconds":row["cumulative_case_seconds"]+common_overhead})
    fields=list(rows[0])
    with (output/"comparison.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)
    keys=("w","C_top","C_bottom","Y_top","Y_bottom","X_top","X_bottom")
    fig,axes=plt.subplots(2,4,figsize=(15,8),layout="constrained")
    for ax,key in zip(axes.flat,keys):
        for method,color in (("adaptive","#0072b2"),("uniform","#d55e00")):
            rs=sorted((r for r in rows if r["method"]==method),key=lambda r:r["step"])
            ax.plot([r["active_dof"] for r in rs],[r[f"error_{key}_percent"] for r in rs],"o-",color=color,label=method)
        ax.axhline(1 if key=="w" else 5,color="green",ls="--",label="primary")
        ax.axhline(.2 if key=="w" else 1,color="gray",ls=":",label="strict")
        ax.set(title=key,xlabel="actual active DoF",ylabel="error vs saved reference (%)");ax.grid(alpha=.25)
    axes.flat[-1].axis("off");axes.flat[0].legend(fontsize=8)
    fig.savefig(output/"errors_vs_dof.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout="constrained")
    for method,color in (("adaptive","#0072b2"),("uniform","#d55e00")):
        rs=sorted((r for r in rows if r["method"]==method),key=lambda r:r["step"])
        time=[r["cumulative_with_terminal_overhead_seconds"] for r in rs]
        axes[0].plot(time,[r["error_w_percent"] for r in rs],"o-",label=method,color=color)
        axes[1].plot(time,[max(r[f"error_{k}_percent"] for k in keys[1:]) for r in rs],"o-",label=method,color=color)
    for ax,title,threshold in zip(axes,("central deflection","worst of six disk VM"),(1,5)):
        ax.set(title=title,xlabel="measured accumulated work + terminal overhead (s)",ylabel="error (%)")
        ax.axhline(threshold,color="green",ls="--");ax.grid(alpha=.25);ax.legend()
    fig.savefig(output/"errors_vs_time.png",dpi=160);plt.close(fig)
    solve_lines=[]
    for path in runtime.rglob("session_*.apdl"):
        solve_lines.extend(line.strip() for line in path.read_text(encoding="utf-8",errors="replace").splitlines() if re.match(r"\s*SOLVE\s*(,|$)",line,re.I))
    summary={"completed_physical_solves":[{"case":r["case"],"elements":r["elements"],"active_dof":r["active_dof"],"measured_case_work_seconds":r["measured_case_work_seconds"],"timing_seconds":r["timing_seconds"],"force_balance_relative":r["force_balance_relative"],"moment_balance_relative":r["moment_balance_relative"],"energy_balance_relative":r["energy_balance_relative"]} for r in physical],
        "comparison":rows,"assessments":method_rows,"rejected_proposals":rejections,
        "timing":{"actual_all_attempts_wall_seconds_before_final_reporting":actual_wall,"unique_physical_case_seconds":sum(r["measured_case_work_seconds"] for r in physical),"mark_remesh_including_rejections_seconds":extra,"session_launch_seconds":overhead["launch"],"session_close_seconds":overhead["close"],"unassigned_measured_validation_overhead_seconds":residual,"common_terminal_overhead_each_method_seconds":common_overhead,"accounting_residual_before_floor_seconds":actual_wall-assigned,"failed_execution_attempts":[{k:v for k,v in a.items() if k!="message"} for a in overhead["failed_attempts"]],"note":"shared0 performed once; terminal overhead is not an observed intermediate goal timestamp; costs attributed in full to both standalone methods, actual wall does not double shared work; uniform uses same recovery/export diagnostics; final plotting/publication separately timed in wall_measurement.json"},
        "solve_commands_observed":len(solve_lines),"neural_inference_performed":False,"training_performed":False,
        "reference":"Saved P1 linear h3.125 numerical comparator; audit and negative independent RST-read limitations unchanged"}
    save(output/"summary.json",summary)
    save(output/"execution_manifest.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":{p.name:sha(p) for p in Path(__file__).parent.glob("*.py")},"protocol_sha256":sha(Path(__file__).with_name("protocol.json")),"packages":{p:importlib.metadata.version(p) for p in ("numpy","scipy","matplotlib","ansys-mapdl-core")},"runtime":"Private new directory; DB/RST/full native logs excluded from Git","native_checkpoints":{p.parent.name:sha(p) for p in runtime.glob("*/solve_completed.json")}})
    print(json.dumps({"completed":True,"actual_solves":len(physical),"adaptive_solves":sum(r["case"].startswith("adaptive") for r in physical),"rows":rows},indent=2))
