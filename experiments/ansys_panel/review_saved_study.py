"""Audit a completed panel study from saved arrays/listings; no ANSYS imports.

Writes a NEW audit directory. Historical measurements and processing modules
are never modified. Metric transitions are recomputed independently of analysis.py.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_exports import audit_case
from review_models import inspect_model
from review_quadrature import cartesian_disk


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def csv_save(path, rows):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def change(old, new):
    difference=abs(new-old)
    return difference/abs(new) if new else (0.0 if not difference else None)


def vm_transitions(sequences):
    rows=[]
    for support, seq in sequences.items():
        for old,new in zip(seq,seq[1:]):
            for surface in ("top","bottom"):
                for name in new["disks"][surface]:
                    a,b=old["disks"][surface][name],new["disks"][surface][name]
                    relative=change(a["mean_von_mises"],b["mean_von_mises"])
                    rows.append({"support":support,"surface":surface,"disk":name,"coarse_h":old["h"],"fine_h":new["h"],"old_vm_mpa":a["mean_von_mises"],"new_vm_mpa":b["mean_von_mises"],"relative_difference":relative,"percent_difference":100*relative,"limit_percent":1.0,"passed":relative<=.01})
    return rows


def references(sequences):
    rows=[]
    for support,seq in sequences.items():
        metrics={k:(lambda r,k=k:r[k],limit) for k,limit in (("central_deflection",.002),("max_abs_uz",.002),("strain_energy",.005))}
        for surface in ("top","bottom"):
            for name in seq[-1]["disks"][surface]:
                metrics[f"{surface}/{name}/mean_von_mises"]=(lambda r,s=surface,n=name:r["disks"][s][n]["mean_von_mises"],.01)
                for component in ("mean_sx","mean_sy","mean_txy"):
                    metrics[f"{surface}/{name}/{component}"]=(lambda r,s=surface,n=name,k=component:r["disks"][s][n][k],None)
            for extreme in ("interior_maximum_von_mises","global_maximum_von_mises"):
                metrics[f"{surface}/{extreme}"]=(lambda r,s=surface,k=extreme:r["stress_extrema"][s][k]["von_mises"],None)
        for metric,(value,limit) in metrics.items():
            component=metric.endswith(("/mean_sx","/mean_sy","/mean_txy"))
            if component:
                surface,name,_=metric.split("/")
                scales=[max(abs(value(a)),abs(value(b)),.001*max(abs(a["disks"][surface][name]["mean_von_mises"]),abs(b["disks"][surface][name]["mean_von_mises"]))) for a,b in zip(seq,seq[1:])]
                transitions=[abs(value(b)-value(a))/scale if scale else 0. for (a,b),scale in zip(zip(seq,seq[1:]),scales)]
            else:
                transitions=[change(value(a),value(b)) for a,b in zip(seq,seq[1:])]
            last=transitions[-2:]
            passes=[v is not None and v<=limit for v in last] if limit is not None else [None,None]
            ok=all(passes) if limit is not None else None
            rows.append({"support":support,"metric":metric,"last_value":value(seq[-1]),"limit_percent":None if limit is None else 100*limit,"penultimate_change_percent":None if last[0] is None else 100*last[0],"last_change_percent":None if last[1] is None else 100*last[1],"penultimate_absolute_change":abs(value(seq[-2])-value(seq[-3])),"last_absolute_change":abs(value(seq[-1])-value(seq[-2])),"change_normalization":"max(abs(new),abs(old),0.001*max(old/new disk VM)); retain absolute MPa changes" if component else "absolute new value","two_transitions_passed":ok,"reference_h_mm":seq[-1]["h"] if ok else None,"interpretation":"stabilized numerical comparator for this metric only" if ok else ("no predefined threshold; no reference claim" if limit is None else "two required transitions not met")})
    return rows


def nonlinear_comparison(records):
    nonlinear=next(r for r in records if r["nonlinear"])
    linear=next(r for r in records if r["support"]=="P1" and not r["nonlinear"] and r["h"]==nonlinear["h"])
    rows=[]
    for surface in ("top","bottom"):
        for name in linear["disks"][surface]:
            a,b=linear["disks"][surface][name],nonlinear["disks"][surface][name]
            diff=abs(b["mean_von_mises"]-a["mean_von_mises"])
            rows.append({"surface":surface,"disk":name,"h_mm":linear["h"],"linear_vm_mpa":a["mean_von_mises"],"nonlinear_vm_mpa":b["mean_von_mises"],"absolute_difference_mpa":diff,"percent_nonlinear_denominator":100*diff/abs(b["mean_von_mises"]),"percent_linear_denominator":100*diff/abs(a["mean_von_mises"]),"limit_percent":2.0,"passed_both_denominators":max(diff/abs(a["mean_von_mises"]),diff/abs(b["mean_von_mises"]))<=.02})
    return rows


def tables(output, means, transitions, nonlinear, reference):
    lines=["# Saved panel study: numerical review", "", "VM means in MPa. C=(500,300), Y=(500,48), X=(48,300); radius8 mm.", ""]
    for support in ("P0","P1"):
        lines += [f"## {support}: original linear means", "", "| h mm | TOP C | TOP Y | TOP X | BOTTOM C | BOTTOM Y | BOTTOM X |", "|---:|---:|---:|---:|---:|---:|---:|"]
        for h in sorted({r["h"] for r in means if r["support"]==support and not r["nonlinear"]},reverse=True):
            row=[next(r["mean_von_mises"] for r in means if r["support"]==support and not r["nonlinear"] and r["h"]==h and r["surface"]==s and r["disk"]==d) for s in ("top","bottom") for d in ("center","near_y_edge","near_x_edge")]
            lines.append("| "+f"{h:g} | "+" | ".join(f"{v:.9f}" for v in row)+" |")
        lines.append("")
    lines += ["## Last two mesh transitions, percentages", "", "| Support | Surface | Disk | 12.5→6.25 % | 6.25→3.125 % | Both ≤1% |", "|---|---|---|---:|---:|---|"]
    for support in ("P0","P1"):
        for surface in ("top","bottom"):
            for disk in ("center","near_y_edge","near_x_edge"):
                selected=[r for r in transitions if r["support"]==support and r["surface"]==surface and r["disk"]==disk and r["fine_h"]<=6.25]
                lines.append(f"| {support} | {surface} | {disk} | {selected[0]['percent_difference']:.6f} | {selected[1]['percent_difference']:.6f} | {'yes' if all(r['passed'] for r in selected) else 'NO'} |")
    lines += ["", "## P1 linear vs nonlinear on h=3.125 mm", "", "| Surface | Disk | Linear MPa | Nonlinear MPa | Δ % / NL | Δ % / linear | ≤2% |", "|---|---|---:|---:|---:|---:|---|"]
    for r in nonlinear:
        lines.append(f"| {r['surface']} | {r['disk']} | {r['linear_vm_mpa']:.9f} | {r['nonlinear_vm_mpa']:.9f} | {r['percent_nonlinear_denominator']:.6f} | {r['percent_linear_denominator']:.6f} | {'yes' if r['passed_both_denominators'] else 'NO'} |")
    lines += ["", "## References by predefined metric", "", "| Support | Metric | Earlier Δ % | Last Δ % | Limit % | Stable twice |", "|---|---|---:|---:|---:|---|"]
    for r in reference:
        if r["limit_percent"] is not None:
            lines.append(f"| {r['support']} | {r['metric']} | {r['penultimate_change_percent']:.6f} | {r['last_change_percent']:.6f} | {r['limit_percent']:g} | {'yes' if r['two_transitions_passed'] else 'NO'} |")
    (output/"tables.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    source=args.input.resolve(); output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    historical={str(p.relative_to(source)):digest(p) for p in source.rglob("*") if p.is_file()}
    records=json.loads((source/"measurements.json").read_text(encoding="utf-8"))
    provenance=json.loads((source/"solve_provenance.json").read_text(encoding="utf-8"))
    config=provenance[0]["manifest"]["config"]
    old_audit=json.loads((source/"export_audit.json").read_text(encoding="utf-8"))
    export_results=[audit_case(r,config) for r in records]
    save(output/"export_audit_repeat.json",{"scope":"Same audit_case function as audit_exports.py CLI; raw arrays and solver listing, no ANSYS imports or solves","passed":all(r["passed"] for r in export_results),"cases":export_results,"measurements_sha256":digest(source/"measurements.json"),"audit_source_sha256":digest(Path(__file__).with_name("audit_exports.py"))})
    model_results=[inspect_model(r,config) for r in records]
    save(output/"actual_models.json",model_results)
    sequences={s:sorted((r for r in records if r["support"]==s and not r["nonlinear"]),key=lambda r:-r["h"]) for s in ("P0","P1")}
    means=[]; quadrature=[]
    for r in records:
        with np.load(Path(r["runtime_directory"])/"fields.npz",allow_pickle=False) as raw:
            for surface in ("top","bottom"):
                for name,center in config["disk_centers"].items():
                    saved=r["disks"][surface][name]
                    means.append({"support":r["support"],"case":r["case"],"nonlinear":r["nonlinear"],"h":r["h"],"surface":surface,"disk":name,**{k:saved[k] for k in ("mean_sx","mean_sy","mean_txy","mean_von_mises")}})
                    low=cartesian_disk(raw["xs"],raw["ys"],raw[surface],center,config["disk_radius"],theta_order=16,y_order=8)
                    high=cartesian_disk(raw["xs"],raw["ys"],raw[surface],center,config["disk_radius"],theta_order=32,y_order=16)
                    comparison={}
                    for k in ("mean_sx","mean_sy","mean_txy","mean_von_mises"):
                        scale=abs(high[k]) if k=="mean_von_mises" else max(abs(high[k]),abs(saved[k]),.001*abs(high["mean_von_mises"]))
                        delta=abs(high[k]-saved[k]); convergence=abs(high[k]-low[k])
                        comparison[k]={"saved":saved[k],"cartesian":high[k],"absolute_difference_mpa":delta,"normalization_scale_mpa":scale,"normalized_difference":delta/scale if scale else 0.,"independent_order_change_mpa":convergence,"passed_saved_tolerance":delta<=1e-9+5e-4*scale,"independent_integration_stable":convergence<=1e-10+1e-6*scale}
                    quadrature.append({"case":r["case"],"surface":surface,"disk":name,"low":low,"high":high,"comparison":comparison,"passed":high["area_relative_error"]<1e-10 and all(v["passed_saved_tolerance"] and v["independent_integration_stable"] for v in comparison.values())})
        print(f"Reviewed {r['case']}",flush=True)
    transitions=vm_transitions(sequences); reference=references(sequences); nonlinear=nonlinear_comparison(records)
    csv_save(output/"disk_means.csv",means); csv_save(output/"mesh_transitions.csv",transitions)
    csv_save(output/"nonlinear_comparison.csv",nonlinear); csv_save(output/"reference_by_metric.csv",reference)
    save(output/"quadrature_crosscheck.json",quadrature)
    tables(output,means,transitions,nonlinear,reference)
    after={str(p.relative_to(source)):digest(p) for p in source.rglob("*") if p.is_file()}
    summary={"audited_utc":datetime.now(timezone.utc).isoformat(),"input_directory":str(source),"historical_artifacts_unchanged":historical==after,"historical_export_audit_passed":old_audit["passed"],"fresh_export_audit_passed":all(r["passed"] for r in export_results),"actual_models_passed":all(r["passed"] for r in model_results),"independent_quadrature_passed":all(r["passed"] for r in quadrature),"maximum_saved_vs_cartesian_vm_relative_difference":max(r["comparison"]["mean_von_mises"]["normalized_difference"] for r in quadrature),"maximum_independent_quadrature_area_error":max(r["high"]["area_relative_error"] for r in quadrature),"records":len(records),"disks_checked":len(quadrature),"reference_by_metric":reference,"nonlinear_comparison":nonlinear,"source_files_sha256":{p.name:digest(p) for p in (Path(__file__),Path(__file__).with_name("review_models.py"),Path(__file__).with_name("review_quadrature.py"),Path(__file__).with_name("audit_exports.py"))}}
    save(output/"review.json",summary)
    save(output/"manifest.json",{"historical_source_sha256":historical,"files":{str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file()}})
    ok=all(summary[k] for k in ("historical_artifacts_unchanged","historical_export_audit_passed","fresh_export_audit_passed","actual_models_passed","independent_quadrature_passed"))
    print(json.dumps({"passed":ok,"records":len(records),"disks_checked":len(quadrature),"max_saved_vs_cartesian_vm_relative":summary["maximum_saved_vs_cartesian_vm_relative_difference"]},indent=2))
    return 0 if ok else 1


if __name__=="__main__":
    raise SystemExit(main())
