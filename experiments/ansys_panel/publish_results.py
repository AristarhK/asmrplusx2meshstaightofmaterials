"""Publish compact measurements and scientific figures; never launches ANSYS."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from analysis import transition_convergence, two_transition_convergence
from run_panel import sha, write_json


def csv_file(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def publish(source, output, audit_path):
    records = json.loads((source / "measurements.json").read_text(encoding="utf-8"))
    navier = json.loads((source / "navier.json").read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit["passed"] or audit["measurements_sha256"] != sha(source/"measurements.json"):
        raise ValueError("Publication requires a passing audit of these exact measurements")
    output.mkdir(parents=True, exist_ok=False)
    figures = output / "figures"
    figures.mkdir()
    sequences = {s:sorted((r for r in records if r["support"] == s and not r["nonlinear"]), key=lambda r:-r["h"]) for s in ("P0","P1")}
    convergence = {s:two_transition_convergence(seq) for s,seq in sequences.items()}
    nonlinear = next((r for r in records if r["nonlinear"]), None)
    nonlinear_comparison = None
    if nonlinear:
        linear = next(r for r in sequences["P1"] if r["h"] == nonlinear["h"])
        nonlinear_comparison = transition_convergence(linear, nonlinear)
        differences = [nonlinear_comparison["values"][k]["relative_difference"] for k in ("central_deflection","max_abs_uz")]
        differences += [v["changes"]["mean_von_mises"]["normalized_difference"] for surface in nonlinear_comparison["disks"].values() for v in surface.values()]
        nonlinear_comparison["maximum_deflection_or_disk_vm_relative_difference"] = max(differences)
        nonlinear_comparison["linear_approximation_2percent_check"] = max(differences) <= .02
    summary = {"convergence":convergence, "navier":navier, "nonlinear":nonlinear_comparison,
               "maximum_force_imbalance":max(r["force_balance_relative"] for r in records),
               "maximum_moment_imbalance":max(r["moment_balance_relative"] for r in records),
               "maximum_linear_energy_imbalance":max(r["linear_energy_relative_check"] for r in records if not r["nonlinear"]),
               "all_disk_quadratures_converged":all(d["quadrature_converged"] for r in records for surf in r["disks"].values() for d in surf.values()),
               "maximum_global_vm":max(v["global_maximum_von_mises"]["von_mises"] for r in records for v in r["stress_extrema"].values()),
               "p0_navier_comparison":[{"h":r["h"],"shell_deflection_mm":r["central_deflection"],"navier_deflection_mm":navier["central_deflection"],"signed_relative_difference":r["central_deflection"]/navier["central_deflection"]-1} for r in sequences["P0"]]}
    write_json(output/"summary.json", summary)
    write_json(output/"measurements.json", records)
    write_json(output/"export_audit.json", audit)
    write_json(output/"navier.json", navier)
    for record in records:
        write_json(output/f"{record['case']}.json", record)
    # Immutable source manifests retain code hashes for each solve stage.
    manifests=[]
    current=source
    while True:
        manifest=json.loads((current/"manifest.json").read_text(encoding="utf-8"))
        session=json.loads((current/"session.json").read_text(encoding="utf-8"))
        manifests.append({"stage_directory":str(current), "manifest":manifest, "observed_session":session})
        previous=manifest.get("previous_stage")
        if not previous:
            break
        current=Path(previous)
    write_json(output/"solve_provenance.json", manifests)
    basic=[]
    for r in records:
        row={k:r[k] for k in ("case","support","nonlinear","h","nx","ny","elements","nodes","active_dof","central_deflection","max_abs_uz","strain_energy","artificial_energy","force_balance_relative","moment_balance_relative")}
        row.update({f"reaction_{axis}":r["reaction_force"][i] for i,axis in enumerate("xyz")})
        row.update({f"time_{k}_seconds":v for k,v in r["timing_seconds"].items()})
        basic.append(row)
    csv_file(output/"comparison.csv", basic)
    disk_rows=[]
    for r in records:
        for surface, disks in r["disks"].items():
            for name,d in disks.items():
                disk_rows.append({"case":r["case"],"surface":surface,"disk":name,**{k:d[k] for k in ("mean_sx","mean_sy","mean_txy","mean_von_mises","quadrature_converged")},"quadrature_samples":d["quadrature"]["sample_count"]})
    csv_file(output/"disk_stresses.csv", disk_rows)
    fig, axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,(s,seq) in zip(axes,sequences.items()):
        ax.plot([r["h"] for r in seq],[r["central_deflection"] for r in seq],"o-",label="SHELL181 center / maximum")
        if s=="P0":
            ax.axhline(navier["central_deflection"],color="black",ls=":",label="Navier Kirchhoff-Love")
        ax.set(xscale="log",xlabel="Mesh size h (mm)",ylabel="Deflection (mm)",title=s)
        ax.set_xticks([r["h"] for r in seq], labels=[f"{r['h']:g}" for r in seq]); ax.minorticks_off()
        ax.invert_xaxis(); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(figures/"deflection_convergence.png",dpi=180); plt.close(fig)
    fig, axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for ax,(s,seq) in zip(axes,sequences.items()):
        for color,name in zip(("tab:blue","tab:orange","tab:green"),("center","near_y_edge","near_x_edge")):
            for face,style in (("top","-"),("bottom","--")):
                ax.plot([r["h"] for r in seq],[r["disks"][face][name]["mean_von_mises"] for r in seq],style+"o",color=color,label=f"{name} {face}",ms=3)
        ax.set(xscale="log",xlabel="Mesh size h (mm)",ylabel="Disk mean von Mises (MPa)",title=s)
        ax.set_xticks([r["h"] for r in seq], labels=[f"{r['h']:g}" for r in seq]); ax.minorticks_off()
        ax.invert_xaxis(); ax.grid(alpha=.25); ax.legend(fontsize=7,ncol=2)
    fig.savefig(figures/"stress_convergence.png",dpi=180); plt.close(fig)
    coarse=sequences["P0"][0]
    with np.load(Path(coarse["runtime_directory"])/"fields.npz") as f:
        xs,ys=f["xs"],f["ys"]
    fig,ax=plt.subplots(figsize=(9,5.7),layout="constrained")
    ax.vlines(xs,ys[0],ys[-1],color=".4",lw=.5); ax.hlines(ys,xs[0],xs[-1],color=".4",lw=.5)
    for center in ((500,300),(500,48),(48,300)):
        ax.add_patch(Circle(center,8,color="red",fill=False,lw=1.5))
    ax.set(aspect="equal",xlabel="x (mm)",ylabel="y (mm)",title=f"Initial full panel mesh: h={coarse['h']:g} mm; {coarse['elements']} SHELL181; +Z pressure")
    fig.savefig(figures/"initial_mesh.png",dpi=180); plt.close(fig)
    fig=plt.figure(figsize=(13,5.5))
    layout=fig.add_gridspec(1,4,width_ratios=[1,.035,1,.035],wspace=.22,left=.03,right=.94,bottom=.12,top=.88)
    height_limit=200*max(seq[-1]["max_abs_uz"] for seq in sequences.values())*1.05
    for i,(s,seq) in enumerate(sequences.items()):
        r=seq[-1]
        with np.load(Path(r["runtime_directory"])/"fields.npz") as f:
            xs,ys,uz=f["xs"],f["ys"],f["uz"].reshape(len(f["ys"]),len(f["xs"]))
        xx,yy=np.meshgrid(xs,ys)
        ax=fig.add_subplot(layout[0,2*i],projection="3d")
        surf=ax.plot_surface(xx,yy,200*uz,cmap="viridis",rstride=max(1,len(ys)//60),cstride=max(1,len(xs)//80),linewidth=0)
        surf.set_clim(0,200*r["max_abs_uz"])
        ax.set(xlabel="x (mm)",ylabel="y (mm)",zlim=(0,height_limit),title=f"{s}, h={r['h']:g} mm; deformation x200")
        ax.tick_params(labelsize=8)
        ax.text2D(.01,.02,"Height axis: UZ x200 (mm)",transform=ax.transAxes,fontsize=8)
        ax.set_box_aspect((1,.6,.25)); ax.view_init(25,-60)
        cb=fig.colorbar(surf,cax=fig.add_subplot(layout[0,2*i+1]))
        ticks=np.linspace(0,200*r["max_abs_uz"],5); cb.set_ticks(ticks,labels=[f"{v/200:.3f}" for v in ticks]); cb.set_label("Actual UZ (mm)",fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.savefig(figures/"deformed_panel.png",dpi=180); plt.close(fig)
    lines=["| h, mm | Elements | P0 DoF | P0 center, mm | P1 DoF | P1 center, mm |", "|---:|---:|---:|---:|---:|---:|"]
    for a,b in zip(sequences["P0"],sequences["P1"]):
        lines.append(f"| {a['h']:g} | {a['elements']} | {a['active_dof']} | {a['central_deflection']:.9f} | {b['active_dof']} | {b['central_deflection']:.9f} |")
    (output/"table.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(output/"artifact_manifest.json", {"source_directory":str(source),"publication_source_sha256":{p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name("analysis.py"))},"files":{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    print(json.dumps({"output":str(output),"cases":len(records),"converged":{s:v["converged"] for s,v in convergence.items()},"nonlinear_max_difference":None if nonlinear_comparison is None else nonlinear_comparison["maximum_deflection_or_disk_vm_relative_difference"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--audit",type=Path,required=True)
    args=parser.parse_args()
    publish(args.input,args.output,args.audit)
