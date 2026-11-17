"""Audit portable strip results without starting ANSYS or applying a policy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/"meshing"))
from run_experiment import assess,read,verify
from analysis import bilinear_stress,stress_metrics
from publish_mesh_gate import expand_loads,quality
from run_mesh_gate import load_check
from recovery import compute_indicator,rotate_resultants,build_observation
from strip_mesh import classical_selector,dorfler_mark,remesh


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("run",type=Path);args=ap.parse_args()
    root=args.run;verify(root);inv=read(root/"invocation.json");c=inv["config"]
    protocol=read(root/"protocol.json");summary=read(root/"summary.json")
    reference=read(HERE.parent/"runs/20261116_panel03/P1_h3.125_linear.json")
    checked={}
    for row in summary["completed_physical_solves"]:
        case=row["case"];directory=root/case;r=read(directory/"measurement.json");mesh=read(directory/"mesh.json")
        q=quality(mesh,c)
        if not q["passed"] or r["active_dof"]!=6*(r["nx"]-1)*(r["ny"]-1) or r["active_dof"]>5382:raise ValueError("Geometry/equation/cap mismatch")
        eq=r["solver_confirmation"]["equation_lines"]
        parsed=re.search(r"number of equations\s*=\s*(\d+)",eq[-1],re.I)
        if not parsed or int(parsed[1])!=r["active_dof"] or not r["solver_confirmation"]["completed_substep_lines"]:raise ValueError("Native solver evidence incomplete")
        loads=load_check(expand_loads(r["native_checks"]["load_evidence"]),mesh,c)
        if not loads["passed"] or loads!=r["native_checks"]["load_check"]:raise ValueError("Boundary/pressure mismatch")
        fields=read(directory/"fields.json");x,y=np.array(fields["x_lines_mm"]),np.array(fields["y_lines_mm"])
        stress=np.zeros((len(y),len(x),6));stress[:,:,0]=np.array(fields["uz_mm"]).reshape(len(y),len(x))
        w=bilinear_stress(x,y,stress,[[500.,300.]])[0,0]
        if not np.isclose(w,r["central_deflection"],rtol=1e-12,atol=1e-13):raise ValueError("Central interpolation mismatch")
        for surface in ("top","bottom"):
            metrics=stress_metrics(x,y,np.array(fields[surface+"_MPa"]),margin=c["interior_margin"],radius=c["disk_radius"],centers=c["disk_centers"])
            for name,disk in metrics["disks"].items():
                if not disk["quadrature_converged"] or not np.isclose(disk["mean_von_mises"],r["disks"][surface][name]["mean_von_mises"],rtol=1e-11):raise ValueError("Disk VM audit mismatch")
        native=read(directory/"resultants.json");values=rotate_resultants(native["local_resultants"],native["bases_xy"])
        if not np.allclose(values,native["global_resultants"],rtol=1e-12,atol=1e-14):raise ValueError("Resultant transformation mismatch")
        indicator=compute_indicator(mesh,values,young=c["young"],poisson=c["poisson"],thickness=c["thickness"],shear_stiffness=native["shear_stiffness_matrix"],strain_energy=r["strain_energy_N_mm"])
        original=read(directory/"indicator.json")
        if not np.allclose(indicator["eta2"],original["eta2"],rtol=1e-11,atol=1e-13):raise ValueError("Mechanical indicator audit mismatch")
        obs=read(directory/"observation.json")
        if not np.array_equal(classical_selector(obs),np.array(original["eta2"])):raise ValueError("Selector mismatch")
        checked[case]={"elements":r["elements"],"actual_dof":r["active_dof"],"assessment":assess(r,reference),"indicator_sum_N_mm":indicator["sum_eta2"]}
    for row in summary["assessments"]:
        if row["relative_errors"]!=checked[row["case"]]["assessment"]["relative_errors"]:raise ValueError("Reference assessment differs")
    for proposal in sorted(root.glob("proposal_*.json")):
        data=read(proposal);step=data["step"];prior=root/("shared0" if step==1 else f"adaptive{step-1}")
        obs=read(prior/"observation.json");mesh=read(prior/"mesh.json")
        ids,mark=dorfler_mark(obs["element_ids"],classical_selector(obs))
        _,_,p=remesh(mesh["x_lines_mm"],mesh["y_lines_mm"],ids,c)
        if mark!=data["marking"] or p!=data["proposal"]:raise ValueError("Dorfler/strip proposal does not reproduce")
    print(json.dumps({"portable_audit_passed":True,"solve_performed":False,"neural_inference":False,"physical_cases":{k:{key:v for key,v in row.items() if key!="assessment"} for k,row in checked.items()}},indent=2))


if __name__=="__main__":main()
