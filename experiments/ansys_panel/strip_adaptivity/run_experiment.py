"""Execute bounded P1 strip adaptation and a fresh uniform comparator.

Uses an existing PyMAPDL environment. Completed solves are checkpointed outside
Git; --resume recovers their POST1 exports without repeating a completed solve.
Reference data are read only by assess(), after recovery and selector inputs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANEL=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PANEL));sys.path.insert(0,str(PANEL/"meshing"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from analysis import bilinear_stress, stress_metrics
from run_panel import export_reactions
from run_mesh_gate import (boundary_ids,cdb_commands,digest,load_check,plot_mesh,
                           save_json,snapshot)
from check_optimized_native import rebuild_commands,compare_mesh,shape_summary,native_messages
from publish_mesh_gate import compact_loads,quality
from strip_mesh import build_mesh,classical_selector,dorfler_mark,remesh
from recovery import compute_indicator,rotate_resultants,build_observation
from native_export import export_resultants

CASES=("center","near_y_edge","near_x_edge")
ERROR_KEYS=("w","C_top","C_bottom","Y_top","Y_bottom","X_top","X_bottom")


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def private_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def verify(root):
    root=root.resolve()
    for name,sha in read(root/"artifact_manifest.json")["files"].items():
        path=(root/name).resolve()
        if root not in path.parents or digest(path)!=sha:raise ValueError("Saved artifact integrity failure")


def assess(record,reference):
    values={"w":record["central_deflection"]}; target={"w":reference["central_deflection"]}
    for symbol,name in zip(("C","Y","X"),CASES):
        for surface in ("top","bottom"):
            key=f"{symbol}_{surface}"
            values[key]=record["disks"][surface][name]["mean_von_mises"]
            target[key]=reference["disks"][surface][name]["mean_von_mises"]
    errors={k:abs(values[k]-target[k])/abs(target[k]) for k in ERROR_KEYS}
    primary={k:v<=(.01 if k=="w" else .05) for k,v in errors.items()}
    strict={k:v<=(.002 if k=="w" else .01) for k,v in errors.items()}
    return {"values":values,"reference_values":target,"relative_errors":errors,
            "primary_checks":primary,"strict_checks":strict,
            "primary_goal_met":all(primary.values()),"strict_goal_met":all(strict.values())}


def execute_case(m,name,x,y,c,protocol,runtime,output):
    public=output/name
    if (public/"measurement.json").exists():return read(public/"measurement.json")
    if m is None:raise ValueError("Publication-only requires every eligible physical case already completed")
    work=runtime/name;work.mkdir(exist_ok=True)
    mesh=build_mesh(x,y,c);expected=mesh["topology_prediction"]["expected_free_dof_topology"]
    if expected>protocol["active_dof_cap"]:raise ValueError("DoF proposal exceeds frozen cap before solve")
    start=time.perf_counter();timings={}
    checkpoint=work/"solve_completed.json"
    resumed=checkpoint.exists()
    print(f"START {name}: E={len(mesh['element_ids'])} predicted DoF={expected}",flush=True)
    if resumed:
        saved=read(checkpoint);timings=dict(saved["timing_seconds"])
        m.finish();m.cwd(str(work));m.filname("strip");m.resume(str(work/"solved"),"db")
        native=read(work/"native_pre_solve.json")
        print(f"RECOVER completed solve {name}; POST1 only",flush=True)
    else:
        stage=time.perf_counter();q=quality(mesh,c)
        if not q["passed"]:raise ValueError(f"Geometry gate failed: {q['failures']}")
        timings["geometry_before_native"]=time.perf_counter()-stage
        stage=time.perf_counter();m.finish();m.cwd(str(work));m.filname("strip")
        commands=rebuild_commands(mesh,c)+[f"D,{int(n)},{d},0" for n in boundary_ids(mesh,c) for d in ("UX","UY","UZ","ROTX","ROTY","ROTZ")]+[f"SFE,ALL,1,PRES,1,{c['pressure']}","ALLSEL,ALL","CHECK"]
        script=work/"build.apdl";script.write_text("\n".join(commands)+"\n",encoding="ascii")
        response=str(m.input(str(script)));(work/"build.log").write_text(response,encoding="utf-8")
        timings["build_native"]=time.perf_counter()-stage
        stage=time.perf_counter();table=shape_summary(str(m.shpp("SUMMARY")))
        if any(v["warnings"] or v["errors"] for v in table.values()):raise ValueError("Native shape gate failed")
        actual,nq=snapshot(m,work,c);comparison=compare_mesh(mesh,actual)
        if not nq["passed"] or not comparison["passed"]:raise ValueError("Native mesh/settings round trip failed")
        loaded,_=cdb_commands(m,work,"loaded_pre_solve");loads=load_check(loaded,actual,c)
        if not loads["passed"]:raise ValueError("Native boundary/pressure gate failed")
        native={"quality":nq,"round_trip":comparison,"shape_tests":table,"load_check":loads,"load_evidence":compact_loads(loaded),"actual_element_section_settings":actual["native_settings"],"messages":native_messages(response)}
        if any(z["severity"] in ("WARNING","ERROR","FATAL") for z in native["messages"]):raise ValueError("Native build/CHECK warning before solve")
        private_json(work/"native_pre_solve.json",native)
        timings["native_checks_and_geometry"]=time.perf_counter()-stage
        m.finish();m.run("/SOLU");m.antype("STATIC");m.eqslv("SPARSE");m.nlgeom("OFF")
        m.outres("ALL","ALL");m.time(1);m.nsubst(1);m.neqit(50);m.rescontrol("NONE")
        m.save(str(work/"before_solve"),"db")
        stage=time.perf_counter();response=str(m.solve());timings["solve"]=time.perf_counter()-stage
        (work/"solve_response.txt").write_text(response,encoding="utf-8")
        eq=re.findall(r"number of equations\s*=\s*(\d+)",response,re.I)
        complete=bool(re.search(r"LOAD STEP\s+1\s+SUBSTEP\s+\d+\s+COMPLETED",response,re.I))
        if not eq or not complete or re.search(r"\*\*\*\s+ERROR|SOLUTION NOT CONVERGED",response,re.I):raise ValueError("No confirmed completed sparse solve")
        if int(eq[-1])!=expected:raise ValueError("Actual Number of equations differs from topology prediction")
        saved={"active_dof":int(eq[-1]),"expected_free_dof":expected,"raw_solu_cnvg":int(m.get_value("ACTIVE",0,"SOLU","CNVG")),"completed":True,"warning_messages":native_messages(response),"timing_seconds":timings}
        m.save(str(work/"solved"),"db");private_json(checkpoint,saved)
        if any(a["severity"] in ("WARNING","ERROR","FATAL") for a in saved["warning_messages"]):raise ValueError("Solver warning/error requires review")
    stage=time.perf_counter();m.finish();m.cwd(str(work));m.post1()
    result_files=list(work.glob("*.rst"))
    if len(result_files)!=1:raise ValueError("Cannot identify unique checkpoint RST")
    m.run(f"FILE,'{str(result_files[0].with_suffix(''))}',rst")
    m.set("LAST");m.allsel()
    actual_time=float(m.get_value("ACTIVE",0,"SET","TIME"))
    if not np.isclose(actual_time,1.,atol=1e-10,rtol=0):raise ValueError("Result time is not full load")
    m.run("/GRAPHICS,FULL");m.rsys(0);m.run("LAYER,1");m.run("AVPRIN,0")
    ids=np.asarray(m.mesh.nnum);xyz=np.asarray(m.mesh.nodes)
    if not np.array_equal(ids,np.array(mesh["node_ids"])) or not np.allclose(xyz,mesh["xyz_mm"],atol=1e-8,rtol=0):raise ValueError("Post1 nodal mapping mismatch")
    uz=np.asarray(m.post_processing.nodal_displacement("Z"))
    if not np.isfinite(uz).all() or uz.max()<=0:raise ValueError("Invalid displacement/sign")
    nx,ny=len(x)-1,len(y)-1;surfaces={};metrics={}
    for surface,opt in (("top","TOP"),("bottom","BOT")):
        m.shell(opt)
        tensor=np.column_stack([np.asarray(m.get_array("NODE",item1="S",it1num=k,item2="ESOL")) for k in ("X","Y","Z","XY","YZ","XZ")]).reshape(ny+1,nx+1,6)
        surfaces[surface]=tensor
        metrics[surface]=stress_metrics(x,y,tensor,margin=c["interior_margin"],radius=c["disk_radius"],centers=c["disk_centers"])
        if not all(d["quadrature_converged"] for d in metrics[surface]["disks"].values()):raise ValueError("Control-disk quadrature did not converge")
    m.etable("ESTRAIN","SENE");m.etable("EART","AENE");m.ssum()
    energy=float(m.get_value("SSUM",0,"ITEM","ESTRAIN"));artificial=float(m.get_value("SSUM",0,"ITEM","EART"))
    constraints={int(n):["UX","UY","UZ","ROTX","ROTY","ROTZ"] for n in boundary_ids(mesh,c)}
    forces,moments=export_reactions(m,work,constraints,xyz)
    reaction_force=forces.sum(axis=0);reaction_moment=(np.cross(xyz,forces)+moments).sum(axis=0)
    load=c["pressure"]*c["a"]*c["b"];external_moment=np.array([load*c["b"]/2,-load*c["a"]/2,0.])
    conn=np.array(mesh["connectivity_node_ids"])-1
    area=(np.diff(y)[:,None]*np.diff(x)[None,:]).ravel();nodal_load=np.zeros(len(xyz))
    np.add.at(nodal_load,conn.ravel(),np.repeat(c["pressure"]*area/4,4))
    half_work=float(nodal_load@uz/2)
    force_error=float(np.linalg.norm(reaction_force+np.array([0.,0.,load]))/load)
    moment_error=float(np.linalg.norm(reaction_moment+external_moment)/np.linalg.norm(external_moment))
    energy_error=abs(energy-half_work)/energy
    if force_error>protocol["force_balance_relative_limit"] or moment_error>protocol["moment_balance_relative_limit"] or energy_error>protocol["linear_energy_relative_limit"]:raise ValueError("Force/moment/linear energy balance gate failed")
    if (work/"native_resultants_section_export.json").exists():
        exported=read(work/"native_resultants_section_export.json");exported["private_export_sha256"]=digest(work/"native_resultants_section_export.json")
    else:exported=export_resultants(m,mesh,work,c)
    timings["export_fields_resultants_and_checks"]=time.perf_counter()-stage
    stage=time.perf_counter()
    global_q=rotate_resultants(np.array(exported["local_resultants"]),np.array(exported["bases_xy"]))
    indicator=compute_indicator(mesh,global_q,young=c["young"],poisson=c["poisson"],thickness=c["thickness"],shear_stiffness=exported["shear_stiffness_matrix"],strain_energy=energy,condition_limit=protocol["patch_condition_limit"],quadrature_rtol=protocol["indicator_quadrature_rtol"])
    timings["indicator"]=time.perf_counter()-stage
    stage=time.perf_counter();observation=build_observation(mesh,global_q,indicator,{"a_mm":c["a"],"b_mm":c["b"],"pressure_N_per_mm2":c["pressure"],"young_N_per_mm2":c["young"],"poisson":c["poisson"],"thickness_mm":c["thickness"]})
    scores=classical_selector(observation)
    if not np.allclose(scores,indicator["eta2"],atol=0,rtol=0):raise ValueError("Classical selector mismatch")
    timings["graph_and_selector"]=time.perf_counter()-stage
    # Central physical coordinate, independent of whether it is a mesh node.
    scalar_tensor=np.zeros((ny+1,nx+1,6));scalar_tensor[:,:,0]=uz.reshape(ny+1,nx+1)
    central=float(bilinear_stress(x,y,scalar_tensor,np.array([[c["a"]/2,c["b"]/2]]))[0,0])
    record={"case":name,"status":"completed","elements":len(conn),"nodes":len(xyz),"nx":nx,"ny":ny,"active_dof":saved["active_dof"],"expected_free_dof":expected,"dof_source":"Native sparse solver Number of equations","converged":True,"raw_solu_cnvg":saved["raw_solu_cnvg"],"result_time":actual_time,"central_deflection":central,"strain_energy_N_mm":energy,"artificial_energy_N_mm":artificial,"half_external_work_N_mm":half_work,"force_balance_relative":force_error,"moment_balance_relative":moment_error,"energy_balance_relative":energy_error,"reaction_force_N":reaction_force.tolist(),"reaction_moment_N_mm":reaction_moment.tolist(),"disks":{s:v["disks"] for s,v in metrics.items()},"diagnostic_max_VM":{s:v["global_maximum_von_mises"] for s,v in metrics.items()},"timing_seconds":timings,"native_checks":native,"solver_messages":saved["warning_messages"],"indicator_summary":{"sum_eta2":indicator["sum_eta2"],"sum_parts":indicator["sum_parts"],"eta_rel":indicator["eta_rel"]},"selector":"classical eta2; no neural inference"}
    solve_text=(work/"solve_response.txt").read_text(encoding="utf-8")
    record["solver_confirmation"]={"equation_lines":[s.strip() for s in solve_text.splitlines() if re.search(r"number of equations\s*=",s,re.I)],"completed_substep_lines":[s.strip() for s in solve_text.splitlines() if re.search(r"LOAD STEP\s+1\s+SUBSTEP\s+\d+\s+COMPLETED",s,re.I)],"native_response_sha256":digest(work/"solve_response.txt")}
    final_public=public
    public=work/("portable_draft_"+datetime.now(timezone.utc).strftime("%H%M%S_%f"));public.mkdir()
    stage=time.perf_counter()
    save_json(public/"mesh.json",mesh);save_json(public/"resultants.json",{**exported,"global_resultants":global_q.tolist()})
    save_json(public/"indicator.json",indicator);save_json(public/"observation.json",observation)
    save_json(public/"fields.json",{"x_lines_mm":x.tolist(),"y_lines_mm":y.tolist(),"uz_mm":uz.tolist(),"top_MPa":surfaces["top"].tolist(),"bottom_MPa":surfaces["bottom"].tolist(),"stress_components":["sx","sy","sz","txy","tyz","txz"],"stress_source":"native NODE S ESOL after FULL,RSYS0,LAYER1,AVPRIN0,SHELL TOP/BOT; recovered nodal field"})
    plot_mesh(mesh,set(),public/"mesh.png",f"{name}: {nx}x{ny}; {saved['active_dof']} active DoF")
    timings["compact_export_and_mesh_figure"]=time.perf_counter()-stage
    elapsed=time.perf_counter()-start
    # Record uninstrumented API/setup/serialization gaps rather than dropping them.
    previous_phases=sum(saved["timing_seconds"].values()) if resumed else 0.
    timings["other_case_overhead"]=max(0.,elapsed+previous_phases-sum(timings.values()))
    record["measured_case_work_seconds"]=float(sum(timings.values()))
    record["solve_completed_checkpoint_sha256"]=digest(checkpoint)
    record["native_file_hashes"]={p.name:digest(p) for p in work.iterdir() if p.is_file() and p.suffix in (".apdl",".csv",".txt")}
    save_json(public/"measurement.json",record)
    private_json(work/"measurement_completed.json",{"measurement_sha256":digest(public/"measurement.json")})
    shutil.copytree(public,final_public)
    print(f"DONE {name}: DoF={saved['active_dof']} w={central:.9f} eta={indicator['sum_eta2']:.6g}",flush=True)
    return record


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exec-file",required=True);ap.add_argument("--runtime-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True);ap.add_argument("--resume",action="store_true")
    ap.add_argument("--publish-only",action="store_true",help="Resume tables/figures from complete physical cases without starting MAPDL")
    args=ap.parse_args();top_start=time.perf_counter();protocol=read(Path(__file__).with_name("protocol.json"))
    c=read(PANEL/"panel.json");c.update({"coordinate_tolerance_mm":1e-8,"quality_limits":protocol["quality_limits"]})
    reference_root=PANEL/"runs/20261116_panel03";verify(reference_root)
    reference=read(reference_root/"P1_h3.125_linear.json")
    if args.resume:
        if (args.output/"completed.json").exists():raise ValueError("Completed output immutable; choose new directory")
        invocation=read(args.output/"invocation.json")
        if invocation["protocol_sha256"]!=digest(Path(__file__).with_name("protocol.json")):raise ValueError("Protocol changed")
        if invocation["config"]!=c or invocation["reference_sha256"]!=digest(reference_root/"P1_h3.125_linear.json"):
            raise ValueError("Resume model/reference provenance changed")
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        invocation={"run_id_utc":datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),"protocol_sha256":digest(Path(__file__).with_name("protocol.json")),"source_commit":"4582fefd4e66804e866d7ae9432472588d344f6e","reference_sha256":digest(reference_root/"P1_h3.125_linear.json"),"config":c}
        save_json(args.output/"invocation.json",invocation);save_json(args.output/"protocol.json",protocol)
    runtime=args.runtime_root.resolve()/invocation["run_id_utc"];runtime.mkdir(parents=True,exist_ok=args.resume)
    from ansys.mapdl.core import launch_mapdl
    m=None;physical=[];method_rows=[];rejections=[];adapt_extra=0.;failure=None
    overhead=read(runtime/"session_overhead.json") if (runtime/"session_overhead.json").exists() else {"launch":0.,"close":0.,"failed_attempts":[]}
    log_index=len(list(runtime.glob("session_*.apdl")))
    try:
        if not args.publish_only:
            stage=time.perf_counter();m=launch_mapdl(exec_file=args.exec_file,run_location=str(runtime),nproc=2,additional_switches="-smp",start_timeout=120,start_instance=True,mode="grpc",remove_temp_dir_on_exit=False,log_apdl=str(runtime/f"session_{log_index}.apdl"));m.finish();m.verify("OFF")
            overhead["launch"]+=time.perf_counter()-stage
        x=np.linspace(0,c["a"],21);y=np.linspace(0,c["b"],13)
        initial=execute_case(m,"shared0",x,y,c,protocol,runtime,args.output);physical.append(initial)
        initial_cost=initial["measured_case_work_seconds"]
        adapt_cost=initial_cost
        method_rows.append({"method":"adaptive","step":0,"case":"shared0","cumulative_case_seconds":adapt_cost,**assess(initial,reference)})
        for step in range(1,protocol["max_adaptive_steps"]+1):
            prior=args.output/physical[-1]["case"]
            path=args.output/f"proposal_{step}.json"
            if path.exists():
                proposed=read(path);proposal=proposed["proposal"]
                mark_time=proposed["timing_seconds"]["marking"];remesh_time=proposed["timing_seconds"]["remesh"]
                newx=np.array(proposal["x_lines_mm"]);newy=np.array(proposal["y_lines_mm"])
            else:
                stage=time.perf_counter();obs=read(prior/"observation.json");scores=classical_selector(obs);marked,mark=dorfler_mark(obs["element_ids"],scores,protocol["theta"]);mark_time=time.perf_counter()-stage
                stage=time.perf_counter();newx,newy,proposal=remesh(x,y,marked,c);remesh_time=time.perf_counter()-stage
                proposed={"step":step,"marking":mark,"proposal":proposal,"timing_seconds":{"marking":mark_time,"remesh":remesh_time}}
                save_json(path,proposed)
            adapt_extra+=mark_time+remesh_time;adapt_cost+=mark_time+remesh_time
            if not proposal["within_dof_budget"]:
                rejections.append(proposed);print(f"REJECT adaptive{step} before solve: predicted DoF={proposal['after']['expected_free_dof_topology']}",flush=True);break
            current=execute_case(m,f"adaptive{step}",newx,newy,c,protocol,runtime,args.output);physical.append(current)
            adapt_cost+=current["measured_case_work_seconds"]
            method_rows.append({"method":"adaptive","step":step,"case":current["case"],"cumulative_case_seconds":adapt_cost,**assess(current,reference)})
            x,y=newx,newy
        uniform=execute_case(m,"uniform25",np.linspace(0,c["a"],41),np.linspace(0,c["b"],25),c,protocol,runtime,args.output);physical.append(uniform)
        method_rows.extend([{"method":"uniform","step":0,"case":"shared0","cumulative_case_seconds":initial_cost,**assess(initial,reference)}, {"method":"uniform","step":1,"case":"uniform25","cumulative_case_seconds":initial_cost+uniform["measured_case_work_seconds"],**assess(uniform,reference)}])
    except Exception as error:
        failure={"type":type(error).__name__,"message":str(error)}
        idx=len(overhead["failed_attempts"]);(runtime/f"failure_{idx}.txt").write_text(traceback.format_exc(),encoding="utf-8")
        overhead["failed_attempts"].append({"elapsed_attempt_seconds":time.perf_counter()-top_start,**failure});print(traceback.format_exc(),flush=True)
    finally:
        if m is not None:
            stage=time.perf_counter();m.exit();overhead["close"]+=time.perf_counter()-stage
        if failure:overhead["failed_attempts"][-1]["attempt_wall_seconds"]=time.perf_counter()-top_start
        private_json(runtime/"session_overhead.json",overhead)
    if failure:return 1
    # Comparison publication is independent from solved checkpoints and can resume.
    from publish_results import publish_results
    publication_start=time.perf_counter()
    publish_results(args.output,physical,method_rows,rejections,overhead,time.perf_counter()-top_start,runtime)
    save_json(args.output/"wall_measurement.json",{"final_invocation_wall_including_publication_seconds":time.perf_counter()-top_start,"publication_seconds":time.perf_counter()-publication_start,"all_attempts_wall_including_publication_seconds":time.perf_counter()-top_start+sum(a.get("attempt_wall_seconds",a["elapsed_attempt_seconds"]) for a in overhead["failed_attempts"]),"end_event":"all comparison artifacts published; final small stamp/hash writes excluded","development_and_user_license_repair_wait_excluded":True})
    save_json(args.output/"completed.json",{"completed":True,"physical_solve_count":len(physical),"adaptive_solve_count":sum(r["case"].startswith("adaptive") for r in physical),"timestamp_utc":datetime.now(timezone.utc).isoformat()})
    save_json(args.output/"artifact_manifest.json",{"files":{p.relative_to(args.output).as_posix():digest(p) for p in args.output.rglob("*") if p.is_file()}})
    return 0


if __name__=="__main__":raise SystemExit(main())
