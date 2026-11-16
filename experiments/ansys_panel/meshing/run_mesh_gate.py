"""Native EREFINE mesh-only gate. This module never calls SOLVE or an indicator.

Own MAPDL sessions, immutable unloaded base, independent database copies.
Full runtime stays private; the public output contains portable audit evidence.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from validate_mesh import validate_mesh
from cdb_loads import parse_cdb_loads


NUMBER=r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write("\n")


def input_commands(mapdl,work,name,lines):
    for line in lines:
        if re.match(r"\s*(SOLVE|LSSOLVE|MFSOLVE|PSOLVE)\s*(,|$)",line,re.I):
            raise ValueError("SOLVE commands are forbidden in this mesh-only runner")
    path=work/(name+".apdl")
    path.write_text("\n".join(lines)+"\n",encoding="ascii")
    response=mapdl.input(str(path))
    (work/(name+".log")).write_text(str(response),encoding="utf-8")
    return str(response)


def base_commands(c):
    nx,ny=20,12; h=50.
    lines=["FINISH","/CLEAR,NOSTART","/PREP7","/NOPR","ET,1,SHELL181","ETCONTROL,OFF"]
    lines += [f"KEYOPT,1,{k},{v}" for k,v in c["keyopts"].items()]
    lines += [f"MP,EX,1,{c['young']}",f"MP,PRXY,1,{c['poisson']}","SECTYPE,1,SHELL",f"SECDATA,{c['thickness']},1,0,{c['thickness_integration_points']}","SECOFFSET,MID","TYPE,1","MAT,1","SECNUM,1"]
    for j in range(ny+1):
        for i in range(nx+1):
            lines.append(f"N,{j*(nx+1)+i+1},{i*h:g},{j*h:g},0")
    for j in range(ny):
        for i in range(nx):
            n=j*(nx+1)+i+1
            lines.append(f"E,{n},{n+1},{n+nx+2},{n+nx+1}")
    return lines+["ALLSEL,ALL","/GOPR","CHECK"]


def cdb_commands(mapdl,work,name):
    mapdl.allsel()
    mapdl.cdwrite("DB",str(work/name),"cdb")
    path=work/(name+".cdb")
    return parse_cdb_loads(path),path


def settings(mapdl,work):
    etext=str(mapdl.etlist())
    stext=str(mapdl.slist(1,1))
    (work/"element_settings.txt").write_text(etext,encoding="utf-8")
    (work/"section_settings.txt").write_text(stext,encoding="utf-8")
    element_name=int(round(mapdl.get_value("ETYP",1,"ATTR","ENAM")))
    all_keys={}
    for match in re.finditer(r"KEYOPT\(\s*(\d+)\s*-\s*(\d+)\)\s*=([^\r\n]+)",etext):
        start,end=int(match[1]),int(match[2]);values=list(map(int,match[3].split()))
        if len(values)!=end-start+1:raise ValueError("Unrecognized native KEYOPT listing")
        all_keys.update({str(i):v for i,v in zip(range(start,end+1),values)})
    keys={str(k):all_keys[str(k)] for k in (1,3,8)}
    match=re.search(r"Total Thickness\s*=\s*("+NUMBER+r")",stext,re.I)
    layer=re.search(r"^\s*1\s+("+NUMBER+r")\s+(\d+)\s+("+NUMBER+r")\s+(\d+)\s*$",stext,re.M)
    if not match or not layer:raise ValueError("Unrecognized native section listing")
    thickness=float(match[1])
    return {"element_name":element_name,"keyopts":keys,"thickness_mm":thickness,"layer_thickness_mm":float(layer[1]),"layer_material":int(layer[2]),"layer_angle_degrees":float(layer[3]),"through_thickness_points":int(layer[4]),"offset":"MID" if re.search(r"offset to MID",stext,re.I) else "UNKNOWN"}


def snapshot(mapdl,work,c):
    mapdl.allsel()
    ids=np.asarray(mapdl.mesh.nnum,dtype=int).copy()
    xyz=np.asarray(mapdl.mesh.nodes,dtype=float).copy()
    raw=[np.asarray(e,dtype=int).copy() for e in mapdl.mesh.elem]
    elem_ids=np.array([e[8] for e in raw],dtype=int)
    conn=np.array([np.pad(e[10:14],(0,max(0,4-len(e[10:14])))) for e in raw],dtype=int)
    native=settings(mapdl,work)
    attrs={"TYPE":np.array([e[1] for e in raw]),"MAT":np.array([e[0] for e in raw]),"SECNUM":np.array([e[3] for e in raw]),"THICKNESS":np.full(len(raw),native["thickness_mm"])}
    # Capture native integer IDs and nodal coordinates, not VTK connectivity.
    data={"node_ids":ids.tolist(),"xyz_mm":xyz.tolist(),"element_ids":elem_ids.tolist(),"connectivity_node_ids":conn.tolist(),"attributes":{k:v.tolist() for k,v in attrs.items()},"native_settings":native}
    quality=validate_mesh(ids,xyz,elem_ids,conn,attrs,c)
    return data,quality


def boundary_ids(data,c):
    xyz=np.array(data["xyz_mm"]); ids=np.array(data["node_ids"])
    tol=c.get("coordinate_tolerance_mm",1e-8)
    mask=(np.abs(xyz[:,0])<=tol)|(np.abs(xyz[:,0]-c["a"])<=tol)|(np.abs(xyz[:,1])<=tol)|(np.abs(xyz[:,1]-c["b"])<=tol)
    return ids[mask]


def load_check(commands,data,c):
    boundary=boundary_ids(data,c)
    expected={(int(n),d,0.) for n in boundary for d in ("UX","UY","UZ","ROTX","ROTY","ROTZ")}
    actual=set(); sfe=[]; other=[]; d_count=0; imaginary_ok=True
    for fields in commands:
        if fields[0]=="D":
            d_count+=1
            actual.add((int(float(fields[1])),fields[2],float(fields[3].replace("D","E"))))
            imaginary_ok &= len(fields)<5 or not fields[4] or float(fields[4].replace("D","E"))==0.
        elif fields[0]=="SFE":
            sfe.append(fields)
        else:
            other.append(fields[0])
    ids=np.array(data["element_ids"])
    pressure={}; pressure_record_ok=True
    for f in sfe:
        if len(f)!=9 or f[3].upper()!="PRES" or f[4] not in ("0","1",""):
            pressure_record_ok=False
            continue
        vals=[float(v.replace("D","E")) for v in f[5:9] if v]
        element_id=int(float(f[1]))
        if element_id in pressure:pressure_record_ok=False
        pressure[element_id]={"face":int(float(f[2])),"values":vals}
    coordinates={int(n):np.array(p) for n,p in zip(data["node_ids"],data["xyz_mm"])}
    force=np.zeros(3); area=0.; pressure_ok=pressure_record_ok and set(pressure)==set(ids.tolist())
    for e,nodes in zip(ids,data["connectivity_node_ids"]):
        unique=list(dict.fromkeys(n for n in nodes if n>0))
        if len(unique)<3:
            pressure_ok=False; continue
        points=np.array([coordinates[n] for n in unique])
        area_vector=sum((np.cross(a,b) for a,b in zip(points,np.roll(points,-1,axis=0))),np.zeros(3))/2
        p=pressure.get(int(e))
        if not p or p["face"]!=1 or len(p["values"])!=4 or not np.allclose(p["values"],c["pressure"],atol=1e-14,rtol=0):
            pressure_ok=False; continue
        area+=float(np.linalg.norm(area_vector))
        force+=p["values"][0]*area_vector
    expected_free=6*len(data["node_ids"])-len(expected)
    exact_dof=actual==expected and d_count==len(actual) and imaginary_ok
    return {"passed":exact_dof and pressure_ok and not other and np.allclose(force,[0,0,3000],atol=3e-6,rtol=0) and abs(area-600000)<=.006,
      "boundary_nodes":len(boundary),"prescribed_dof":len(actual),"expected_prescribed_dof":len(expected),"exact_p1_dof_set":exact_dof,"pressure_element_count":len(pressure),"all_elements_face1_pressure_correct":pressure_ok,"loaded_area_mm2":area,"pressure_resultant_N":force.tolist(),"other_load_or_constraint_commands":other,
      "no_duplicate_dof_records":d_count==len(actual),"zero_imaginary_displacements":bool(imaginary_ok),
      "expected_free_dof_topology":expected_free,"solver_equation_count":None,"dof_note":"Topology prediction only; no SOLVE/equation assembly performed"}


def plot_mesh(data,marked,out,title,region=None):
    xy={int(n):np.array(p[:2]) for n,p in zip(data["node_ids"],data["xyz_mm"])}
    polygons=[np.array([xy[n] for n in nodes if n>0]) for nodes in data["connectivity_node_ids"]]
    fig,ax=plt.subplots(figsize=(10,6.4),layout="constrained")
    ax.add_collection(PolyCollection(polygons,facecolors=["#f4ac7b" if int(e) in marked else "#f6f7f9" for e in data["element_ids"]],edgecolors="#283749",linewidths=.55))
    if region:
        ax.plot(*np.array(region).T,color="#d55e00",linewidth=1.8)
    ax.set(xlim=(-15,1015),ylim=(-15,615),aspect="equal",xlabel="x (mm)",ylabel="y (mm)",title=title)
    fig.savefig(out,dpi=170);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exec-file",default=os.getenv("PYMAPDL_MAPDL_EXEC"),required=not os.getenv("PYMAPDL_MAPDL_EXEC"))
    ap.add_argument("--runtime-root",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--config",type=Path,default=Path(__file__).resolve().parents[1]/"panel.json")
    args=ap.parse_args()
    c=json.loads(args.config.read_text(encoding="utf-8")); c.update({"coordinate_tolerance_mm":1e-8,"quality_limits":{"aspect_ratio_max":4.,"angle_min_degrees":30.,"angle_max_degrees":150.,"scaled_jacobian_min":.2}})
    args.output.mkdir(parents=True,exist_ok=False)
    run_id=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    runtime=args.runtime_root.resolve()/run_id; runtime.mkdir(parents=True,exist_ok=False)
    figures=args.output/"figures";figures.mkdir()
    private=Path(".local")/f"mesh-gate-{run_id}.json";private.parent.mkdir(exist_ok=True)
    save_json(private,{"runtime_directory":str(runtime),"output_directory":str(args.output.resolve()),"solver_executable":str(Path(args.exec_file).resolve())})
    from ansys.mapdl.core import launch_mapdl
    def launch(work):
        session=launch_mapdl(exec_file=args.exec_file,run_location=str(work),nproc=2,additional_switches="-smp",start_timeout=120,start_instance=True,mode="grpc",remove_temp_dir_on_exit=False,log_apdl=str(work/"session.apdl"))
        session.finish();session.verify("OFF")
        return session
    basework=runtime/"base";basework.mkdir();m=None
    try:
        m=launch(basework)
        native_version=str(m.version)
        input_commands(m,basework,"base_model",base_commands(c))
        empty,base_cdb=cdb_commands(m,basework,"unloaded")
        base,base_quality=snapshot(m,basework,c)
        if empty or not base_quality["passed"] or len(base["element_ids"])!=240:
            raise RuntimeError("Initial unloaded 240-element mesh failed acceptance")
        m.save(str(basework/"base_unloaded"),"db")
        save_json(args.output/"base_mesh.json",base)
        save_json(args.output/"base_quality.json",base_quality)
        plot_mesh(base,set(),figures/"base.png","Base: 240 SHELL181; h=50 mm; unloaded")
    finally:
        if m is not None:m.exit()
    pts={int(n):np.array(p) for n,p in zip(base["node_ids"],base["xyz_mm"])}
    centers=np.array([np.mean([pts[n] for n in ns],axis=0) for ns in base["connectivity_node_ids"]])
    eid=np.array(base["element_ids"])
    definitions={"interior_single":(centers[:,0]==475)&(centers[:,1]==275),"edge_patch":(centers[:,0]>=450)&(centers[:,0]<=550)&(centers[:,1]<100),"corner_patch":(centers[:,0]<100)&(centers[:,1]<100),"all_elements":np.ones(len(eid),dtype=bool)}
    records=[]
    for name,mask in definitions.items():
        work=runtime/name;work.mkdir();marked=eid[mask].tolist();m=None
        record={"case":name,"marked_element_ids":marked,"marked_count":len(marked),"solve_performed":False,"indicator_computed":False}
        print(f"START {name}: marked={len(marked)}",flush=True)
        try:
            m=launch(work);m.resume(str(basework/"base_unloaded"),"db");m.prep7();m.cwd(str(work));m.allsel()
            before,_=cdb_commands(m,work,"before_refine")
            record["unloaded_before_refine"]=not before;record["before_load_commands"]=before
            if before:raise RuntimeError("Refinement input has prohibited loads/constraints")
            input_commands(m,work,"mark",["ESEL,NONE"]+[f"ESEL,A,ELEM,,{e}" for e in marked]+["CM,MARKED,ELEM","ALLSEL,ALL"])
            t=time.perf_counter()
            try:
                response=str(m.run("EREFINE,MARKED,,,1,0,OFF,ON"))
                record["native_refinement_error"]=None
            except Exception as error:
                response=str(error);record["native_refinement_error"]={"type":type(error).__name__,"message":str(error)}
            record["refinement_seconds"]=time.perf_counter()-t
            (work/"refine_response.txt").write_text(response,encoding="utf-8")
            record["native_warning_or_error_lines"]=[s.strip() for s in response.splitlines() if re.search(r"\*\*\*\s+(?:ERROR|WARNING)\s+\*\*\*",s,re.I)]
            after,quality=snapshot(m,work,c)
            save_json(args.output/f"{name}_mesh.json",after)
            record["quality"]=quality;record["elements_before"]=len(eid);record["elements_after"]=len(after["element_ids"]);record["nodes_before"]=len(base["node_ids"]);record["nodes_after"]=len(after["node_ids"])
            record["element_increase"]=record["elements_after"]-len(eid);record["node_increase"]=record["nodes_after"]-len(base["node_ids"])
            record["element_growth_factor"]=record["elements_after"]/len(eid)
            actual_settings=after["native_settings"]
            record["element_section_settings_preserved"]=actual_settings==base["native_settings"]
            input_commands(m,work,"reapply_p1",["ALLSEL,ALL"]+[f"D,{n},{d},0" for n in boundary_ids(after,c) for d in ("UX","UY","UZ","ROTX","ROTY","ROTZ")]+[f"SFE,ALL,1,PRES,,{c['pressure']}","FINISH","/SOLU","ANTYPE,STATIC","NLGEOM,OFF","EQSLV,SPARSE","FINISH","/PREP7","ALLSEL,ALL"])
            loaded,_=cdb_commands(m,work,"after_loading")
            record["boundary_and_pressure"]=load_check(loaded,after,c)
            m.save(str(work/"mesh_with_p1"),"db")
            record["passed"]=bool(quality["passed"] and record["unloaded_before_refine"] and record["native_refinement_error"] is None and record["element_increase"]>0 and record["element_section_settings_preserved"] and record["boundary_and_pressure"]["passed"] and not record["native_warning_or_error_lines"])
            plot_mesh(base,set(marked),figures/f"{name}_before.png",f"{name}: before; orange=marked ({len(marked)})")
            plot_mesh(after,set(),figures/f"{name}_after.png",f"{name}: after; {record['elements_after']} elements; {'PASS' if record['passed'] else 'FAIL'}")
        except Exception as error:
            record["passed"]=False;record["case_error"]={"type":type(error).__name__,"message":str(error)}
            (work/"case_exception.txt").write_text(traceback.format_exc(),encoding="utf-8")
        finally:
            if m is not None:m.exit()
        record["raw_input_sha256"]={p.name:digest(p) for p in work.iterdir() if p.is_file() and p.suffix in (".apdl",".cdb")}
        save_json(args.output/f"{name}.json",record);records.append(record)
        print(f"DONE {name}: passed={record['passed']} elements={record.get('elements_after')}",flush=True)
    forbidden=[]
    for log in runtime.rglob("session.apdl"):
        for line in log.read_text(encoding="utf-8",errors="replace").splitlines():
            if re.match(r"\s*(SOLVE|LSSOLVE|MFSOLVE|PSOLVE)\s*(,|$)",line,re.I):forbidden.append(line)
    solution_files=[p.name for p in runtime.rglob("*") if p.is_file() and p.suffix.lower() in (".rst",".full",".rth",".mntr")]
    summary={"case_count":len(records),"all_cases_passed":all(r["passed"] for r in records),"eligible_for_adaptive_solves":all(r["passed"] for r in records),"solve_commands":forbidden,"solution_files":solution_files,"solve_performed":bool(forbidden or solution_files),"cases":records}
    save_json(args.output/"summary.json",summary)
    save_json(args.output/"manifest.json",{"run_id_utc":run_id,"native_version":native_version,"packages":{k:importlib.metadata.version(k) for k in ("ansys-mapdl-core","numpy","matplotlib")},"source_sha256":{p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},"config":c,"refinement":{"LEVEL":1,"DEPTH":0,"POST":"OFF","RETAIN":"ON"},"base_unloaded_cdb_sha256":digest(base_cdb),"machine_paths":"Private runtime pointer only; not published"})
    fields=["case","marked_count","elements_before","elements_after","nodes_before","nodes_after","element_increase","element_growth_factor","passed"]
    with (args.output/"comparison.csv").open("x",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:r.get(k) for k in fields} for r in records)
    save_json(args.output/"artifact_manifest.json",{"files":{p.relative_to(args.output).as_posix():digest(p) for p in args.output.rglob("*") if p.is_file()}})
    print(json.dumps({k:v for k,v in summary.items() if k!="cases"},indent=2))
    return 1 if summary["solve_performed"] else 0


if __name__=="__main__":raise SystemExit(main())
