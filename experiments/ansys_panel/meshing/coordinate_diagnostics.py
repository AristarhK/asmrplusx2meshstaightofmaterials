"""Localize and visualize planar Q4 shape defects without MAPDL or a solve.

The Jacobian condition metric uses exactly the eight samples of validate_mesh.
The opposite-edge angle is an explicitly defined geometric diagnostic; it is
not an identification of native ANSYS Parallel Deviation warning elements.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize


GAUSS = 1. / math.sqrt(3.)
SAMPLES = [(-1.,-1.), (1.,-1.), (1.,1.), (-1.,1.),
           (-GAUSS,-GAUSS), (GAUSS,-GAUSS), (GAUSS,GAUSS), (-GAUSS,GAUSS)]


def _cross(first, second):
    return first[...,0]*second[...,1]-first[...,1]*second[...,0]


def _quads(mesh):
    points = {int(n): np.asarray(p[:2],float)
              for n,p in zip(mesh["node_ids"],mesh["xyz_mm"])}
    return np.asarray([[points[int(n)] for n in row]
                       for row in mesh["connectivity_node_ids"]])


def _inside_polygon(points, polygon, tolerance):
    """Containment in one convex source cell, independent of its orientation."""
    edges = np.roll(polygon,-1,axis=0)-polygon
    crosses = _cross(edges[None,:,:],points[:,None,:]-polygon[None,:,:])
    threshold = tolerance*np.maximum(np.linalg.norm(edges,axis=1),1.)
    return np.all(crosses >= -threshold,axis=1) | np.all(crosses <= threshold,axis=1)


def _statistics(quads):
    if len(quads)==0:
        return {"elements":0,"edge_length_min_mm":None,"edge_length_max_mm":None,
                "sqrt_area_min_mm":None,"sqrt_area_max_mm":None,"total_area_mm2":0.}
    lengths = np.linalg.norm(np.roll(quads,-1,axis=1)-quads,axis=2)
    area = np.abs(np.sum(_cross(quads,np.roll(quads,-1,axis=1)),axis=1)/2)
    return {"elements":len(quads),"edge_length_min_mm":float(lengths.min()),
            "edge_length_max_mm":float(lengths.max()),
            "sqrt_area_min_mm":float(np.sqrt(area).min()),
            "sqrt_area_max_mm":float(np.sqrt(area).max()),
            "total_area_mm2":float(area.sum())}


def diagnose(mesh, source_base, marked_ids, config, reference_mesh=None):
    """Return JSON-ready element diagnostics and immutable marked-region IDs.

    For an after-mesh, pass its original failed mesh as ``reference_mesh``. The
    protected IDs are then selected from original coordinates and their sizes
    are measured on the current coordinates. An element is protected when all
    four original vertices lie in the union of the marked source-base cells.
    This selection is intentionally more conservative than centroid selection.
    """
    tolerance = float(config.get("coordinate_tolerance_mm",1e-8))
    limits = config["quality_limits"]
    quads = _quads(mesh)
    ids = np.asarray(mesh["element_ids"],int)
    kappa, scaled, determinants = [],[],[]
    for xi,eta in SAMPLES:
        dxi = .25*np.array([-(1-eta),1-eta,1+eta,-(1+eta)])
        deta = .25*np.array([-(1-xi),-(1+xi),1+xi,1-xi])
        gx = np.einsum("j,ijk->ik",dxi,quads)
        gy = np.einsum("j,ijk->ik",deta,quads)
        jac = np.stack((gx,gy),axis=2)
        singular = np.linalg.svd(jac,compute_uv=False)
        det = _cross(gx,gy)
        denominator = np.linalg.norm(gx,axis=1)*np.linalg.norm(gy,axis=1)
        kappa.append(singular[:,0]/singular[:,1])
        scaled.append(det/denominator)
        determinants.append(det)
    kappa = np.asarray(kappa).T
    scaled = np.asarray(scaled).T
    determinants = np.asarray(determinants).T
    edges = np.roll(quads,-1,axis=1)-quads
    lengths = np.linalg.norm(edges,axis=2)
    areas = np.sum(_cross(quads,np.roll(quads,-1,axis=1)),axis=1)/2
    angle_values=[]
    for corner in range(4):
        first = quads[:,(corner-1)%4]-quads[:,corner]
        second = quads[:,(corner+1)%4]-quads[:,corner]
        cosine = np.sum(first*second,axis=1)/(np.linalg.norm(first,axis=1)*np.linalg.norm(second,axis=1))
        angle_values.append(np.degrees(np.arccos(np.clip(cosine,-1.,1.))))
    angles = np.asarray(angle_values).T
    opposite = []
    for first,second in ((0,2),(1,3)):
        cosine = np.abs(np.sum(edges[:,first]*edges[:,second],axis=1)/(lengths[:,first]*lengths[:,second]))
        opposite.append(np.degrees(np.arccos(np.clip(cosine,-1.,1.))))
    opposite = np.asarray(opposite).T
    bad_mask = ((kappa.max(axis=1)>limits["aspect_ratio_max"]+1e-12)
                | (scaled.min(axis=1)<limits["scaled_jacobian_min"]-1e-12)
                | (angles.min(axis=1)<limits["angle_min_degrees"]-1e-10)
                | (angles.max(axis=1)>limits["angle_max_degrees"]+1e-10)
                | (determinants.min(axis=1)<=0))
    records=[]
    for i,(eid,ns) in enumerate(zip(ids,mesh["connectivity_node_ids"])):
        violations=[]
        if kappa[i].max()>limits["aspect_ratio_max"]+1e-12:violations.append("aspect_ratio_threshold")
        if scaled[i].min()<limits["scaled_jacobian_min"]-1e-12:violations.append("scaled_jacobian_threshold")
        if angles[i].min()<limits["angle_min_degrees"]-1e-10 or angles[i].max()>limits["angle_max_degrees"]+1e-10:violations.append("angle_threshold")
        if determinants[i].min()<=0:violations.append("positive_jacobian")
        records.append({"element_id":int(eid),"node_ids":list(map(int,ns)),
                        "sample_kappa":kappa[i].tolist(),"maximum_kappa":float(kappa[i].max()),
                        "minimum_scaled_jacobian":float(scaled[i].min()),
                        "minimum_det_jacobian_mm2":float(determinants[i].min()),
                        "minimum_angle_degrees":float(angles[i].min()),
                        "maximum_angle_degrees":float(angles[i].max()),
                        "opposite_edge_line_angle_max_degrees":float(opposite[i].max()),
                        "area_mm2":float(areas[i]),"edge_lengths_mm":lengths[i].tolist(),
                        "violations":violations})
    marked_set = set(map(int,marked_ids))
    if not marked_set or not marked_set.issubset(set(map(int,source_base["element_ids"]))):
        raise ValueError("Marked IDs must be a nonempty subset of the source-base elements")
    source_quads = _quads(source_base)
    polygons = source_quads[[int(e) in marked_set for e in source_base["element_ids"]]]
    reference = mesh if reference_mesh is None else reference_mesh
    reference_quads = _quads(reference)
    if list(reference["element_ids"])!=list(mesh["element_ids"]):
        raise ValueError("Reference/current element order or IDs differ")
    inside = np.zeros((len(reference_quads),4),bool)
    for polygon in polygons:
        inside |= _inside_polygon(reference_quads.reshape(-1,2),polygon,tolerance).reshape(-1,4)
    protected = np.all(inside,axis=1)
    protected_nodes = sorted({int(n) for ns,yes in zip(reference["connectivity_node_ids"],protected) if yes for n in ns})
    # A fixed 50-mm margin provides the same physical overview in every case.
    focus = quads[bad_mask].reshape(-1,2) if bad_mask.any() else polygons.reshape(-1,2)
    if not focus.size:focus=quads.reshape(-1,2)
    low,high = focus.min(axis=0)-50.,focus.max(axis=0)+50.
    bounds = [max(0.,float(low[0])),min(float(config["a"]),float(high[0])),
              max(0.,float(low[1])),min(float(config["b"]),float(high[1]))]
    return {"metric_definition":"maximum sigma_max(J)/sigma_min(J) at four Q4 corners and four 2x2 Gauss points",
            "jacobian_samples_xi_eta":[list(p) for p in SAMPLES],
            "opposite_edge_angle_note":"Acute angle between unoriented opposite edge lines; geometric diagnostic, not native ANSYS warning identification",
            "native_warning_element_ids":None,
            "quality_limits":dict(limits),
            "element_metrics":records,"bad_element_ids":ids[bad_mask].tolist(),
            "violation_elements":[r for r in records if r["violations"]],
            "transition_bounds_mm":bounds,
            "quality_summary":{"maximum_kappa":float(kappa.max()),"minimum_scaled_jacobian":float(scaled.min()),
                               "minimum_angle_degrees":float(angles.min()),"maximum_angle_degrees":float(angles.max()),
                               "opposite_edge_line_angle_max_degrees":float(opposite.max()),
                               "violation_element_count":int(bad_mask.sum())},
            "marked_region":{"selection":"All four original vertices individually inside union of marked base-cell polygons; IDs fixed from reference mesh",
                             "marked_base_element_ids":sorted(marked_set),"base_cell_polygons_mm":polygons.tolist(),
                             "protected_element_ids":ids[protected].tolist(),"protected_node_ids":protected_nodes,
                             "size_statistics":_statistics(quads[protected])}}


def plot_diagnostics(mesh, diag, outprefix, title, bounds=None):
    """Create global and annotated transition PNGs without overwriting files.

    Detailed boxes show worst/near-limit element kappa, scaled J and angles.
    Other boxes show just E ID to avoid hiding narrow refined cells. Complete
    per-element metrics are present in ``diag`` regardless of plotted labels.
    Purple labels identify nodes. Red edges identify current quality failures.
    Supply before ``transition_bounds_mm`` to keep an after image comparable.
    """
    outprefix=Path(outprefix)
    outprefix.parent.mkdir(parents=True,exist_ok=True)
    quads=_quads(mesh)
    metrics=diag["element_metrics"]
    values=np.array([r["maximum_kappa"] for r in metrics])
    failed=set(diag["bad_element_ids"])
    detailed=failed | {int(r["element_id"]) for r in metrics if r["maximum_kappa"]>=3.}
    detailed.update(int(mesh["element_ids"][i]) for i in np.argsort(values,kind="stable")[-4:])
    nodes={int(n):np.asarray(p[:2],float) for n,p in zip(mesh["node_ids"],mesh["xyz_mm"])}
    marked=np.asarray(diag["marked_region"]["base_cell_polygons_mm"])
    if bounds is None:bounds=diag["transition_bounds_mm"]
    norm=Normalize(1.,max(4.5,float(values.max())))
    paths=[]
    for suffix,zoom,annotate in (("overview",None,False),("transition",bounds,True)):
        path=outprefix.with_name(outprefix.name+"_"+suffix+".png")
        if path.exists():raise FileExistsError(path)
        fig,ax=plt.subplots(figsize=(16,11) if annotate else (11,7),layout="constrained")
        collection=PolyCollection(quads,cmap="viridis",norm=norm,linewidths=.7,edgecolors="#374151")
        collection.set_array(values);ax.add_collection(collection)
        for quad,eid in zip(quads,mesh["element_ids"]):
            if int(eid) in failed:
                closed=np.vstack((quad,quad[0]));ax.plot(closed[:,0],closed[:,1],color="#d62728",linewidth=2.)
        for polygon in marked:
            closed=np.vstack((polygon,polygon[0]));ax.plot(closed[:,0],closed[:,1],"--",color="#ff8c00",linewidth=1.8)
        if zoom:
            xmin,xmax,ymin,ymax=zoom
        else:
            points=np.array(list(nodes.values()));xmin,ymin=points.min(axis=0)-12.;xmax,ymax=points.max(axis=0)+12.
        ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),aspect="equal",xlabel="x (mm)",ylabel="y (mm)")
        if annotate:
            visible_nodes=set()
            for quad,eid,row in zip(quads,mesh["element_ids"],metrics):
                center=quad.mean(axis=0)
                if not xmin <= center[0] <= xmax or not ymin <= center[1] <= ymax:continue
                visible_nodes.update(row["node_ids"])
                text=f"E{int(eid)}"
                if int(eid) in detailed:
                    text+=(f"  k={row['maximum_kappa']:.3f}\n"
                           f"sJ={row['minimum_scaled_jacobian']:.3f}\n"
                           f"{row['minimum_angle_degrees']:.0f}-{row['maximum_angle_degrees']:.0f} deg")
                ax.text(*center,text,ha="center",va="center",fontsize=7,color="#071b35",
                        bbox={"facecolor":"white","alpha":.84,"edgecolor":"none","pad":.7})
            for node in sorted(visible_nodes):
                point=nodes[node]
                if xmin<=point[0]<=xmax and ymin<=point[1]<=ymax:
                    ax.plot(*point,".",color="#6f1d90",markersize=2)
                    ax.annotate(f"N{node}",point,xytext=(2,2),textcoords="offset points",
                                fontsize=6.5,color="#6f1d90",clip_on=True,
                                bbox={"facecolor":"white","alpha":.85,"edgecolor":"none","pad":.3})
        ax.set_title(title+"\nred=quality failure; orange dashed=original marked area"+(
            "; labels: E ID / kappa / scaled J / angles; purple=N ID" if annotate else ""),fontsize=11)
        fig.colorbar(collection,ax=ax,shrink=.8,label="Worst Q4 Jacobian condition number (8 samples)")
        fig.savefig(path,dpi=170);plt.close(fig);paths.append(path)
    return paths
