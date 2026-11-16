"""Independent geometry/topology checks for the planar panel Q4 mesh.

No MAPDL imports or solver calls. IDs need not be contiguous. Aspect ratio is
defined conservatively as the largest singular-value ratio of the planar Q4
Jacobian at four corners and four Gauss points; side-length ratio is also saved.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import math

import numpy as np


def _cross(a, b):
    return a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]


def _area(points):
    return float(np.sum(_cross(points, np.roll(points, -1, axis=0)))/2)


def _clip(subject, clipper, tolerance):
    """Convex polygon intersection (Sutherland-Hodgman), independent of mesh IDs."""
    output = [np.array(p, dtype=float) for p in subject]
    for a, b in zip(clipper, np.roll(clipper, -1, axis=0)):
        incoming, output = output, []
        if not incoming:
            break
        direction = b-a
        threshold = tolerance*max(float(np.linalg.norm(direction)), 1.)
        previous = incoming[-1]
        previous_value = float(_cross(direction, previous-a))
        for current in incoming:
            current_value = float(_cross(direction, current-a))
            previous_inside, current_inside = previous_value >= -threshold, current_value >= -threshold
            if previous_inside != current_inside:
                denominator = previous_value-current_value
                if denominator:
                    output.append(previous+(previous_value/denominator)*(current-previous))
            if current_inside:
                output.append(current)
            previous, previous_value = current, current_value
    return np.asarray(output).reshape(-1, 2)


def _edge_relation(a, b, c, d, tolerance):
    """Return proper crossing/positive collinear overlap/contact flags."""
    ab, cd = b-a, d-c
    lab, lcd = float(np.linalg.norm(ab)), float(np.linalg.norm(cd))
    if min(lab, lcd) <= tolerance:
        return False, False, False
    ta, tc = tolerance*lab, tolerance*lcd
    o1, o2 = float(_cross(ab, c-a)), float(_cross(ab, d-a))
    o3, o4 = float(_cross(cd, a-c)), float(_cross(cd, b-c))
    opposite_a = (o1 > ta and o2 < -ta) or (o2 > ta and o1 < -ta)
    opposite_c = (o3 > tc and o4 < -tc) or (o4 > tc and o3 < -tc)
    proper = opposite_a and opposite_c
    collinear = abs(o1) <= ta and abs(o2) <= ta and abs(o3) <= tc and abs(o4) <= tc
    overlap = False
    if collinear:
        projected = np.array([np.dot(c-a, ab/lab), np.dot(d-a, ab/lab)])
        overlap = min(lab, float(projected.max()))-max(0., float(projected.min())) > tolerance
    contact = False
    for p, start, end, value, limit in ((c,a,b,o1,ta),(d,a,b,o2,ta),(a,c,d,o3,tc),(b,c,d,o4,tc)):
        if abs(value) <= limit and np.all(p >= np.minimum(start,end)-tolerance) and np.all(p <= np.maximum(start,end)+tolerance):
            contact = True
    return proper, overlap, contact


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def validate_mesh(node_ids, xyz, elem_ids, conn, attrs, config):
    """Return JSON-serializable checks, quality metrics and bounded diagnostics.

    ``conn`` stores four actual node IDs. ``attrs`` must supply TYPE, MAT,
    SECNUM, THICKNESS, scalars or arrays aligned with elem_ids. Thickness must
    come from native section evidence, not merely the requested configuration.
    This function checks supplied evidence, not the solver's database itself.
    """
    checks, metrics, examples = {}, {}, {}
    result = {"passed": False, "checks": checks, "metrics": metrics, "examples": examples, "failures": []}

    def check(name, passed, detail=None):
        checks[name] = bool(passed)
        if detail is not None:
            examples[name] = detail

    def finish():
        result["failures"] = [name for name, passed in checks.items() if not passed]
        result["passed"] = bool(checks) and not result["failures"]
        return result

    try:
        nodes = np.asarray(node_ids, dtype=np.int64)
        points = np.asarray(xyz, dtype=float)
        elements = np.asarray(elem_ids, dtype=np.int64)
        connectivity = np.asarray(conn, dtype=np.int64)
        a, b = float(config["a"]), float(config["b"])
        tolerance = float(config.get("coordinate_tolerance_mm", 1e-8))
        quality = config.get("quality_limits", config.get("quality", {}))
        limits = {"aspect_ratio_max": float(quality.get("aspect_ratio_max", quality.get("max_aspect_ratio", 4.))),
                  "angle_min_degrees": float(quality.get("angle_min_degrees", quality.get("min_angle_degrees", 30.))),
                  "angle_max_degrees": float(quality.get("angle_max_degrees", quality.get("max_angle_degrees", 150.))),
                  "scaled_jacobian_min": float(quality.get("scaled_jacobian_min", .2))}
        metrics.update({"nodes": int(nodes.size), "elements": int(elements.size), "coordinate_tolerance_mm": tolerance,
                        "quality_limits": limits, "aspect_ratio_definition": "maximum planar Q4 Jacobian singular-value ratio at four corners and four Gauss points"})
        shape_ok = nodes.ndim == elements.ndim == 1 and points.shape == (len(nodes),3) and connectivity.shape == (len(elements),4) and len(nodes)>0 and len(elements)>0
        check("four_vertices_per_element", shape_ok)
        if not shape_ok:
            return finish()
        check("finite_coordinates", np.isfinite(points).all())
        check("unique_positive_node_ids", len(np.unique(nodes)) == len(nodes) and np.all(nodes > 0))
        check("unique_positive_element_ids", len(np.unique(elements)) == len(elements) and np.all(elements > 0))
        if not all(checks.values()):
            return finish()
        lookup = {int(node):index for index,node in enumerate(nodes)}
        missing = sorted(set(map(int,connectivity.ravel()))-lookup.keys())
        check("connectivity_nodes_exist", not missing, missing[:20])
        if missing:
            return finish()
        indices = np.array([[lookup[int(node)] for node in row] for row in connectivity])
        quads3 = points[indices]
        quads = quads3[:,:,:2]
        distinct = np.array([len(set(map(int,row))) == 4 for row in connectivity])
        check("four_distinct_vertices_no_triangles", distinct.all(), elements[~distinct].tolist()[:20])
        signatures = Counter(tuple(sorted(map(int,row))) for row in connectivity)
        duplicate_elements = [list(key) for key,value in signatures.items() if value > 1]
        check("no_duplicate_elements", not duplicate_elements, duplicate_elements[:20])
        used = set(map(int,connectivity.ravel()))
        unused = sorted(set(map(int,nodes))-used)
        check("no_unused_nodes", not unused, unused[:20])
        check("flat_z_zero", np.all(np.abs(points[:,2]) <= tolerance))
        check("inside_requested_rectangle", np.all(points[:,0]>=-tolerance) and np.all(points[:,0]<=a+tolerance) and np.all(points[:,1]>=-tolerance) and np.all(points[:,1]<=b+tolerance))
        duplicate_nodes = []
        order = np.argsort(points[:,0])
        for position, i in enumerate(order):
            end = int(np.searchsorted(points[order,0], points[i,0]+tolerance, side="right"))
            candidates = order[position+1:end]
            if candidates.size:
                duplicates = candidates[np.linalg.norm(points[candidates]-points[i],axis=1)<=tolerance]
                duplicate_nodes.extend((int(nodes[i]),int(nodes[j])) for j in duplicates)
        check("no_duplicate_nodes", not duplicate_nodes, duplicate_nodes[:20])
        metrics["duplicate_node_pairs"] = len(duplicate_nodes)

        gauss = 1/math.sqrt(3.)
        samples = [(-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.),(-gauss,-gauss),(gauss,-gauss),(gauss,gauss),(-gauss,gauss)]
        determinants, scales, aspects, normals_z = [], [], [], []
        for xi, eta in samples:
            dxi = .25*np.array([-(1-eta),1-eta,1+eta,-(1+eta)])
            deta = .25*np.array([-(1-xi),-(1+xi),1+xi,1-xi])
            gxi, geta = np.einsum("j,ijk->ik",dxi,quads),np.einsum("j,ijk->ik",deta,quads)
            det = _cross(gxi,geta)
            cross3=np.cross(np.einsum("j,ijk->ik",dxi,quads3),np.einsum("j,ijk->ik",deta,quads3))
            cross_norm=np.linalg.norm(cross3,axis=1)
            normals_z.append(np.divide(cross3[:,2],cross_norm,out=np.full(det.shape,np.nan),where=cross_norm>0))
            denominator = np.linalg.norm(gxi,axis=1)*np.linalg.norm(geta,axis=1)
            scaled = np.divide(det,denominator,out=np.full(det.shape,np.nan),where=denominator>0)
            singular = np.linalg.svd(np.stack((gxi,geta),axis=2),compute_uv=False)
            aspect = np.divide(singular[:,0],singular[:,1],out=np.full(det.shape,np.inf),where=singular[:,1]>0)
            determinants.append(det); scales.append(scaled); aspects.append(aspect)
        determinants, scales, aspects, normals_z = map(np.asarray,(determinants,scales,aspects,normals_z))
        sides = np.roll(quads,-1,axis=1)-quads
        side_lengths = np.linalg.norm(sides,axis=2)
        angles = []
        for corner in range(4):
            first = quads[:,(corner-1)%4]-quads[:,corner]
            second = quads[:,(corner+1)%4]-quads[:,corner]
            denominator=np.linalg.norm(first,axis=1)*np.linalg.norm(second,axis=1)
            cosine=np.divide(np.sum(first*second,axis=1),denominator,out=np.full(denominator.shape,np.nan),where=denominator>0)
            angles.append(np.degrees(np.arccos(np.clip(cosine,-1.,1.))))
        angles=np.asarray(angles)
        check("positive_jacobians_and_normals_z", np.isfinite(determinants).all() and np.all(determinants>0)
              and np.isfinite(normals_z).all() and np.all(normals_z>=1-1e-10))
        check("scaled_jacobian_threshold", np.isfinite(scales).all() and np.all(scales>=limits["scaled_jacobian_min"]-1e-12))
        check("aspect_ratio_threshold", np.isfinite(aspects).all() and np.all(aspects<=limits["aspect_ratio_max"]+1e-12))
        check("angle_threshold", np.isfinite(angles).all() and np.all(angles>=limits["angle_min_degrees"]-1e-10) and np.all(angles<=limits["angle_max_degrees"]+1e-10))
        metrics.update({"minimum_jacobian_mm2":_finite(np.min(determinants)),"minimum_scaled_jacobian":_finite(np.nanmin(scales)),
                        "maximum_aspect_ratio":_finite(np.max(aspects)),"minimum_angle_degrees":_finite(np.nanmin(angles)),"maximum_angle_degrees":_finite(np.nanmax(angles)),
                        "maximum_side_length_ratio":_finite(np.max(np.divide(side_lengths.max(axis=1),side_lengths.min(axis=1),out=np.full(len(elements),np.inf),where=side_lengths.min(axis=1)>0))),
                        "normal_z_minimum":_finite(np.nanmin(normals_z))})
        signed_areas = np.array([_area(q) for q in quads])
        metrics.update({"signed_area_mm2":float(signed_areas.sum()),"absolute_element_area_mm2":float(np.abs(signed_areas).sum()),"target_area_mm2":a*b})
        area_tolerance=max(1e-8,a*b*1e-10)
        check("rectangle_area", abs(signed_areas.sum()-a*b)<=area_tolerance)

        edge_use=defaultdict(list)
        for eid,row in zip(elements,connectivity):
            for start,end in zip(row,np.roll(row,-1)):
                edge_use[tuple(sorted((int(start),int(end))))].append((int(eid),int(start),int(end)))
        boundaries=[]; bad_edges=[]; bad_orientation=[]
        for edge,uses in edge_use.items():
            if len(uses)==1:
                boundaries.append(edge)
            elif len(uses)!=2:
                bad_edges.append({"edge":list(edge),"uses":len(uses)})
            elif uses[0][1:]!=tuple(reversed(uses[1][1:])):
                bad_orientation.append(list(edge))
        check("edge_multiplicity_at_most_two",not bad_edges,bad_edges[:20])
        check("shared_edges_opposite_orientation",not bad_orientation,bad_orientation[:20])
        sides_intervals={"bottom":[],"top":[],"left":[],"right":[]}; interior_unshared=[]
        for edge in boundaries:
            p,q=points[[lookup[n] for n in edge],:2]
            side=None
            for name,axis,target,parameter in (("bottom",1,0.,0),("top",1,b,0),("left",0,0.,1),("right",0,a,1)):
                if abs(p[axis]-target)<=tolerance and abs(q[axis]-target)<=tolerance:
                    side=(name,parameter); break
            if side is None:
                interior_unshared.append(list(edge))
            else:
                sides_intervals[side[0]].append(sorted((float(p[side[1]]),float(q[side[1]]))))
        check("unshared_edges_only_on_outer_boundary",not interior_unshared,interior_unshared[:20])
        boundary_coverage={}
        for side,intervals in sides_intervals.items():
            endpoint=a if side in ("top","bottom") else b
            cursor=0.; okay=bool(intervals)
            for lo,hi in sorted(intervals):
                okay &= abs(lo-cursor)<=tolerance and hi-lo>tolerance
                cursor=hi
            okay &= abs(cursor-endpoint)<=tolerance
            boundary_coverage[side]=bool(okay)
        check("boundary_covers_rectangle_once",all(boundary_coverage.values()),boundary_coverage)
        parent={int(e):int(e) for e in elements}
        def find(value):
            while parent[value]!=value:
                parent[value]=parent[parent[value]]; value=parent[value]
            return value
        for uses in edge_use.values():
            if len(uses)==2:
                parent[find(uses[0][0])]=find(uses[1][0])
        components=len({find(int(e)) for e in elements})
        euler=len(used)-len(edge_use)+len(elements)
        check("one_connected_element_component",components==1)
        check("disk_euler_identity",euler==1)
        metrics.update({"unique_edges":len(edge_use),"boundary_edges":len(boundaries),"connected_components":components,"euler_v_minus_e_plus_f":euler})

        unique_edges=list(edge_use)
        edge_points=np.array([[points[lookup[n],:2] for n in edge] for edge in unique_edges])
        edge_min,edge_max=edge_points.min(axis=1),edge_points.max(axis=1)
        hanging=[]
        for edge,(p,q),lo,hi in zip(unique_edges,edge_points,edge_min,edge_max):
            direction=q-p; length2=float(direction@direction)
            if length2<=tolerance*tolerance:
                continue
            candidates=np.flatnonzero(np.all(points[:,:2]>=lo-tolerance,axis=1)&np.all(points[:,:2]<=hi+tolerance,axis=1))
            relative=points[candidates,:2]-p
            parameter=relative@direction/length2
            distance=np.abs(_cross(direction,relative))/math.sqrt(length2)
            mask=(parameter>tolerance/math.sqrt(length2))&(parameter<1-tolerance/math.sqrt(length2))&(distance<=tolerance)
            hanging.extend({"node":int(nodes[index]),"edge":list(edge)} for index in candidates[mask])
        check("no_hanging_nodes",not hanging,hanging[:20]); metrics["node_on_foreign_edge_count"]=len(hanging)
        crossings=[]; collinear_overlaps=[]; unshared_contacts=[]
        for i,(edge,(p,q)) in enumerate(zip(unique_edges,edge_points)):
            candidates=np.flatnonzero(np.all(edge_max>=edge_min[i]-tolerance,axis=1)&np.all(edge_min<=edge_max[i]+tolerance,axis=1)&(np.arange(len(unique_edges))>i))
            for j in candidates:
                proper,overlap,contact=_edge_relation(p,q,*edge_points[j],tolerance)
                description=[list(edge),list(unique_edges[j])]
                if proper: crossings.append(description)
                if overlap: collinear_overlaps.append(description)
                if contact and not set(edge).intersection(unique_edges[j]): unshared_contacts.append(description)
        check("no_proper_edge_crossings",not crossings,crossings[:20])
        check("no_collinear_edge_overlaps",not collinear_overlaps,collinear_overlaps[:20])
        check("contacts_share_node_ids",not unshared_contacts,unshared_contacts[:20])
        metrics.update({"proper_edge_crossings":len(crossings),"collinear_edge_overlaps":len(collinear_overlaps),"unshared_edge_contacts":len(unshared_contacts)})
        cell_min,cell_max=quads.min(axis=1),quads.max(axis=1)
        overlaps=[]
        convex=np.all(determinants[:4]>0,axis=0)
        for i,quad in enumerate(quads):
            if not convex[i]:
                continue
            candidates=np.flatnonzero(np.all(cell_max>cell_min[i]+tolerance,axis=1)&np.all(cell_min<cell_max[i]-tolerance,axis=1)&(np.arange(len(elements))>i)&convex)
            for j in candidates:
                polygon=_clip(quad,quads[j],tolerance)
                intersection=abs(_area(polygon)) if len(polygon)>=3 else 0.
                if intersection>area_tolerance:
                    overlaps.append({"elements":[int(elements[i]),int(elements[j])],"intersection_area_mm2":intersection})
        check("no_overlapping_cell_interiors",not overlaps,overlaps[:20]); metrics["overlapping_element_pairs"]=len(overlaps)
        expected={"TYPE":1.,"MAT":1.,"SECNUM":1.,"THICKNESS":float(config["thickness"])}
        for name,wanted in expected.items():
            supplied=attrs.get(name)
            values=np.asarray(supplied,dtype=float) if supplied is not None else np.array([])
            if values.ndim==0:
                values=np.full(len(elements),float(values))
            check(f"preserved_{name.lower()}",values.shape==(len(elements),) and np.isfinite(values).all() and np.allclose(values,wanted,rtol=0,atol=1e-10))
            metrics[f"actual_{name.lower()}_values"]=[float(v) for v in np.unique(values) if np.isfinite(v)]
        boundary_nodes=np.flatnonzero((np.abs(points[:,0])<=tolerance)|(np.abs(points[:,0]-a)<=tolerance)|(np.abs(points[:,1])<=tolerance)|(np.abs(points[:,1]-b)<=tolerance))
        metrics["boundary_node_ids"]=nodes[boundary_nodes].tolist()
        metrics["boundary_nodes"]=int(len(boundary_nodes))
        metrics["expected_free_dof_p1_from_topology"]=int(6*(len(used)-len(set(map(int,nodes[boundary_nodes])).intersection(used))))
        result["limitations"]="Topology-based expected DoF only; no solver equation count. Native shape checks and native section/attribute evidence must also be supplied by the runner."
    except Exception as error:
        check("validation_completed",False,{"type":type(error).__name__,"message":str(error)})
    return finish()
