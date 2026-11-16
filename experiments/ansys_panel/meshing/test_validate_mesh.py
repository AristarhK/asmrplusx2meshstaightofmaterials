"""Physical counterexamples for the independent Q4 topology/quality validator."""
import json
import unittest

import numpy as np
from validate_mesh import validate_mesh


def report(points, conn, a, b, node_ids=None, attrs=None):
    points=np.asarray(points,dtype=float)
    if points.shape[1]==2:
        points=np.column_stack((points,np.zeros(len(points))))
    node_ids=np.arange(1,len(points)+1) if node_ids is None else np.asarray(node_ids)
    return validate_mesh(node_ids,points,np.arange(101,101+len(conn)),conn,
                         {"TYPE":1,"MAT":1,"SECNUM":1,"THICKNESS":8} if attrs is None else attrs,
                         {"a":a,"b":b,"thickness":8,"coordinate_tolerance_mm":1e-8})


def structured(xs,ys,node_start=1,node_increment=1):
    xx,yy=np.meshgrid(xs,ys)
    points=np.column_stack((xx.ravel(),yy.ravel()))
    ids=node_start+node_increment*np.arange(len(points))
    ny,nx=len(ys)-1,len(xs)-1
    conn=[]
    for j in range(ny):
        for i in range(nx):
            k=j*(nx+1)+i
            conn.append(ids[[k,k+1,k+nx+2,k+nx+1]].tolist())
    return points,conn,ids


class ValidMeshes(unittest.TestCase):
    def test_original_panel(self):
        points,conn,ids=structured(np.arange(21)*50,np.arange(13)*50)
        result=report(points,conn,1000,600,ids)
        self.assertTrue(result["passed"],result["failures"])
        self.assertEqual(result["metrics"]["nodes"],273)
        self.assertEqual(result["metrics"]["elements"],240)
        self.assertEqual(result["metrics"]["expected_free_dof_p1_from_topology"],1254)
        self.assertEqual(result["metrics"]["boundary_nodes"],64)
        self.assertAlmostEqual(result["metrics"]["minimum_scaled_jacobian"],1)
        self.assertAlmostEqual(result["metrics"]["maximum_aspect_ratio"],1)
        self.assertAlmostEqual(result["metrics"]["signed_area_mm2"],600000)

    def test_noncontiguous_ids_and_graded_grid(self):
        points,conn,ids=structured([0,1,3],[0,1,2],node_start=10,node_increment=7)
        result=report(points,conn,3,2,ids)
        self.assertTrue(result["passed"],result["failures"])
        self.assertEqual(result["metrics"]["expected_free_dof_p1_from_topology"],6)
        self.assertAlmostEqual(result["metrics"]["maximum_aspect_ratio"],2)

    def test_three_quad_transition_patch(self):
        # Two adjacent bisected sides; three convex all-Q4 transition cells.
        points=[[0,0],[1,0],[1,1],[0,1],[.5,0],[1,.5],[.5,.5]]
        conn=[[5,2,6,7],[6,3,4,7],[4,1,5,7]]
        result=report(points,conn,1,1)
        self.assertTrue(result["passed"],result["failures"])
        self.assertEqual(result["metrics"]["expected_free_dof_p1_from_topology"],6)
        self.assertLess(result["metrics"]["maximum_aspect_ratio"],4)


class InvalidMeshes(unittest.TestCase):
    def test_degenerate_triangle(self):
        result=report([[0,0],[1,0],[1,1],[0,1]],[[1,2,3,3]],1,1)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["four_distinct_vertices_no_triangles"])
        json.dumps(result,allow_nan=False)

    def test_three_column_triangle(self):
        result=report([[0,0],[1,0],[1,1]],[[1,2,3]],1,1)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["four_vertices_per_element"])

    def test_hanging_node(self):
        points=[[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[1,.5],[2,.5]]
        result=report(points,[[1,2,5,4],[2,3,8,7],[7,8,6,5]],2,1)
        self.assertFalse(result["checks"]["no_hanging_nodes"])
        self.assertFalse(result["checks"]["unshared_edges_only_on_outer_boundary"])
        self.assertFalse(result["passed"])

    def test_duplicate_node_geometry(self):
        points=[[0,0],[1,0],[2,0],[0,1],[1,1],[2,1],[1,0]]
        result=report(points,[[1,2,5,4],[7,3,6,5]],2,1)
        self.assertFalse(result["checks"]["no_duplicate_nodes"])
        self.assertFalse(result["checks"]["contacts_share_node_ids"])

    def test_same_area_but_overlapping_and_uncovered_region(self):
        # Unit square plus area-.5 diamond gives requested area1.5, yet overlap.
        points=[[0,0],[1,0],[1,1],[0,1],[.3,.5],[.8,0],[1.3,.5],[.8,1]]
        result=report(points,[[1,2,3,4],[5,6,7,8]],1.5,1)
        self.assertTrue(result["checks"]["rectangle_area"])
        self.assertFalse(result["checks"]["no_proper_edge_crossings"])
        self.assertFalse(result["checks"]["no_overlapping_cell_interiors"])
        self.assertFalse(result["passed"])

    def test_nested_cell_overlap_without_edge_crossing(self):
        points=[[0,0],[2,0],[2,2],[0,2],[.5,.5],[1,.5],[1,1],[.5,1]]
        result=report(points,[[1,2,3,4],[5,6,7,8]],2,2)
        self.assertTrue(result["checks"]["no_proper_edge_crossings"])
        self.assertFalse(result["checks"]["no_overlapping_cell_interiors"])

    def test_clockwise_normal(self):
        result=report([[0,0],[1,0],[1,1],[0,1]],[[1,4,3,2]],1,1)
        self.assertFalse(result["checks"]["positive_jacobians_and_normals_z"])

    def test_quality_aspect_ratio_threshold_not_weakened(self):
        result=report([[0,0],[5,0],[5,1],[0,1]],[[1,2,3,4]],5,1)
        self.assertTrue(result["checks"]["boundary_covers_rectangle_once"])
        self.assertFalse(result["checks"]["aspect_ratio_threshold"])
        self.assertEqual(result["metrics"]["quality_limits"]["aspect_ratio_max"],4)

    def test_missing_or_wrong_native_section_evidence(self):
        points,conn,ids=structured([0,1],[0,1])
        missing=report(points,conn,1,1,ids,{"TYPE":1,"MAT":1,"SECNUM":1})
        wrong=report(points,conn,1,1,ids,{"TYPE":1,"MAT":1,"SECNUM":1,"THICKNESS":7})
        self.assertFalse(missing["checks"]["preserved_thickness"])
        self.assertFalse(wrong["checks"]["preserved_thickness"])

    def test_unused_node_is_not_counted_as_free_solver_dof(self):
        points=[[0,0],[1,0],[1,1],[0,1],[.5,.5]]
        result=report(points,[[1,2,3,4]],1,1)
        self.assertFalse(result["checks"]["no_unused_nodes"])
        self.assertEqual(result["metrics"]["expected_free_dof_p1_from_topology"],0)

    def test_duplicate_element(self):
        result=report([[0,0],[1,0],[1,1],[0,1]],[[1,2,3,4],[1,2,3,4]],1,1)
        self.assertFalse(result["checks"]["no_duplicate_elements"])


if __name__=="__main__":
    unittest.main()
