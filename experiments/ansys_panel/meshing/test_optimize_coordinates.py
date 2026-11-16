"""Synthetic coordinate-repair tests; no MAPDL import or historical mesh use."""
import copy
import json
import unittest

import numpy as np

from optimize_coordinates import (evaluate_candidate, geometry, optimize_coordinates,
                                  prepare_problem, _constraint_values)
from validate_mesh import validate_mesh


def distorted_grid():
    xy = np.array([[x, y] for y in (0., 1., 2.) for x in (0., 1., 2.)])
    xy[4, 0] = 1.65
    ids = np.arange(11, 38, 3)
    conn = [[ids[j*3+i], ids[j*3+i+1], ids[(j+1)*3+i+1], ids[(j+1)*3+i]]
            for j in range(2) for i in range(2)]
    mesh = {"node_ids": ids.tolist(), "xyz_mm": np.column_stack((xy, np.zeros(9))).tolist(),
            "element_ids": [101, 107, 113, 127], "connectivity_node_ids": conn,
            "attributes": {"TYPE": [1]*4, "MAT": [1]*4, "SECNUM": [1]*4, "THICKNESS": [8]*4},
            "native_settings": {"element_name": 181, "thickness_mm": 8}}
    return mesh, {"a": 2., "b": 2., "thickness": 8., "coordinate_tolerance_mm": 1e-8}


class CoordinateRepair(unittest.TestCase):
    def test_condition_metric_matches_original_validator(self):
        mesh, config = distorted_grid()
        p = prepare_problem(mesh, config, [])
        old = validate_mesh(mesh["node_ids"], mesh["xyz_mm"], mesh["element_ids"],
                            mesh["connectivity_node_ids"], mesh["attributes"], config)
        self.assertAlmostEqual(p["original"]["condition"].max(), old["metrics"]["maximum_aspect_ratio"], places=12)
        self.assertGreater(old["metrics"]["maximum_aspect_ratio"], 4)
        self.assertTrue(evaluate_candidate(p, p["xy"])["passed"])

    def test_only_interior_node_and_original_minimum_incident_edge(self):
        mesh, config = distorted_grid()
        p = prepare_problem(mesh, config, [])
        self.assertEqual(p["nodes"][p["active_indices"]].tolist(), [23])
        self.assertAlmostEqual(p["hi"][4], .35)
        self.assertEqual(int(p["boundary"].sum()), 8)

    def test_fixed_protocol_optimizes_and_preserves_every_other_field(self):
        mesh, config = distorted_grid()
        saved = copy.deepcopy(mesh)
        output, audit = optimize_coordinates(mesh, config, [])
        self.assertEqual(mesh, saved)
        self.assertTrue(audit["optimizer_performed"])
        self.assertTrue(audit["accepted_coordinate_change"], audit)
        self.assertLess(audit["after"]["maximum_condition"], 4)
        self.assertTrue(audit["after"]["passed"])
        self.assertTrue(audit["independent_candidate_validation"]["passed"])
        self.assertLessEqual(audit["after"]["maximum_displacement_over_hi"], .2+1e-8)
        for key in mesh:
            if key != "xyz_mm":
                self.assertEqual(mesh[key], output[key])
        oldxyz, newxyz = np.asarray(mesh["xyz_mm"]), np.asarray(output["xyz_mm"])
        self.assertTrue(np.array_equal(oldxyz[:, 2], newxyz[:, 2]))
        self.assertTrue(np.array_equal(oldxyz[[0,1,2,3,5,6,7,8]], newxyz[[0,1,2,3,5,6,7,8]]))
        self.assertFalse(audit["solve_performed"])
        json.dumps(audit, allow_nan=False)

    def test_repeat_is_deterministic(self):
        mesh, config = distorted_grid()
        first, _ = optimize_coordinates(mesh, config, [])
        second, _ = optimize_coordinates(mesh, config, [])
        np.testing.assert_array_equal(first["xyz_mm"], second["xyz_mm"])

    def test_protection_uses_original_vertex_union_and_freezes_nodes(self):
        mesh, config = distorted_grid()
        quad = np.asarray(mesh["xyz_mm"])[[0,1,4,3], :2]
        p = prepare_problem(mesh, config, [quad])
        self.assertEqual(p["element_ids"][p["protected_elements"]].tolist(), [101])
        self.assertTrue(p["protected"][4])
        output, audit = optimize_coordinates(mesh, config, [quad])
        self.assertEqual(output, mesh)
        self.assertFalse(audit["optimizer_performed"])
        self.assertEqual(audit["reason"], "no movable nodes")

    def test_radial_cap_rejects_diagonal_that_individual_bounds_allow(self):
        mesh, config = distorted_grid()
        p = prepare_problem(mesh, config, [])
        candidate = p["xy"].copy()
        candidate[4] += .19*p["hi"][4]
        result = evaluate_candidate(p, candidate)
        self.assertFalse(result["checks"]["radial_displacement_cap"])

    def test_boundary_motion_is_rejected(self):
        mesh, config = distorted_grid()
        p = prepare_problem(mesh, config, [])
        candidate = p["xy"].copy(); candidate[0] += [.01, 0]
        self.assertFalse(evaluate_candidate(p, candidate)["checks"]["fixed_nodes_unchanged"])

    def test_original_jacobian_constraint_rejects_collapse(self):
        mesh, config = distorted_grid()
        p = prepare_problem(mesh, config, [])
        candidate = p["xy"].copy(); candidate[4] = [1.99, 1.]
        result = evaluate_candidate(p, candidate)
        self.assertFalse(result["checks"]["original_jacobian_fraction"])
        self.assertFalse(result["checks"]["edge_length_ratio_bounds"])

    def test_epigraph_starts_feasible(self):
        mesh, config = distorted_grid()
        p = prepare_problem(mesh, config, [])
        vector = np.zeros(2*len(p["active_indices"])+1)
        vector[-1] = p["original"]["condition"].max()
        self.assertGreaterEqual(_constraint_values(vector, p).min(), -1e-12)

    def test_no_bad_elements_does_not_optimize(self):
        mesh, config = distorted_grid()
        mesh["xyz_mm"][4][0] = 1.
        output, audit = optimize_coordinates(mesh, config, [])
        self.assertEqual(output, mesh)
        self.assertFalse(audit["optimizer_performed"])
        self.assertEqual(audit["bad_original_element_ids"], [])

    def test_relaxed_quality_configuration_is_rejected(self):
        mesh, config = distorted_grid()
        config["quality_limits"] = {"aspect_ratio_max": 5}
        with self.assertRaisesRegex(ValueError, "original"):
            prepare_problem(mesh, config, [])


if __name__ == "__main__":
    unittest.main()
