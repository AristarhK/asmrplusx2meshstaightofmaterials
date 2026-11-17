"""Offline mechanical/geometry tests; this module never starts ANSYS."""
from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from recovery import (RecoveryError, build_observation, compute_indicator,
                      constitutive_matrices, rotate_resultants)


def mesh_grid(x=None, y=None):
    x = np.asarray([0., 1., 2., 3.] if x is None else x, dtype=float)
    y = np.asarray([0., 1., 2., 3.] if y is None else y, dtype=float)
    xyz = [[float(xx), float(yy), 0.] for yy in y for xx in x]
    ids = [10+3*i for i in range(len(xyz))]
    cells = []
    for j in range(len(y)-1):
        for i in range(len(x)-1):
            a = j*len(x)+i
            cells.append([ids[a], ids[a+1], ids[a+1+len(x)], ids[a+len(x)]])
    return {"node_ids": ids, "xyz_mm": xyz, "element_ids": [101+7*i for i in range(len(cells))],
            "connectivity_node_ids": cells}


def centers(mesh):
    lookup = dict(zip(mesh["node_ids"], mesh["xyz_mm"]))
    return np.asarray([[lookup[n] for n in row] for row in mesh["connectivity_node_ids"]]).mean(axis=1)[:, :2]


def indicator(mesh, q, **kwargs):
    settings = {"young": 210000., "poisson": .3, "thickness": 8.,
                "shear_stiffness": np.eye(2)*(5./6.*210000./2.6*8.), "strain_energy": 100.}
    settings.update(kwargs)
    return compute_indicator(mesh, q, **settings)


class RecoveryTests(unittest.TestCase):
    def test_constant_field_zero_to_roundoff_and_serializable(self):
        mesh = mesh_grid()
        q0 = np.array([1., -2., .4, 7., -11., 3., 13., -19.])
        result = indicator(mesh, np.tile(q0, (9, 1)))
        np.testing.assert_allclose(result["recovered_nodal_resultants"], np.tile(q0, (16, 1)), rtol=1e-13, atol=1e-13)
        self.assertLess(result["sum_eta2"], 1e-25)
        self.assertTrue(result["quadrature"]["passed"])
        self.assertEqual(result["quadrature"]["accepted_order"], 3)
        self.assertEqual(result["element_ids"], mesh["element_ids"])
        json.dumps(result, allow_nan=False)

    def test_affine_field_boundary_patches_and_exact_integrated_energy(self):
        mesh = mesh_grid(x=[0., .4, 1.3, 3.], y=[0., .7, 1.9, 3.])
        c = centers(mesh)
        base = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
        gx = np.array([2., -1., .7, 6., -4., 3., -5., 8.])
        gy = np.array([-3., 4., 2., 1., 9., -2., 7., -6.])
        q = base+c[:, 0, None]*gx+c[:, 1, None]*gy
        result = indicator(mesh, q)
        xy = np.asarray(mesh["xyz_mm"])[:, :2]
        np.testing.assert_allclose(result["recovered_nodal_resultants"], base+xy[:, 0, None]*gx+xy[:, 1, None]*gy, atol=3e-13)
        self.assertTrue(all(row["rank"] == 3 for row in result["patches"]["nodes"]))
        self.assertGreater(result["patches"]["max_rings_added"], 0)
        self.assertLessEqual(result["patches"]["max_condition"], 1e8)
        a, d, s = constitutive_matrices(210000., .3, 8., 5./6.*210000./2.6*8.)
        inverse = np.zeros((8, 8))
        inverse[:3, :3], inverse[3:6, 3:6], inverse[6:8, 6:8] = np.linalg.inv(a), np.linalg.inv(d), np.linalg.inv(s)
        hx = np.tile(np.diff([0., .4, 1.3, 3.]), 3)
        hy = np.repeat(np.diff([0., .7, 1.9, 3.]), 3)
        expected = hx*hy/12.*(hx**2*(gx@inverse@gx)+hy**2*(gy@inverse@gy))
        np.testing.assert_allclose(result["eta2"], expected, rtol=5e-13, atol=1e-18)
        self.assertAlmostEqual(result["eta_rel"], np.sqrt(sum(expected)/200.), places=14)
        for order in result["quadrature"]["attempts"]:
            self.assertAlmostEqual(order["sum_eta2_N_mm"], sum(expected), places=14)

    def test_failed_rank_cannot_fall_back(self):
        mesh = mesh_grid(x=[0., 1., 2., 3.], y=[0., 1.])
        with self.assertRaisesRegex(RecoveryError, "rank=2"):
            indicator(mesh, np.ones((3, 8)))

    def test_explicit_shear_matrix_is_mandatory_and_spd(self):
        with self.assertRaises(TypeError):
            compute_indicator(mesh_grid(), np.ones((9, 8)), young=210000., poisson=.3,
                              thickness=8., strain_energy=10.)
        for bad in (-1., np.zeros((2, 2)), [[1., 2.], [0., 1.]]):
            with self.assertRaises(RecoveryError):
                constitutive_matrices(210000., .3, 8., bad)

    def test_tensor_rotation_preserves_energy_and_engineering_twist(self):
        angle = .731
        r = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        source = np.random.default_rng(71).normal(size=(15, 8))
        rotated = rotate_resultants(source, r)
        a, d, s = constitutive_matrices(210000., .3, 8., 500000.)
        for section, stiffness in zip((slice(0, 3), slice(3, 6), slice(6, 8)), (a, d, s)):
            inverse = np.linalg.inv(stiffness)
            original_energy = np.einsum("ei,ij,ej->e", source[:, section], inverse, source[:, section])
            rotated_energy = np.einsum("ei,ij,ej->e", rotated[:, section], inverse, rotated[:, section])
            np.testing.assert_allclose(rotated_energy, original_energy, rtol=1e-14)
        twist = np.array([0., 0., 0., 0., 0., 3., 0., 0.])
        _, d0, _ = constitutive_matrices(12., 0., 1., 1.)
        r45 = np.array([[1., -1.], [1., 1.]])/np.sqrt(2.)
        transformed = rotate_resultants(twist, r45)
        np.testing.assert_allclose(transformed[3:6], [-3., 3., 0.], atol=1e-15)
        self.assertAlmostEqual(twist[3:6]@np.linalg.inv(d0)@twist[3:6], 18.)
        self.assertAlmostEqual(transformed[3:6]@np.linalg.inv(d0)@transformed[3:6], 18.)
        r3 = np.eye(3)
        r3[:2, :2] = r
        np.testing.assert_allclose(rotate_resultants(source, r3), rotated)
        np.testing.assert_allclose(rotate_resultants(rotated, r.T), source, atol=2e-15)
        with self.assertRaises(RecoveryError):
            rotate_resultants(source, np.diag([1., -1.]))

    def test_area_scaling_with_no_component_weights(self):
        mesh = mesh_grid()
        q = np.random.default_rng(78).normal(size=(9, 8))
        result = indicator(mesh, q)
        scaled = copy.deepcopy(mesh)
        scaled["xyz_mm"] = (2.*np.asarray(mesh["xyz_mm"])).tolist()
        other = indicator(scaled, q)
        np.testing.assert_allclose(other["eta2"], 4.*np.asarray(result["eta2"]), rtol=1e-13)
        for part in ("membrane", "bending", "shear"):
            np.testing.assert_allclose(other["parts_per_element"][part],
                                       4.*np.asarray(result["parts_per_element"][part]), rtol=1e-13)
        # Pure Mxy sees compliance 2(1+nu)/(E*t^3/12), not a hand-tuned shear weight.
        a, d, s = constitutive_matrices(210000., .3, 8., 500000.)
        self.assertAlmostEqual(np.linalg.inv(d)[2, 2], 24.*1.3/(210000.*8.**3))

    def test_distorted_q4_independent_quadrature_agreement(self):
        mesh = mesh_grid(x=[0., .4, 1.3, 3.], y=[0., .7, 1.9, 3.])
        xy = np.asarray(mesh["xyz_mm"])
        xy[5, :2] += [.07, -.05]
        xy[6, :2] += [-.04, .06]
        xy[9, :2] += [-.08, .02]
        mesh["xyz_mm"] = xy.tolist()
        q = np.random.default_rng(23).normal(size=(9, 8))
        result = indicator(mesh, q)
        totals = [row["sum_eta2_N_mm"] for row in result["quadrature"]["attempts"]]
        self.assertEqual(len(totals), 2)
        self.assertLess(abs(totals[0]-totals[1])/totals[1], 1e-13)

    def test_area_weighted_least_squares_independent_calculation(self):
        mesh = mesh_grid(x=[0., .4, 1.3, 3.], y=[0., .7, 1.9, 3.])
        q = np.random.default_rng(129).normal(size=(9, 8))
        result = indicator(mesh, q)
        # Interior node (x=.4,y=.7) initially has four incident cells; nonuniform
        # areas make equal-weight recovery measurably different.
        node_row = 5
        elements = np.array([0, 1, 3, 4])
        delta = centers(mesh)[elements]-np.asarray(mesh["xyz_mm"])[node_row, :2]
        design = np.column_stack((np.ones(4), delta))
        areas = np.outer(np.diff([0., .7, 1.9, 3.]), np.diff([0., .4, 1.3, 3.])).ravel()[elements]
        expected = np.linalg.solve(design.T@(areas[:, None]*design), design.T@(areas[:, None]*q[elements]))[0]
        np.testing.assert_allclose(result["recovered_nodal_resultants"][node_row], expected, rtol=1e-13, atol=1e-13)
        unweighted = np.linalg.lstsq(design, q[elements], rcond=None)[0][0]
        self.assertGreater(np.linalg.norm(unweighted-expected), .01)

    def test_observation_graph_physics_normalizers_and_actual_ids(self):
        mesh = mesh_grid()
        q = np.random.default_rng(1).normal(size=(9, 8))
        result = indicator(mesh, q)
        config = {"a_mm": 3., "b_mm": 3., "pressure_N_per_mm2": .005,
                  "young_N_per_mm2": 210000., "poisson": .3, "thickness_mm": 8.}
        observation = build_observation(mesh, q, result, config)
        other = build_observation(mesh, q*100., indicator(mesh, q*100.), config)
        self.assertEqual(observation["normalization"], other["normalization"])
        self.assertEqual(observation["element_ids"], mesh["element_ids"])
        self.assertEqual(observation["eta2"], result["eta2"])
        self.assertEqual(len(observation["edge_index"][0]), 24)
        self.assertEqual(observation["geometry"]["neighbor_element_ids"][0], [108, 122])
        self.assertEqual(len(observation["node_features"][0]), len(observation["feature_names"]))
        self.assertEqual(len(observation["node_features"][0]), 18)
        np.testing.assert_allclose(np.asarray(observation["node_features"])[:, 6:14]*100.,
                                   np.asarray(other["node_features"])[:, 6:14])
        self.assertFalse(observation["training_or_inference_executed"])
        json.dumps(observation, allow_nan=False)
        wrong = copy.deepcopy(result)
        wrong["element_ids"] = list(reversed(wrong["element_ids"]))
        with self.assertRaises(RecoveryError):
            build_observation(mesh, q, wrong, config)

    def test_invalid_geometry_and_input_rejected(self):
        mesh = mesh_grid()
        mesh["connectivity_node_ids"][0].reverse()
        with self.assertRaisesRegex(RecoveryError, "Jacobian"):
            indicator(mesh, np.ones((9, 8)))
        with self.assertRaisesRegex(RecoveryError, "strain_energy"):
            indicator(mesh_grid(), np.ones((9, 8)), strain_energy=0.)


if __name__ == "__main__":
    unittest.main()
