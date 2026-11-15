"""Numerical checks independent of ANSYS, using unittest and existing NumPy.

Run from the repository root with an interpreter providing NumPy:
    python -B -m unittest discover -s experiments/ansys_panel -p test_analysis.py
"""

import copy
import math
import unittest

import numpy as np

try:
    from .analysis import (
        bilinear_stress,
        disk_average,
        maximum_von_mises,
        navier_center,
        navier_convergence,
        navier_deflection,
        plate_rigidity,
        transition_convergence,
        two_transition_convergence,
        von_mises,
    )
except ImportError:
    from analysis import (
        bilinear_stress,
        disk_average,
        maximum_von_mises,
        navier_center,
        navier_convergence,
        navier_deflection,
        plate_rigidity,
        transition_convergence,
        two_transition_convergence,
        von_mises,
    )


class NavierChecks(unittest.TestCase):
    def test_center_benchmark_and_series_convergence(self):
        report = navier_convergence()
        self.assertTrue(report["converged"])
        self.assertAlmostEqual(report["central_deflection"], 0.570415, delta=1e-6)
        self.assertLess(report["rows"][-1]["relative_difference"], 1e-8)
        self.assertAlmostEqual(plate_rigidity(), 9846153.846153846)

    def test_support_boundary_symmetry_and_scaling(self):
        self.assertAlmostEqual(navier_deflection(0.0, 230.0), 0.0, places=14)
        self.assertAlmostEqual(navier_deflection(1000.0, 230.0), 0.0, places=14)
        left = navier_deflection(np.array([240.0, 320.0]), 215.0)
        reflected = navier_deflection(np.array([760.0, 680.0]), 385.0)
        np.testing.assert_allclose(left, reflected, rtol=1e-13, atol=1e-14)
        self.assertAlmostEqual(navier_center(pressure=0.010), 2.0 * navier_center())
        self.assertAlmostEqual(navier_center(thickness=16.0), navier_center() / 8.0)
        with self.assertRaises(ValueError):
            navier_deflection(-1.0, 30.0)


class PhysicalStressChecks(unittest.TestCase):
    def setUp(self):
        # A deliberately coarse, nonuniform grid; the disk crosses cells.
        self.x = np.array([0.0, 4.0, 11.0, 20.0])
        self.y = np.array([0.0, 7.0, 13.0, 20.0])
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="xy")

    def test_constant_general_3d_tensor_disk(self):
        vector = np.array([10.0, -3.0, 4.0, 2.0, 5.0, -1.0])
        stresses = np.broadcast_to(vector, (4, 4, 6))
        average = disk_average(self.x, self.y, stresses, (10.0, 10.0), 8.0)
        expected_vm = math.sqrt(0.5 * ((10 + 3) ** 2 + (-3 - 4) ** 2 + (4 - 10) ** 2) + 3 * (2**2 + 5**2 + 1**2))
        self.assertTrue(average["quadrature_converged"])
        self.assertAlmostEqual(average["mean_sx"], 10.0, places=12)
        self.assertAlmostEqual(average["mean_sy"], -3.0, places=12)
        self.assertAlmostEqual(average["mean_txy"], 2.0, places=12)
        self.assertAlmostEqual(average["mean_von_mises"], expected_vm, places=12)
        self.assertEqual(average["quadrature"]["radial_points"], 32)
        self.assertEqual(average["quadrature"]["sample_count"], 32 * 128)

    def test_affine_tensor_mean_and_vm_not_vm_of_mean(self):
        # sx=x-10 changes sign across the disk: VM=|x-10|. The mean
        # tensor is zero, but mean VM=4R/(3pi), from exact disk geometry.
        stresses = np.zeros((4, 4, 6))
        stresses[:, :, 0] = self.xx - 10.0
        result = disk_average(self.x, self.y, stresses, (10.0, 10.0), 8.0, rtol=1e-5)
        self.assertTrue(result["quadrature_converged"])
        self.assertAlmostEqual(result["mean_sx"], 0.0, places=12)
        self.assertAlmostEqual(result["mean_von_mises"], 32.0 / (3 * math.pi), delta=5e-5)
        self.assertGreater(result["mean_von_mises"], 3.0)

    def test_disk_fixed_geometry_and_full_tensor_interpolation(self):
        coarse_x, coarse_y = np.array([0.0, 20.0]), np.array([0.0, 20.0])
        def field(x, y):
            out = np.zeros(x.shape + (6,))
            out[..., 0] = 10 + x + 0.1 * x * y
            out[..., 1] = 2.0 * y - x
            out[..., 3] = 3.0 + 0.2 * x - 0.1 * y
            return out
        fine_tensor = field(self.xx, self.yy)
        cxx, cyy = np.meshgrid(coarse_x, coarse_y, indexing="xy")
        coarse_tensor = field(cxx, cyy)
        points = np.array([[5.3, 4.2], [20.0, 20.0], [0.0, 0.0]])
        np.testing.assert_allclose(bilinear_stress(self.x, self.y, fine_tensor, points), field(points[:, 0], points[:, 1]), rtol=1e-14)
        fine = disk_average(self.x, self.y, fine_tensor, (10.0, 10.0), 8.0)
        coarse = disk_average(coarse_x, coarse_y, coarse_tensor, (10.0, 10.0), 8.0)
        for metric in ("mean_sx", "mean_sy", "mean_txy", "mean_von_mises"):
            self.assertAlmostEqual(fine[metric], coarse[metric], places=11)
        self.assertAlmostEqual(fine["mean_sx"], 30.0, places=11)
        self.assertAlmostEqual(fine["mean_sy"], 10.0, places=11)
        self.assertAlmostEqual(fine["mean_txy"], 4.0, places=11)
        with self.assertRaises(ValueError):
            disk_average(self.x, self.y, fine_tensor, (1.0, 1.0), 8.0)

    def test_piecewise_field_requires_integration_refinement(self):
        # sx is the FE-style piecewise-linear interpolant of |x-11|.
        # The disk's exact integral can be integrated independently with
        # a dense 1-D trapezoidal slice integral 2 sqrt(R^2-u^2).
        stresses = np.zeros((4, 4, 6))
        stresses[:, :, 0] = np.abs(self.xx - 11.0)
        result = disk_average(self.x, self.y, stresses, (10.0, 10.0), 8.0, rtol=1e-5)
        self.assertTrue(result["quadrature_converged"])
        u = np.linspace(-8.0, 8.0, 200001)
        slice_values = np.abs(10.0 + u - 11.0) * 2 * np.sqrt(np.maximum(64.0 - u**2, 0.0))
        expected = np.trapezoid(slice_values, u) / (math.pi * 64.0) if hasattr(np, "trapezoid") else np.trapz(slice_values, u) / (math.pi * 64.0)
        self.assertAlmostEqual(result["mean_von_mises"], expected, delta=4e-5)
        self.assertGreater(result["quadrature"]["radial_points"], 32)
        capped = disk_average(self.x, self.y, stresses, (10.0, 10.0), 8.0, rtol=1e-14, max_levels=3)
        self.assertFalse(capped["quadrature_converged"])

    def test_clipped_rectangle_includes_interpolated_boundaries(self):
        stresses = np.zeros((4, 4, 6))
        stresses[:, :, 0] = self.xx + 2.0 * self.yy
        interior = maximum_von_mises(self.x, self.y, stresses, bounds=(2.0, 15.0, 3.0, 16.0))
        global_maximum = maximum_von_mises(self.x, self.y, stresses)
        self.assertAlmostEqual(interior["von_mises"], 47.0)
        self.assertEqual(interior["location_mm"], [15.0, 16.0])
        self.assertAlmostEqual(global_maximum["von_mises"], 60.0)
        # VM is convex, even when its tensor changes sign and direction.
        varying = stresses.copy()
        varying[..., 1] = 50.0 - 4.0 * self.xx
        varying[..., 4] = self.xx * self.yy / 20.0
        bounded = maximum_von_mises(self.x, self.y, varying, bounds=(2.0, 15.0, 3.0, 16.0))
        rng = np.random.default_rng(123)
        points = rng.uniform((2, 3), (15, 16), size=(10000, 2))
        sampled = von_mises(bilinear_stress(self.x, self.y, varying, points))
        self.assertLessEqual(float(sampled.max()), bounded["von_mises"] + 1e-12)


class ConvergenceChecks(unittest.TestCase):
    @staticmethod
    def case(value):
        disk = {"mean_sx": 0.0, "mean_sy": value, "mean_txy": 0.0, "mean_von_mises": value, "quadrature_converged": True}
        return {"central_deflection": value, "max_abs_uz": value, "strain_energy": value, "disks": {"top": {"center": disk}, "bottom": {"center": copy.deepcopy(disk)}}}

    def test_two_successive_transitions_and_regression(self):
        cases = [self.case(1.0), self.case(1.001), self.case(1.002)]
        result = two_transition_convergence(cases)
        self.assertTrue(result["converged"])
        self.assertEqual(result["consecutive_passes"], 2)
        self.assertFalse(two_transition_convergence(cases[:2])["converged"])
        self.assertFalse(two_transition_convergence(cases + [self.case(1.02)])["converged"])

    def test_zero_components_keep_absolute_difference_and_vm_scale(self):
        coarse, fine = self.case(10.0), self.case(10.0)
        fine["disks"]["top"]["center"]["mean_txy"] = 1e-6
        transition = transition_convergence(coarse, fine)
        change = transition["disks"]["top"]["center"]["changes"]["mean_txy"]
        self.assertEqual(change["absolute_difference_mpa"], 1e-6)
        self.assertEqual(change["normalization_scale_mpa"], 0.01)
        self.assertAlmostEqual(change["normalized_difference"], 1e-4)
        self.assertTrue(transition["passed"])

    def test_unconverged_quadrature_or_stress_prevents_reference(self):
        coarse, fine = self.case(10.0), self.case(10.0)
        fine["disks"]["bottom"]["center"]["quadrature_converged"] = False
        self.assertFalse(transition_convergence(coarse, fine)["passed"])
        fine["disks"]["bottom"]["center"]["quadrature_converged"] = True
        fine["disks"]["bottom"]["center"]["mean_von_mises"] = 10.2
        self.assertFalse(transition_convergence(coarse, fine)["passed"])


if __name__ == "__main__":
    unittest.main()
