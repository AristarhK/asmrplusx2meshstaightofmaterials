"""Independent Cartesian-slice disk quadrature for saved panel stresses.

This module does not import the production analysis module, ANSYS, a result
reader, or a launcher. It independently interpolates six global stress
components and forms the 3-D von Mises stress before area integration.

With x=cx+R*sin(theta), theta in [-pi/2,pi/2], the vertical slice is
cy-R*cos(theta) <= y <= cy+R*cos(theta), and dx=R*cos(theta)dtheta.
Split theta at every x gridline and every circle crossing of a y gridline.
Each remaining slice interval therefore has a fixed set of y gridlines;
split the vertical integral at those gridlines. The tensor is bilinear on
each integration piece. Gauss--Legendre orders control theta and y
separately. This differs from the production polar quadrature and handles
its piecewise-bilinear derivative discontinuities explicitly.

Run numerical checks without ANSYS:
    python -B experiments/ansys_panel/review_quadrature.py --self-test
"""

from __future__ import annotations

import argparse
import math
import unittest

import numpy as np


def _inputs(xs, ys, tensor, center, radius):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    stresses = np.asarray(tensor, dtype=float)
    c = np.asarray(center, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2:
        raise ValueError("Coordinate axes must be 1-D and have at least two nodes")
    if stresses.shape != (len(y), len(x), 6):
        raise ValueError("Expected tensor shape (len(ys), len(xs), 6)")
    if c.shape != (2,) or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Require a finite center pair and positive radius")
    if not all(np.isfinite(a).all() for a in (x, y, stresses, c)):
        raise ValueError("All input values must be finite")
    if np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0):
        raise ValueError("Grid coordinates must be strictly increasing")
    if c[0] - radius < x[0] or c[0] + radius > x[-1] or c[1] - radius < y[0] or c[1] + radius > y[-1]:
        raise ValueError("Complete disk must lie inside the grid")
    return x, y, stresses, c


def _independent_tensor(xs, ys, tensors, px, py):
    """Vectorized bilinear interpolation; no production helper is used."""
    xcell = np.searchsorted(xs, px, side="right") - 1
    ycell = np.searchsorted(ys, py, side="right") - 1
    xcell = np.clip(xcell, 0, len(xs) - 2)
    ycell = np.clip(ycell, 0, len(ys) - 2)
    u = (px - xs[xcell]) / (xs[xcell + 1] - xs[xcell])
    v = (py - ys[ycell]) / (ys[ycell + 1] - ys[ycell])
    lower = tensors[ycell, xcell] + u[:, None] * (
        tensors[ycell, xcell + 1] - tensors[ycell, xcell]
    )
    upper = tensors[ycell + 1, xcell] + u[:, None] * (
        tensors[ycell + 1, xcell + 1] - tensors[ycell + 1, xcell]
    )
    return lower + v[:, None] * (upper - lower)


def _independent_mises(tensors):
    """Equivalent stress formed as sqrt(3/2 * deviator:deviator)."""
    # This invariant form is algebraically equivalent to the production
    # implementation's pairwise-normal-difference form, written separately.
    trace_third = np.sum(tensors[:, :3], axis=1) / 3.0
    deviatoric_norm_squared = np.sum(
        (tensors[:, :3] - trace_third[:, None]) ** 2, axis=1
    ) + 2.0 * np.sum(tensors[:, 3:] ** 2, axis=1)
    return np.sqrt(1.5 * deviatoric_norm_squared)


def cartesian_disk(
    xs, ys, tensor, center, radius=8.0, *, theta_order=32, y_order=12
):
    """Independently integrate one disk, returning JSON-compatible metrics.

    Component order is [sx,sy,sz,txy,tyz,txz], in global MPa. Coordinate
    units are mm. Means divide by the analytic pi*R^2 area; computed area
    is separately returned as a quadrature diagnostic. To check numerical
    stabilization, compare e.g. orders (16,8), (32,16), and (64,24).
    A single call does not set a convergence flag or claim an error bound.
    """
    x, y, stresses, c = _inputs(xs, ys, tensor, center, radius)
    if any(not isinstance(n, (int, np.integer)) or n < 1 for n in (theta_order, y_order)):
        raise ValueError("Quadrature orders must be positive integers")
    # Angular events: slice crosses an x gridline; or either circle edge
    # crosses a horizontal gridline. All events are physical coordinates.
    interior_x = x[(x > c[0] - radius) & (x < c[0] + radius)]
    interior_y = y[(y > c[1] - radius) & (y < c[1] + radius)]
    events = [-math.pi / 2.0, math.pi / 2.0]
    events.extend(np.arcsin((interior_x - c[0]) / radius).tolist())
    y_angles = np.arccos(np.clip(np.abs(interior_y - c[1]) / radius, 0.0, 1.0))
    events.extend(y_angles.tolist())
    events.extend((-y_angles).tolist())
    breaks = np.unique(np.asarray(events))
    theta_nodes, theta_weights = np.polynomial.legendre.leggauss(int(theta_order))
    y_nodes, y_weights = np.polynomial.legendre.leggauss(int(y_order))
    totals = np.zeros(4)
    computed_area = 0.0
    sample_count = 0
    piece_count = 0
    for angle_start, angle_end in zip(breaks[:-1], breaks[1:]):
        if angle_end - angle_start <= 16.0 * np.finfo(float).eps:
            continue
        theta_mid = (angle_start + angle_end) / 2.0
        theta_half = (angle_end - angle_start) / 2.0
        theta = theta_mid + theta_half * theta_nodes
        half_height = radius * np.cos(theta)
        px = c[0] + radius * np.sin(theta)
        low = c[1] - half_height
        high = c[1] + half_height
        # The event split makes this interior-gridline set constant on the
        # open theta interval, so no moving grid crossing is left unhandled.
        midpoint_height = radius * math.cos(theta_mid)
        fixed_y = y[(y > c[1] - midpoint_height) & (y < c[1] + midpoint_height)]
        boundaries = np.column_stack(
            (low, np.broadcast_to(fixed_y, (len(theta), len(fixed_y))), high)
        )
        lower = boundaries[:, :-1]
        upper = boundaries[:, 1:]
        half_segment = (upper - lower) / 2.0
        if np.any(half_segment < -1e-11):
            raise RuntimeError("Angular event splitting left an invalid vertical segment")
        py = ((lower + upper) / 2.0)[:, :, None] + half_segment[:, :, None] * y_nodes
        x_points = np.broadcast_to(px[:, None, None], py.shape)
        values = _independent_tensor(x, y, stresses, x_points.ravel(), py.ravel())
        metrics = np.column_stack(
            (values[:, 0], values[:, 1], values[:, 3], _independent_mises(values))
        )
        area_weights = (
            theta_half
            * theta_weights[:, None, None]
            * half_height[:, None, None]
            * half_segment[:, :, None]
            * y_weights
        )
        totals += np.sum(metrics * area_weights.ravel()[:, None], axis=0)
        computed_area += float(np.sum(area_weights))
        sample_count += int(area_weights.size)
        piece_count += int(half_segment.shape[1])
    analytic_area = math.pi * radius**2
    means = totals / analytic_area
    return {
        "mean_sx": float(means[0]),
        "mean_sy": float(means[1]),
        "mean_txy": float(means[2]),
        "mean_von_mises": float(means[3]),
        "center_mm": c.tolist(),
        "radius_mm": float(radius),
        "computed_area_mm2": computed_area,
        "analytic_area_mm2": analytic_area,
        "area_relative_error": abs(computed_area - analytic_area) / analytic_area,
        "theta_order": int(theta_order),
        "y_order": int(y_order),
        "theta_intervals": int(len(breaks) - 1),
        "vertical_pieces": piece_count,
        "sample_count": sample_count,
        "method": "grid-split Cartesian slices; x=cx+R*sin(theta); independent bilinear tensor and deviatoric-norm VM",
    }


def _self_tests():
    class SliceTests(unittest.TestCase):
        def setUp(self):
            self.x = np.array([0.0, 4.0, 10.0, 11.0, 20.0])
            self.y = np.array([0.0, 7.0, 10.0, 13.0, 20.0])
            self.xx, self.yy = np.meshgrid(self.x, self.y)

        def test_constant_and_area(self):
            vector = np.array([10.0, -3.0, 4.0, 2.0, 5.0, -1.0])
            field = np.broadcast_to(vector, self.xx.shape + (6,))
            result = cartesian_disk(self.x, self.y, field, (10.0, 10.0), theta_order=16, y_order=8)
            self.assertLess(result["area_relative_error"], 2e-14)
            for name, expected in (("mean_sx", 10.0), ("mean_sy", -3.0), ("mean_txy", 2.0)):
                self.assertAlmostEqual(result[name], expected, places=12)
            expected_vm = math.sqrt(((10+3)**2+(-3-4)**2+(4-10)**2)/2 + 3*(2**2+5**2+1**2))
            self.assertAlmostEqual(result["mean_von_mises"], expected_vm, places=12)

        def test_affine_components_and_tensor_order(self):
            field = np.zeros(self.xx.shape + (6,))
            field[..., 0] = 10 + self.xx + 0.1 * self.xx * self.yy
            field[..., 1] = 2 * self.yy - self.xx
            field[..., 3] = 3 + 0.2 * self.xx - 0.1 * self.yy
            low = cartesian_disk(self.x, self.y, field, (10.0, 10.0), theta_order=16, y_order=8)
            high = cartesian_disk(self.x, self.y, field, (10.0, 10.0), theta_order=32, y_order=16)
            self.assertAlmostEqual(high["mean_sx"], 30.0, places=12)
            self.assertAlmostEqual(high["mean_sy"], 10.0, places=12)
            self.assertAlmostEqual(high["mean_txy"], 4.0, places=12)
            self.assertAlmostEqual(low["mean_von_mises"], high["mean_von_mises"], places=11)

        def test_mean_vm_is_not_vm_of_mean_tensor(self):
            field = np.zeros(self.xx.shape + (6,))
            field[..., 0] = self.xx - 10.0
            result = cartesian_disk(self.x, self.y, field, (10.0, 10.0), theta_order=16, y_order=8)
            self.assertAlmostEqual(result["mean_sx"], 0.0, places=13)
            self.assertAlmostEqual(result["mean_von_mises"], 32.0 / (3 * math.pi), places=12)

        def test_piecewise_linear_grid_crossing(self):
            # |x-11| has a kink on a gridline away from the disk center.
            field = np.zeros(self.xx.shape + (6,))
            field[..., 0] = np.abs(self.xx - 11.0)
            low = cartesian_disk(self.x, self.y, field, (10.0, 10.0), theta_order=16, y_order=8)
            high = cartesian_disk(self.x, self.y, field, (10.0, 10.0), theta_order=32, y_order=16)
            self.assertAlmostEqual(low["mean_von_mises"], high["mean_von_mises"], places=12)
            # Independent closed form for integral of |u-d| over a disk.
            r, d = 8.0, 1.0
            exact = 2.0 / (math.pi * r*r) * (
                (2*r*r + d*d) * math.sqrt(r*r-d*d) / 3.0
                + d*r*r*math.asin(d/r)
            )
            self.assertAlmostEqual(high["mean_von_mises"], exact, places=12)

        def test_requires_whole_disk_and_finite_grid(self):
            field = np.zeros(self.xx.shape + (6,))
            with self.assertRaises(ValueError):
                cartesian_disk(self.x, self.y, field, (1.0, 1.0))
            field[0, 0, 0] = math.nan
            with self.assertRaises(ValueError):
                cartesian_disk(self.x, self.y, field, (10.0, 10.0))

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SliceTests)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test; import cartesian_disk for saved-array reviews")
    return 0 if _self_tests() else 1


if __name__ == "__main__":
    raise SystemExit(main())
