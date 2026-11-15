"""Numerical checks for the steel-panel experiment; no ANSYS dependency.

Stress arrays use the panel's global Cartesian coordinates, MPa, and shape
``(len(y_nodes), len(x_nodes), 6)``. The component order is
``[sx, sy, sz, txy, tyz, txz]``. Bilinear interpolation reconstructs the tensor;
von Mises stress is computed *after* interpolation. Consequently a disk's
mean von Mises stress is not the von Mises stress of its mean tensor.

All returned dictionaries contain ordinary JSON-compatible Python values.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss


CONTROL_DISKS = {
    "center": (500.0, 300.0),
    "near_y_edge": (500.0, 48.0),
    "near_x_edge": (48.0, 300.0),
}
STRESS_COMPONENTS = ("sx", "sy", "sz", "txy", "tyz", "txz")
DISK_METRICS = ("mean_sx", "mean_sy", "mean_txy", "mean_von_mises")


def plate_rigidity(
    thickness: float = 8.0, young: float = 210000.0, poisson: float = 0.30
) -> float:
    """Kirchhoff--Love plate rigidity D in N mm for N/mm/MPa units."""
    if not all(math.isfinite(v) for v in (thickness, young, poisson)):
        raise ValueError("Material and thickness must be finite")
    if thickness <= 0 or young <= 0 or not -1.0 < poisson < 0.5:
        raise ValueError("Require thickness > 0, E > 0, and -1 < nu < 0.5")
    return young * thickness**3 / (12.0 * (1.0 - poisson**2))


def navier_deflection(
    x: Any,
    y: Any,
    *,
    a: float = 1000.0,
    b: float = 600.0,
    thickness: float = 8.0,
    young: float = 210000.0,
    poisson: float = 0.30,
    pressure: float = 0.005,
    odd_terms: int = 80,
) -> float | np.ndarray:
    """Navier displacement in mm, positive in the pressure direction.

    ``odd_terms`` is the number of positive odd indices in EACH direction:
    1, 3, ..., 2*odd_terms-1. The inputs x and y broadcast like NumPy arrays.
    The formula is the independent Kirchhoff--Love thin-plate comparison for
    P0; it does not incorporate SHELL181 transverse-shear deformation.
    """
    if not isinstance(odd_terms, (int, np.integer)) or odd_terms < 1:
        raise ValueError("odd_terms must be a positive integer")
    if not all(math.isfinite(v) for v in (a, b, pressure)) or a <= 0 or b <= 0:
        raise ValueError("Require finite pressure and positive plate dimensions")
    rigidity = plate_rigidity(thickness, young, poisson)
    px, py = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if not np.isfinite(px).all() or not np.isfinite(py).all():
        raise ValueError("Evaluation coordinates must be finite")
    if (px < 0).any() or (px > a).any() or (py < 0).any() or (py > b).any():
        raise ValueError("Evaluation coordinates lie outside the plate")
    indices = np.arange(1, 2 * int(odd_terms), 2, dtype=float)
    # Sum n for each m: memory is O(odd_terms * number of evaluation points),
    # instead of storing all (m,n,point) combinations at once.
    sine_y = np.sin(indices[:, None] * math.pi * py.ravel()[None, :] / b)
    total = np.zeros(px.size, dtype=float)
    flat_x = px.ravel()
    for m in indices:
        denominator = m * indices * ((m / a) ** 2 + (indices / b) ** 2) ** 2
        total += np.sin(m * math.pi * flat_x / a) * np.sum(
            sine_y / denominator[:, None], axis=0
        )
    result = (16.0 * pressure / (math.pi**6 * rigidity) * total).reshape(px.shape)
    return float(result) if result.ndim == 0 else result


def navier_center(**kwargs: Any) -> float:
    """Convenience wrapper evaluating the panel center."""
    a, b = kwargs.get("a", 1000.0), kwargs.get("b", 600.0)
    return float(navier_deflection(a / 2.0, b / 2.0, **kwargs))


def navier_convergence(
    odd_terms: Sequence[int] = (4, 8, 16, 32, 64, 128),
    *,
    rtol: float = 1e-7,
    **kwargs: Any,
) -> dict[str, Any]:
    """Record the central series convergence, with a fine-value denominator.

    Convergence requires two consecutive changes <= rtol. Every supplied
    truncation is evaluated; the final value is a converged-series estimate,
    not a reference solution for the shear-deformable shell model.
    """
    terms = list(odd_terms)
    if len(terms) < 3 or any(n <= 0 for n in terms) or any(
        fine <= coarse for coarse, fine in zip(terms, terms[1:])
    ):
        raise ValueError("Supply at least three strictly increasing positive truncations")
    if not math.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be positive and finite")
    rows: list[dict[str, Any]] = []
    previous = None
    consecutive = 0
    for count in terms:
        value = navier_center(odd_terms=count, **kwargs)
        difference = None if previous is None else abs(value - previous)
        relative = None if difference is None else _relative_difference(value, previous)
        passed = relative is not None and relative <= rtol
        consecutive = consecutive + 1 if passed else 0
        rows.append(
            {
                "odd_terms_per_axis": int(count),
                "largest_odd_index": 2 * int(count) - 1,
                "central_deflection": value,
                "absolute_difference": difference,
                "relative_difference": relative,
                "transition_passed": bool(passed),
            }
        )
        previous = value
    return {
        "rows": rows,
        "central_deflection": rows[-1]["central_deflection"],
        "relative_tolerance": rtol,
        "consecutive_passes": consecutive,
        "converged": consecutive >= 2,
        "rigidity_n_mm": plate_rigidity(
            kwargs.get("thickness", 8.0), kwargs.get("young", 210000.0), kwargs.get("poisson", 0.30)
        ),
    }


def von_mises(stress: Any) -> np.ndarray:
    """3-D von Mises stress from global [sx,sy,sz,txy,tyz,txz]."""
    tensor = np.asarray(stress, dtype=float)
    if tensor.ndim < 1 or tensor.shape[-1] != 6:
        raise ValueError("The last stress-array dimension must contain six components")
    if not np.isfinite(tensor).all():
        raise ValueError("Stress components must be finite")
    sx, sy, sz, txy, tyz, txz = np.moveaxis(tensor, -1, 0)
    return np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
        + 3.0 * (txy**2 + tyz**2 + txz**2)
    )


def _grid(x_nodes: Any, y_nodes: Any, stress: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x_nodes, dtype=float)
    y = np.asarray(y_nodes, dtype=float)
    tensor = np.asarray(stress, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or min(x.size, y.size) < 2:
        raise ValueError("Each coordinate axis must be a one-dimensional array with >=2 nodes")
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(tensor).all():
        raise ValueError("Grid and stresses must be finite")
    if (np.diff(x) <= 0).any() or (np.diff(y) <= 0).any():
        raise ValueError("Grid coordinates must be strictly increasing")
    if tensor.shape != (y.size, x.size, 6):
        raise ValueError("Stress shape must be (len(y_nodes), len(x_nodes), 6)")
    return x, y, tensor


def _interpolate(x: np.ndarray, y: np.ndarray, stress: np.ndarray, points: np.ndarray) -> np.ndarray:
    px, py = points[:, 0], points[:, 1]
    ix = np.clip(np.searchsorted(x, px, side="right") - 1, 0, x.size - 2)
    iy = np.clip(np.searchsorted(y, py, side="right") - 1, 0, y.size - 2)
    tx = ((px - x[ix]) / (x[ix + 1] - x[ix]))[:, None]
    ty = ((py - y[iy]) / (y[iy + 1] - y[iy]))[:, None]
    return (
        (1.0 - tx) * (1.0 - ty) * stress[iy, ix]
        + tx * (1.0 - ty) * stress[iy, ix + 1]
        + (1.0 - tx) * ty * stress[iy + 1, ix]
        + tx * ty * stress[iy + 1, ix + 1]
    )


def bilinear_stress(x_nodes: Any, y_nodes: Any, stress: Any, points: Any) -> np.ndarray:
    """Interpolate the stress tensor at physical points of shape (...,2).

    The grid may be nonuniform. No nearest-node or element-center sampling is
    used. Extrapolation is deliberately rejected.
    """
    x, y, tensor = _grid(x_nodes, y_nodes, stress)
    locations = np.asarray(points, dtype=float)
    if locations.ndim < 1 or locations.shape[-1] != 2 or not np.isfinite(locations).all():
        raise ValueError("points must contain finite (x,y) pairs")
    flat = locations.reshape(-1, 2)
    if (
        (flat[:, 0] < x[0]).any()
        or (flat[:, 0] > x[-1]).any()
        or (flat[:, 1] < y[0]).any()
        or (flat[:, 1] > y[-1]).any()
    ):
        raise ValueError("Interpolation points lie outside the grid")
    return _interpolate(x, y, tensor, flat).reshape(locations.shape[:-1] + (6,))


def _relative_difference(current: float, previous: float) -> float:
    difference = abs(current - previous)
    return difference / abs(current) if current != 0.0 else (0.0 if difference == 0.0 else math.inf)


def _stress_changes(
    current: Mapping[str, float], previous: Mapping[str, float], *, scale_fraction: float
) -> dict[str, dict[str, float]]:
    """Stable stress changes, always retaining absolute MPa differences.

    Components use max(|new|, |old|, scale_fraction*max(VM_new,VM_old))
    as the scale. For a component near zero this is a VM-based absolute
    normalization, not a misleading component-relative percentage.
    """
    vm_scale = max(abs(current["mean_von_mises"]), abs(previous["mean_von_mises"]))
    changes = {}
    for name in DISK_METRICS:
        fine, coarse = float(current[name]), float(previous[name])
        scale = (
            abs(fine)
            if name == "mean_von_mises"
            else max(abs(fine), abs(coarse), scale_fraction * vm_scale)
        )
        difference = abs(fine - coarse)
        changes[name] = {
            "absolute_difference_mpa": difference,
            "normalization_scale_mpa": scale,
            "normalized_difference": difference / scale if scale else (0.0 if difference == 0 else math.inf),
        }
    return changes


def disk_average(
    x_nodes: Any,
    y_nodes: Any,
    stress: Any,
    center: Sequence[float],
    radius: float = 8.0,
    *,
    rtol: float = 5e-4,
    atol: float = 1e-9,
    initial_radial: int = 8,
    initial_angular: int = 32,
    max_levels: int = 7,
    consecutive_required: int = 2,
) -> dict[str, Any]:
    """Mean stresses on the SAME physical disk independently of mesh size.

    Map s=r^2/R^2 to [0,1]. Then dA/(pi R^2)=ds*dtheta/(2*pi):
    Gauss--Legendre quadrature in s and equally spaced angular midpoint
    quadrature therefore have area-uniform weights. Both orders double at
    every level. The defaults stop after two consecutive passing transitions
    or at 512 radial x 2048 angular points (1,048,576 samples/last level).

    A transition passes when the changes in sx, sy, txy AND mean VM obey
    ``abs_difference <= atol + rtol*scale``. For VM scale=|new mean VM|;
    for components scale=max(|old|,|new|,0.001*max(old/new mean VM)).
    Thus 5e-4 is a 0.05% VM integration-stability target, not a rigorous
    quadrature-error bound. The absolute floor atol is in MPa. If the cap is
    reached without convergence the result explicitly reports False.
    """
    x, y, tensor = _grid(x_nodes, y_nodes, stress)
    origin = np.asarray(center, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all() or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Require a finite center pair and positive radius")
    if origin[0] - radius < x[0] or origin[0] + radius > x[-1] or origin[1] - radius < y[0] or origin[1] + radius > y[-1]:
        raise ValueError("The complete averaging disk must lie inside the grid")
    if not math.isfinite(rtol) or rtol <= 0 or not math.isfinite(atol) or atol < 0:
        raise ValueError("Require rtol > 0 and atol >= 0, both finite")
    integer_values = (initial_radial, initial_angular, max_levels, consecutive_required)
    if any(not isinstance(v, (int, np.integer)) or v < 1 for v in integer_values):
        raise ValueError("Quadrature orders, level cap, and required passes must be positive integers")
    if max_levels < consecutive_required + 1:
        raise ValueError("max_levels must allow the required number of convergence transitions")
    previous = None
    changes = None
    consecutive = 0
    total_samples = 0
    levels = []
    means: dict[str, float] = {}
    for level in range(max_levels):
        radial_count = int(initial_radial * 2**level)
        angular_count = int(initial_angular * 2**level)
        abscissae, weights = leggauss(radial_count)
        radial = radius * np.sqrt((abscissae + 1.0) / 2.0)
        angles = (np.arange(angular_count) + 0.5) * (2.0 * math.pi / angular_count)
        offsets = radial[:, None, None] * np.stack((np.cos(angles), np.sin(angles)), axis=-1)[None, :, :]
        samples = _interpolate(x, y, tensor, (origin + offsets).reshape(-1, 2))
        area_weights = (weights / 2.0)[:, None] / angular_count
        sample_vm = von_mises(samples).reshape(radial_count, angular_count)
        components = samples.reshape(radial_count, angular_count, 6)
        mean_tensor = np.sum(components * area_weights[:, :, None], axis=(0, 1))
        means = {
            "mean_sx": float(mean_tensor[0]),
            "mean_sy": float(mean_tensor[1]),
            "mean_txy": float(mean_tensor[3]),
            "mean_von_mises": float(np.sum(sample_vm * area_weights)),
        }
        passed = False
        if previous is not None:
            changes = _stress_changes(means, previous, scale_fraction=1e-3)
            passed = all(
                change["absolute_difference_mpa"] <= atol + rtol * change["normalization_scale_mpa"]
                for change in changes.values()
            )
        consecutive = consecutive + 1 if passed else 0
        sample_count = radial_count * angular_count
        total_samples += sample_count
        levels.append(
            {
                "radial_points": radial_count,
                "angular_points": angular_count,
                "sample_count": sample_count,
                **means,
                "transition_passed": passed,
                "changes": changes,
            }
        )
        previous = means
        if consecutive >= consecutive_required:
            break
    return {
        "center_mm": origin.tolist(),
        "radius_mm": float(radius),
        "area_mm2": math.pi * radius**2,
        **means,
        "quadrature_converged": consecutive >= consecutive_required,
        "quadrature": {
            "method": "Gauss-Legendre in s=r^2/R^2; periodic midpoint in theta; bilinear tensor interpolation",
            "relative_tolerance": rtol,
            "absolute_tolerance_mpa": atol,
            "component_scale_floor_fraction": 1e-3,
            "consecutive_required": int(consecutive_required),
            "consecutive_passes": consecutive,
            "radial_points": radial_count,
            "angular_points": angular_count,
            "sample_count": sample_count,
            "total_sample_count": total_samples,
            "changes": changes,
            "levels": levels,
        },
    }


def maximum_von_mises(
    x_nodes: Any, y_nodes: Any, stress: Any, *, bounds: Sequence[float] | None = None
) -> dict[str, Any]:
    """Maximum VM of the bilinear tensor field over a rectangle.

    ``bounds=(xmin,xmax,ymin,ymax)`` can cut cells. Evaluate every corner of
    each clipped cell, including interpolated boundary intersections. Each
    interior tensor is a convex combination of its four corner tensors;
    von Mises is convex, so no larger value occurs inside the rectangle.
    This is exact for this reconstructed field, not a bound on FE stress
    recovery error or an unexported integration-point field.
    """
    x, y, tensor = _grid(x_nodes, y_nodes, stress)
    rectangle = np.asarray(bounds if bounds is not None else (x[0], x[-1], y[0], y[-1]), dtype=float)
    if rectangle.shape != (4,) or not np.isfinite(rectangle).all():
        raise ValueError("bounds must be finite (xmin,xmax,ymin,ymax)")
    xlo, xhi, ylo, yhi = rectangle
    if not (x[0] <= xlo < xhi <= x[-1] and y[0] <= ylo < yhi <= y[-1]):
        raise ValueError("Bounds must form a nonempty rectangle inside the grid")
    clipped_x = np.concatenate(([xlo], x[(x > xlo) & (x < xhi)], [xhi]))
    clipped_y = np.concatenate(([ylo], y[(y > ylo) & (y < yhi)], [yhi]))
    xx, yy = np.meshgrid(clipped_x, clipped_y, indexing="xy")
    points = np.column_stack((xx.ravel(), yy.ravel()))
    vm = von_mises(_interpolate(x, y, tensor, points))
    index = int(np.argmax(vm))
    return {
        "von_mises": float(vm[index]),
        "location_mm": points[index].tolist(),
        "bounds_mm": rectangle.tolist(),
        "sample_count": int(vm.size),
        "method": "maximum at corners of clipped cells of bilinear global tensor reconstruction",
    }


def stress_metrics(
    x_nodes: Any,
    y_nodes: Any,
    stress: Any,
    *,
    margin: float = 24.0,
    radius: float = 8.0,
    centers: Mapping[str, Sequence[float]] | None = None,
    **quadrature_options: Any,
) -> dict[str, Any]:
    """One shell surface's physical-disk averages and diagnostic maxima."""
    x, y, tensor = _grid(x_nodes, y_nodes, stress)
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be finite and nonnegative")
    disk_centers = CONTROL_DISKS if centers is None else centers
    return {
        "disks": {
            name: disk_average(x, y, tensor, center, radius, **quadrature_options)
            for name, center in disk_centers.items()
        },
        "interior_maximum_von_mises": maximum_von_mises(
            x, y, tensor, bounds=(x[0] + margin, x[-1] - margin, y[0] + margin, y[-1] - margin)
        ),
        "global_maximum_von_mises": maximum_von_mises(x, y, tensor),
    }


def transition_convergence(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    """Check one mesh refinement; a reference needs TWO passing transitions.

    Case dictionaries contain central_deflection, max_abs_uz, strain_energy,
    and disks={surface: {name: disk_average_result}}. Displacements use 0.2%,
    energy 0.5%, and all disk mean VM values 1%, normalized by the fine value.
    Component changes are diagnostic, normalized using the documented
    max(|old|,|new|,0.001*max(old/new disk VM)) MPa scale, with absolute MPa
    changes retained. Their percentages are not extra convergence gates.
    """
    values: dict[str, Any] = {}
    for key, limit in (("central_deflection", 0.002), ("max_abs_uz", 0.002), ("strain_energy", 0.005)):
        fine, coarse = float(current[key]), float(previous[key])
        if not math.isfinite(fine) or not math.isfinite(coarse):
            raise ValueError("Convergence metrics must be finite")
        relative = _relative_difference(fine, coarse)
        values[key] = {
            "absolute_difference": abs(fine - coarse),
            "relative_difference": relative,
            "relative_tolerance": limit,
            "passed": relative <= limit,
        }
    if not current["disks"] or set(current["disks"]) != set(previous["disks"]):
        raise ValueError("The same nonempty set of surfaces must be compared")
    disks: dict[str, Any] = {}
    integration_converged = True
    for surface, fine_disks in current["disks"].items():
        coarse_disks = previous["disks"][surface]
        if not fine_disks or set(fine_disks) != set(coarse_disks):
            raise ValueError("The same nonempty set of disk names must be compared")
        disks[surface] = {}
        for name, fine_disk in fine_disks.items():
            coarse_disk = coarse_disks[name]
            changes = _stress_changes(fine_disk, coarse_disk, scale_fraction=1e-3)
            for metric in DISK_METRICS:
                if not math.isfinite(float(fine_disk[metric])) or not math.isfinite(float(coarse_disk[metric])):
                    raise ValueError("Disk metrics must be finite")
            quadrature_ok = bool(fine_disk["quadrature_converged"] and coarse_disk["quadrature_converged"])
            integration_converged &= quadrature_ok
            relative = changes["mean_von_mises"]["normalized_difference"]
            disks[surface][name] = {
                "changes": changes,
                "mean_von_mises_relative_tolerance": 0.01,
                "mean_von_mises_passed": relative <= 0.01,
                "quadrature_converged": quadrature_ok,
            }
    passed = (
        all(item["passed"] for item in values.values())
        and all(disk["mean_von_mises_passed"] for surface in disks.values() for disk in surface.values())
        and integration_converged
    )
    return {
        "passed": passed,
        "values": values,
        "disks": disks,
        "quadrature_converged": integration_converged,
        "component_normalization": "max(abs(fine component),abs(coarse component),0.001*max(fine/coarse mean VM)); MPa absolute differences retained",
        "relative_normalization": "absolute difference divided by absolute fine-mesh value",
    }


def two_transition_convergence(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report stabilization at the LAST available mesh of one formulation.

    This does not check load balance or solver success; orchestration must
    accept only successful, balanced solves. A passing pair demonstrates
    stabilization under the stated criteria, not a proven discretization
    error bound. Earlier passing pairs are not substituted for a failing
    transition at the final mesh.
    """
    transitions = [transition_convergence(old, new) for old, new in zip(cases, cases[1:])]
    consecutive = 0
    for transition in reversed(transitions):
        if not transition["passed"]:
            break
        consecutive += 1
    return {
        "transitions": transitions,
        "consecutive_passes": consecutive,
        "converged": consecutive >= 2,
        "required_consecutive_passes": 2,
    }
