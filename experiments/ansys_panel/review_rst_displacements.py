"""Compare saved RST nodal solutions with panel exports using an offline reader.

This script does not import ansys.mapdl.core, start a server, run MAPDL, install
packages, or change raw results. The existing ansys-mapdl-reader reads each
saved file.rst directly. A new output directory is required.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(path):
    stat = Path(path).stat()
    return {"sha256":sha(path), "bytes":stat.st_size, "mtime_ns":stat.st_mtime_ns}


def inspect_case(record, read_binary, import_error=None, tolerance=1e-10):
    runtime = Path(record["runtime_directory"])
    raw_paths = {name:runtime/name for name in ("file.rst", "fields.npz", "model.apdl")}
    checks, details, before, after = {}, {}, {}, {}

    def check(name, passed, **values):
        checks[name] = bool(passed)
        if values:
            details[name] = values

    try:
        before = {name:snapshot(path) for name,path in raw_paths.items()}
        if import_error:
            raise RuntimeError(import_error)
        # This installed legacy reader opens local binary files. VTK grid
        # parsing is unnecessary; the non-VTK mesh retains coordinates/angles.
        result = read_binary(raw_paths["file.rst"], flag_vtk_parse=False)
        times = np.asarray(result.time_values, dtype=float)
        ids, solution = result.nodal_solution(-1)
        ids, solution = np.asarray(ids, dtype=np.int64), np.asarray(solution)
        labels = list(result.result_dof(-1))
        check("finite_binary_solution", np.isfinite(solution).all() and np.isfinite(times).all())
        check("final_result_time_one", times.size > 0 and np.isclose(times[-1], 1, rtol=0, atol=1e-12))
        check("explicit_uz_dof_column", "UZ" in labels and solution.ndim == 2 and solution.shape[1] == len(labels), labels=labels)
        if "UZ" not in labels:
            raise ValueError("RST solution header does not identify a UZ column")
        with np.load(raw_paths["fields.npz"], allow_pickle=False) as fields:
            raw_uz = fields["uz"].copy()
            raw_xyz = fields["xyz"].copy()
        expected_ids = np.arange(1, len(raw_uz)+1, dtype=np.int64)
        order = np.argsort(ids)
        sorted_ids = ids[order]
        check("unique_node_ids_match_all_exported_nodes", np.array_equal(sorted_ids, expected_ids)
              and np.unique(ids).size == ids.size, rst_nodes=int(ids.size), exported_nodes=len(raw_uz))
        if not np.array_equal(sorted_ids, expected_ids):
            raise ValueError("Cannot compare all UZ: RST IDs differ from exported node IDs 1..N")
        rst_uz = solution[order, labels.index("UZ")]
        mesh_ids = np.asarray(result.mesh.nnum, dtype=np.int64)
        mesh_xyz = np.asarray(result.mesh.nodes)
        mesh_order = np.argsort(mesh_ids)
        check("binary_mesh_ids_match_exported_nodes", np.array_equal(mesh_ids[mesh_order], expected_ids))
        check("binary_mesh_coordinates_match_export", mesh_xyz.shape == raw_xyz.shape
              and np.allclose(mesh_xyz[mesh_order], raw_xyz, rtol=0, atol=1e-10))
        difference = np.abs(rst_uz-raw_uz)
        largest = int(np.argmax(difference))
        check("all_uz_match_export", np.isfinite(raw_uz).all() and np.max(difference) <= tolerance,
              maximum_absolute_difference_mm=float(difference[largest]), at_node=int(expected_ids[largest]), absolute_tolerance_mm=tolerance)
        center_nodes = np.flatnonzero(np.all(np.isclose(raw_xyz, [500,300,0], rtol=0, atol=1e-10), axis=1))
        check("unique_physical_center", center_nodes.size == 1)
        if center_nodes.size != 1:
            raise ValueError("No unique exported node at the physical panel center (500,300,0)")
        center = int(center_nodes[0])
        check("center_matches_published_measurement", abs(rst_uz[center]-record["central_deflection"]) <= tolerance,
              node=int(expected_ids[center]), xyz_mm=raw_xyz[center].tolist(), binary_uz_mm=float(rst_uz[center]),
              exported_uz_mm=float(raw_uz[center]), published_uz_mm=float(record["central_deflection"]),
              absolute_difference_mm=float(abs(rst_uz[center]-record["central_deflection"])))
        check("maximum_uz_matches_published_measurement", abs(float(np.abs(rst_uz).max())-record["max_abs_uz"]) <= tolerance)
        details["reader_result"] = {"class":f"{type(result).__module__}.{type(result).__name__}",
            "result_index":-1, "time_values":times.tolist(), "solution_shape":list(solution.shape),
            "mesh_coordinate_shape":list(mesh_xyz.shape), "uz_column":labels.index("UZ"),
            "maximum_absolute_binary_uz_mm":float(np.abs(rst_uz).max())}
        check("raw_hashes_match_published_record", before["fields.npz"]["sha256"] == record["fields_sha256"]
              and before["model.apdl"]["sha256"] == record["apdl_sha256"])
        del result
    except Exception as error:
        checks["offline_read_completed"] = False
        details["limitation_or_error"] = {"type":type(error).__name__, "message":str(error)}
    finally:
        try:
            after = {name:snapshot(path) for name,path in raw_paths.items()}
            check("raw_files_unchanged", bool(before) and before == after)
        except Exception as error:
            checks["raw_files_unchanged"] = False
            details["after_snapshot_error"] = {"type":type(error).__name__, "message":str(error)}
    return {"case":record["case"], "runtime_directory":str(runtime), "passed":bool(checks) and all(checks.values()),
            "checks":checks, "details":details, "raw_files_before":before, "raw_files_after":after}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Published run directory containing measurements.json")
    parser.add_argument("--output", type=Path, required=True, help="New audit directory")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = list(Path(__file__).parent.glob("*.py")) + [args.input/"measurements.json"]
    source_before = {str(path):snapshot(path) for path in source_paths}
    records = json.loads((args.input/"measurements.json").read_text(encoding="utf-8"))
    read_binary, import_error = None, None
    try:
        # Legacy local binary-file reader, distinct from PyMAPDL's live API.
        from ansys.mapdl.reader import read_binary
    except Exception as error:
        import_error = f"Existing offline reader unavailable: {type(error).__name__}: {error}"
    cases = []
    for record in records:
        result = inspect_case(record, read_binary, import_error)
        cases.append(result)
        print(f"{result['case']}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    source_after = {str(path):snapshot(path) for path in source_paths}
    packages = {}
    for name in ("ansys-mapdl-reader", "numpy", "pyvista"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    unchanged = source_before == source_after
    report = {"created_utc":datetime.now(timezone.utc).isoformat(), "case_count":len(cases),
        "passed":bool(cases) and all(case["passed"] for case in cases) and unchanged,
        "method":"Existing ansys.mapdl.reader.read_binary(file.rst), nodal_solution(-1), explicit result_dof UZ column, all IDs matched before comparison",
        "absolute_tolerance_mm":1e-10, "packages":packages, "python":sys.version,
        "source_files_unchanged":unchanged, "source_files_before":source_before, "source_files_after":source_after,
        "cases":cases, "limitation":"Confirms saved RST versus saved extraction/indexing; no new solve, independent FE solution, or proof of the P0/Navier discrepancy's origin"}
    with (args.output/"rst_displacement_audit.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(f"Offline RST audit: {len(cases)} cases; passed={report['passed']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
