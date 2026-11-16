"""Rebuild coordinate-improved Q4 meshes in fresh MAPDL, without a solution.

Only offline meshes with an explicit passing geometry gate are eligible. Native
shape checks remain enabled at their fresh-session limits. Full databases, CDBs,
APDL and native logs stay in a new private runtime; compact evidence is written
exclusively to a new output directory. This program never refines a mesh.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np

from publish_mesh_gate import compact_loads, expand_loads, quality, read_json
from run_mesh_gate import (boundary_ids, cdb_commands, digest, input_commands,
                           load_check, save_json, snapshot)


LIMITS = {"aspect_ratio_max": 4., "angle_min_degrees": 30.,
          "angle_max_degrees": 150., "scaled_jacobian_min": .2}
SHAPE_NAMES = ("Aspect Ratio", "Parallel Deviation", "Maximum Angle",
               "Jacobian Ratio", "Warping Factor", "Any")
EXPECTED_CASES = {"interior_single", "edge_patch", "corner_patch"}
FORBIDDEN = re.compile(r"\s*(SOLVE|LSSOLVE|MFSOLVE|PSOLVE)\s*(,|$)", re.I)
PATH_PATTERN = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|/Users/|/home/|/tmp/)")


def verify_manifest(directory):
    """Verify published provenance without executing or changing source files."""
    root = directory.resolve()
    manifest = root / "artifact_manifest.json"
    for name, sha in read_json(manifest)["files"].items():
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file() or digest(path) != sha:
            raise ValueError(f"Artifact integrity check failed: {Path(name).name}")


def portable_text(text):
    """Remove timing/machine annotations from diagnostic message excerpts."""
    clean = []
    for line in str(text).splitlines():
        if PATH_PATTERN.search(line):
            clean.append("[private path omitted]")
            continue
        if re.search(r"\b(?:COMPUTER|HOST|USERNAME|WORKING DIRECTORY|JOBNAME|LICENSE SERVER)\b", line, re.I):
            continue
        line = re.sub(r"(\*\*\*\s+(?:NOTE|WARNING|ERROR|FATAL)\s+\*\*\*)[^\r\n]*", r"\1", line)
        clean.append(line.rstrip())
    return "\n".join(clean).strip()


def native_messages(text):
    """Retain diagnostic bodies, never the native product/machine banner."""
    lines = str(text).splitlines()
    messages = []
    for index, line in enumerate(lines):
        match = re.search(r"\*\*\*\s+(NOTE|WARNING|ERROR|FATAL)\s+\*\*\*", line, re.I)
        if not match:
            continue
        body = []
        for following in lines[index + 1:index + 13]:
            if not following.strip() or re.search(r"\*\*\*\s+(?:NOTE|WARNING|ERROR|FATAL)", following, re.I):
                break
            body.append(following.strip())
        messages.append({"severity": match[1].upper(), "message": portable_text("\n".join(body))})
    return messages


def classify_messages(messages):
    """Separate the expected unloaded no-D warning without hiding diagnostics."""
    counts = Counter()
    for item in messages:
        body = " ".join(item["message"].split())
        expected = (item["stage"] == "check_unloaded" and item["severity"] == "WARNING"
                    and body == "No constraints have been defined using the D command.")
        native_parallel_warning = re.fullmatch(
            r"Quadrilateral element \d+ has a pair of opposite edges that are "
            r"[0-9.]+ degrees away from being parallel\. This exceeds the warning limit of "
            r"[0-9.]+ degrees\.", body, re.I)
        shape = bool(native_parallel_warning or re.search(
            r"shape testing|shape warning|shape error|aspect ratio|parallel deviation|maximum angle|jacobian ratio|warping factor|poorly shaped",
            body, re.I))
        item["expected_unloaded_no_constraints_warning"] = expected
        item["shape_diagnostic"] = shape
        if expected:
            counts["expected_unloaded_no_constraints_warning_count"] += 1
        elif item["severity"] == "WARNING":
            counts["unexpected_warning_count"] += 1
            counts["unexpected_warning_message_count"] += 1
            if not shape:
                counts["nonshape_unexpected_warning_count"] += 1
        elif item["severity"] in ("ERROR", "FATAL"):
            counts["error_or_fatal_count"] += 1
    return counts


def shape_summary(text):
    tests = {}
    for name in SHAPE_NAMES:
        matches = re.findall(r"^\s*" + re.escape(name) + r"\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s*%", str(text), re.M)
        if len(matches) != 1:
            raise ValueError(f"Native shape summary missing or ambiguous: {name}")
        tested, warnings, errors, percentage = matches[0]
        tests[name] = {"tested": int(tested), "warnings": int(warnings),
                       "errors": int(errors), "warning_or_error_percent": float(percentage)}
    return tests


def rebuild_commands(data, config):
    """Use native node/element IDs and original connectivity, changing no topology."""
    native = data["native_settings"]
    if (native["element_name"] != 181 or native["offset"] != "MID"
            or native["keyopts"] != {str(k): v for k, v in config["keyopts"].items()}
            or native["thickness_mm"] != config["thickness"]
            or native["layer_thickness_mm"] != config["thickness"]
            or native["layer_material"] != 1 or native["layer_angle_degrees"] != 0
            or native["through_thickness_points"] != config["thickness_integration_points"]):
        raise ValueError("Input element/section settings differ from the preserved P1 model")
    lines = ["FINISH", "/CLEAR,NOSTART", "/PREP7", "/GOPR", "SHPP,ON",
             "ET,1,SHELL181", "ETCONTROL,OFF"]
    lines += [f"KEYOPT,1,{key},{value}" for key, value in config["keyopts"].items()]
    lines += [f"MP,EX,1,{config['young']:.17g}", f"MP,PRXY,1,{config['poisson']:.17g}",
              "SECTYPE,1,SHELL", f"SECDATA,{config['thickness']:.17g},1,0,{config['thickness_integration_points']}",
              "SECOFFSET,MID"]
    lines += [f"N,{int(node)}," + ",".join(format(float(value), ".17g") for value in xyz)
              for node, xyz in zip(data["node_ids"], data["xyz_mm"])]
    previous = None
    for index, (element, nodes) in enumerate(zip(data["element_ids"], data["connectivity_node_ids"])):
        current = tuple(int(data["attributes"][name][index]) for name in ("TYPE", "MAT", "SECNUM"))
        if current != previous:
            lines += [f"TYPE,{current[0]}", f"MAT,{current[1]}", f"SECNUM,{current[2]}"]
            previous = current
        lines.append(f"EN,{int(element)}," + ",".join(str(int(node)) for node in nodes))
    return lines + ["ALLSEL,ALL"]


def compare_mesh(expected, actual, tolerance=1e-8):
    """Compare by actual IDs; ordering returned by MAPDL is immaterial."""
    nodes_equal = set(expected["node_ids"]) == set(actual["node_ids"])
    elements_equal = set(expected["element_ids"]) == set(actual["element_ids"])
    before_nodes = dict(zip(expected["node_ids"], expected["xyz_mm"]))
    after_nodes = dict(zip(actual["node_ids"], actual["xyz_mm"]))
    delta = max((float(np.max(np.abs(np.array(before_nodes[node]) - after_nodes[node])))
                 for node in before_nodes if node in after_nodes), default=0.)
    before_elements = {element: index for index, element in enumerate(expected["element_ids"])}
    after_elements = {element: index for index, element in enumerate(actual["element_ids"])}
    conn_equal = elements_equal
    attributes_equal = elements_equal and set(expected["attributes"]) == set(actual["attributes"])
    for element in before_elements.keys() & after_elements.keys():
        old, new = before_elements[element], after_elements[element]
        conn_equal &= expected["connectivity_node_ids"][old] == actual["connectivity_node_ids"][new]
        for name in expected["attributes"].keys() & actual["attributes"].keys():
            attributes_equal &= expected["attributes"][name][old] == actual["attributes"][name][new]
    result = {"node_ids_unchanged": nodes_equal, "element_ids_unchanged": elements_equal,
              "connectivity_unchanged": bool(conn_equal), "attributes_unchanged": bool(attributes_equal),
              "native_settings_unchanged": expected["native_settings"] == actual["native_settings"],
              "maximum_coordinate_difference_mm": delta, "coordinate_tolerance_mm": tolerance,
              "coordinates_within_tolerance": nodes_equal and delta <= tolerance}
    result["passed"] = all(value for key, value in result.items()
                           if key not in ("maximum_coordinate_difference_mm", "coordinate_tolerance_mm"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--exec-file", required=True)
    args = parser.parse_args()
    total_start = time.perf_counter()
    verify_manifest(args.source_gate)
    execution = read_json(args.source_gate / "execution_manifest.json")
    config = execution["config"]
    if config["quality_limits"] != LIMITS or config["coordinate_tolerance_mm"] != 1e-8:
        raise ValueError("The preserved geometry acceptance thresholds have changed")
    source_settings = read_json(args.source_gate / "base_mesh.json")["native_settings"]
    if (args.input / "artifact_manifest.json").exists():
        verify_manifest(args.input)
    mesh_paths = sorted(args.input.glob("*_mesh.json"))
    if not mesh_paths:
        raise ValueError("No saved candidate meshes found")
    candidates = []
    for path in mesh_paths:
        name = path.name[:-len("_mesh.json")]
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError("Unsupported case name")
        metadata_path = args.input / f"{name}.json"
        data, metadata = read_json(path), read_json(metadata_path)
        if not (args.source_gate / f"{name}.json").is_file():
            raise ValueError(f"No preserved baseline case: {name}")
        q = quality(data, config)
        skip_reasons = []
        if metadata.get("quality", {}).get("passed") is not True:
            skip_reasons.append("published_offline_quality_not_passed")
        if not q["passed"]:
            skip_reasons.append("independent_geometry_quality_not_passed")
        if data["native_settings"] != source_settings:
            skip_reasons.append("native_settings_differ_from_preserved_model")
        candidates.append((name, path, metadata_path, data, q, skip_reasons))
    args.output.mkdir(parents=True, exist_ok=False)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    runtime = args.runtime_root.resolve() / run_id
    runtime.mkdir(parents=True, exist_ok=False)
    records = []
    from ansys.mapdl.core import launch_mapdl
    for name, mesh_path, metadata_path, expected, offline_quality, skip_reasons in candidates:
        print(f"START native coordinate check: {name}", flush=True)
        case_start = time.perf_counter()
        work = runtime / name
        work.mkdir()
        source_record = read_json(args.source_gate / f"{name}.json")
        record = {"case": name, "passed": False, "solve_performed": False,
                  "indicator_computed": False, "training_performed": False,
                  "refinement_performed": False, "offline_quality": offline_quality,
                  "input_mesh_sha256": digest(mesh_path), "input_gate_sha256": digest(metadata_path),
                  "source_gate_record_sha256": digest(args.source_gate / f"{name}.json"),
                  "source_native_parallel_deviation_warnings": source_record.get("native_parallel_deviation_warnings"),
                  "source_native_parallel_deviation_errors": source_record.get("native_parallel_deviation_errors"),
                  "wall_seconds": {}, "native_messages": [], "native_launch_attempted": False,
                  "native_run_performed": False, "native_check_performed": False}
        source_response = args.source_gate / f"{name}_native_response.txt"
        if source_response.is_file():
            record["source_native_shape_tests"] = shape_summary(source_response.read_text(encoding="utf-8"))
        if skip_reasons:
            record["status"] = "skipped_offline_gate_failed"
            record["skip_reasons"] = skip_reasons
            record["wall_seconds"]["case_total"] = time.perf_counter() - case_start
            save_json(args.output / f"{name}.json", record)
            records.append(record)
            print(f"SKIP native coordinate check: {name}; offline gate failed", flush=True)
            continue
        session = None
        try:
            start = time.perf_counter()
            record["native_launch_attempted"] = True
            session = launch_mapdl(exec_file=args.exec_file, run_location=str(work), nproc=2,
                                   additional_switches="-smp", start_timeout=120, start_instance=True,
                                   mode="grpc", remove_temp_dir_on_exit=False,
                                   log_apdl=str(work / "session.apdl"))
            record["native_run_performed"] = True
            record["wall_seconds"]["launch"] = time.perf_counter() - start
            record["native_version"] = str(session.version)
            start = time.perf_counter()
            build_response = input_commands(session, work, "rebuild_coordinates", rebuild_commands(expected, config))
            record["wall_seconds"]["build"] = time.perf_counter() - start
            record["native_messages"] += [{"stage": "build", **item} for item in native_messages(build_response)]
            start = time.perf_counter()
            check_response = input_commands(session, work, "check_full", ["ALLSEL,ALL", "SHPP,ON", "CHECK"])
            summary_response = str(session.shpp("SUMMARY"))
            status_response = str(session.shpp("STATUS"))
            (work / "shape_summary.txt").write_text(summary_response, encoding="utf-8")
            (work / "shape_status.txt").write_text(status_response, encoding="utf-8")
            record["wall_seconds"]["native_check"] = time.perf_counter() - start
            record["native_shape_tests_before_loading"] = shape_summary(summary_response)
            record["native_messages"] += [{"stage": "check_unloaded", **item} for item in native_messages(check_response + "\n" + summary_response + "\n" + status_response)]
            record["shape_check_control"] = {"enabled_before_element_creation": True,
                                              "all_tests_enabled_before_full_check": True,
                                              "full_check_all_selected": True,
                                              "limits_modified": False,
                                              "status_sha256": digest(work / "shape_status.txt")}
            start = time.perf_counter()
            actual, native_quality = snapshot(session, work, config)
            record["native_quality"] = native_quality
            record["round_trip"] = compare_mesh(expected, actual)
            unloaded, unloaded_path = cdb_commands(session, work, "unloaded")
            record["unloaded_before_boundary_conditions"] = not unloaded
            record["unloaded_other_records"] = unloaded
            record["wall_seconds"]["export_and_geometry_validation"] = time.perf_counter() - start
            save_json(args.output / f"{name}_mesh.json", actual)
            start = time.perf_counter()
            load_response = input_commands(session, work, "reapply_p1", ["ALLSEL,ALL"] +
                [f"D,{int(node)},{dof},0" for node in boundary_ids(actual, config)
                 for dof in ("UX", "UY", "UZ", "ROTX", "ROTY", "ROTZ")] +
                [f"SFE,ALL,1,PRES,1,{config['pressure']:.17g}", "ALLSEL,ALL"])
            record["native_messages"] += [{"stage": "loads", **item} for item in native_messages(load_response)]
            loaded, loaded_path = cdb_commands(session, work, "after_loading")
            audit = load_check(loaded, actual, config)
            evidence = compact_loads(loaded)
            if load_check(expand_loads(evidence), actual, config) != audit:
                raise ValueError("Compact native load evidence failed round trip")
            record["boundary_and_pressure"] = audit
            save_json(args.output / f"{name}_loads.json", evidence)
            session.save(str(work / "mesh_with_p1"), "db")
            record["wall_seconds"]["load_validation"] = time.perf_counter() - start
            start = time.perf_counter()
            loaded_check_response = input_commands(session, work, "check_loaded", ["ALLSEL,ALL", "SHPP,ON", "CHECK"])
            loaded_summary_response = str(session.shpp("SUMMARY"))
            (work / "shape_summary_loaded.txt").write_text(loaded_summary_response, encoding="utf-8")
            record["wall_seconds"]["loaded_native_check"] = time.perf_counter() - start
            record["native_shape_tests_after_loading"] = shape_summary(loaded_summary_response)
            record["native_shape_tests"] = record["native_shape_tests_after_loading"]
            record["native_check_performed"] = True
            record["native_messages"] += [{"stage": "check_loaded", **item} for item in native_messages(loaded_check_response + "\n" + loaded_summary_response)]
            counts = Counter(item["severity"] for item in record["native_messages"])
            record["native_message_counts"] = dict(counts)
            classified = classify_messages(record["native_messages"])
            record["expected_unloaded_no_constraints_warning_count"] = classified["expected_unloaded_no_constraints_warning_count"]
            record["native_unexpected_warning_count"] = classified["unexpected_warning_count"]
            record["native_unexpected_warning_message_count"] = classified["unexpected_warning_message_count"]
            record["native_nonshape_unexpected_warning_count"] = classified["nonshape_unexpected_warning_count"]
            record["native_error_or_fatal_count"] = classified["error_or_fatal_count"]
            all_tested = all(row["tested"] == len(actual["element_ids"])
                             for row in record["native_shape_tests"].values())
            record["all_native_elements_shape_tested"] = all_tested
            no_shape_issues = all(row["warnings"] == row["errors"] == 0 for row in record["native_shape_tests"].values())
            record["native_shape_warning_free"] = no_shape_issues
            record["passed"] = bool(native_quality["passed"] and record["round_trip"]["passed"]
                and not unloaded and audit["passed"] and all_tested and no_shape_issues
                and not classified["unexpected_warning_count"] and not classified["error_or_fatal_count"])
        except Exception as error:
            record["case_error"] = {"type": type(error).__name__, "message": portable_text(str(error))}
            (work / "case_exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            if session is not None:
                start = time.perf_counter()
                try:
                    session.exit()
                except Exception as error:
                    record["passed"] = False
                    record["exit_error"] = {"type": type(error).__name__, "message": portable_text(str(error))}
                record["wall_seconds"]["session_exit"] = time.perf_counter() - start
        record["wall_seconds"]["case_total"] = time.perf_counter() - case_start
        record["status"] = "passed" if record["passed"] else "native_check_failed"
        record["private_artifact_sha256"] = {path.name: digest(path) for path in work.iterdir()
                                              if path.is_file() and path.suffix.lower() in (".apdl", ".cdb", ".txt", ".log")}
        save_json(args.output / f"{name}.json", record)
        records.append(record)
        print(f"DONE native coordinate check: {name}; passed={record['passed']}", flush=True)
    forbidden = []
    for path in runtime.rglob("session.apdl"):
        forbidden += [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
                      if FORBIDDEN.match(line)]
    solution_files = [path.name for path in runtime.rglob("*") if path.is_file()
                      and path.suffix.lower() in (".rst", ".rth", ".full", ".mntr")]
    summary = {"case_count": len(records), "all_cases_passed": all(row["passed"] for row in records),
               "solve_performed": bool(forbidden or solution_files), "solve_commands": forbidden,
               "solution_files": solution_files, "indicator_computed": False,
               "training_performed": False, "refinement_performed": False,
               "total_wall_seconds": time.perf_counter() - total_start, "cases": records}
    summary["all_three_requested_cases_present"] = {row["case"] for row in records} == EXPECTED_CASES
    summary["eligible_for_next_mesh_stage"] = (summary["all_cases_passed"]
        and summary["all_three_requested_cases_present"] and not summary["solve_performed"])
    save_json(args.output / "summary.json", summary)
    save_json(args.output / "manifest.json", {"run_id_utc": run_id, "config": config,
        "source_gate_artifact_manifest_sha256": digest(args.source_gate / "artifact_manifest.json"),
        "source_gate_execution_manifest_sha256": digest(args.source_gate / "execution_manifest.json"),
        "packages": {name: importlib.metadata.version(name) for name in ("ansys-mapdl-core", "numpy")},
        "source_sha256": {path.name: digest(path) for path in Path(__file__).parent.glob("*.py")},
        "runtime_policy": "Fresh private runtime; native binary models, CDB and machine logs are not published",
        "scope": "Coordinate round trip, unchanged topology/settings, native shape checks and P1 load checks only"})
    save_json(args.output / "artifact_manifest.json", {"files": {path.relative_to(args.output).as_posix(): digest(path)
                                                               for path in args.output.rglob("*") if path.is_file()}})
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))
    return 0 if summary["eligible_for_next_mesh_stage"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
