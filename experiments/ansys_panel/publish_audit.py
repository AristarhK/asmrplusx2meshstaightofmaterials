"""Create a NEW machine-independent publication of existing panel audit reports.

Stdlib only; never imports a solver, changes source reports or reruns a check.
Absolute paths, local source snapshots and file mtimes stay in private reports.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


LOCAL_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\[^\\\s]+[\\/]|/(?:Users|home)/")
PRIVATE_KEYS = {"runtime_directory", "mtime_ns", "source_files_before", "source_files_after", "python"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in PRIVATE_KEYS:
                continue
            if key == "input_directory":
                result["source_run"] = "20261116_panel03"
                continue
            if LOCAL_PATH.search(key):
                raise ValueError("Unexpected machine-specific dictionary key")
            result[key] = clean(item)
        return result
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, str) and LOCAL_PATH.search(value):
        return "<local path omitted; see private source report>"
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-audits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {"review": "20261116_review01", "rst": "20261116_rst01", "rst_diagnostics": "20261116_rst_diagnostics01"}
    snapshots = {f"{logical}/{p.name}": sha(p) for logical, directory in sources.items() for p in (args.input_audits/directory).iterdir() if p.is_file()}
    args.output.mkdir(parents=True, exist_ok=False)
    for logical, directory in sources.items():
        destination = args.output/logical
        destination.mkdir()
        for source in sorted((args.input_audits/directory).iterdir()):
            if not source.is_file() or source.name == "manifest.json":
                continue
            if source.suffix not in (".json", ".csv", ".md"):
                raise ValueError("Unexpected noncompact audit file")
            target = destination/source.name
            if source.suffix == ".json":
                data = json.loads(source.read_text(encoding="utf-8"))
                if logical == "rst" and data.get("passed") is not False:
                    raise ValueError("Expected explicitly failed historical RST audit")
                sanitized = clean(data)
                text = json.dumps(sanitized, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
                if LOCAL_PATH.search(text):
                    raise ValueError("Machine-specific content remains")
                with target.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(text)
            else:
                if LOCAL_PATH.search(source.read_text(encoding="utf-8")):
                    raise ValueError("Machine-specific content in table")
                shutil.copyfile(source, target)
    after = {f"{logical}/{p.name}": sha(p) for logical, directory in sources.items() for p in (args.input_audits/directory).iterdir() if p.is_file()}
    if snapshots != after:
        raise ValueError("Source audit reports changed during publication")
    manifest = {
        "published_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Sanitized copies only; no numerical recomputation, solver or RST-reader execution",
        "source_reports_unchanged": True,
        "rst_reading_status": "FAILED; retained explicitly, no reader repair performed",
        "removed_metadata": sorted(PRIVATE_KEYS | {"input_directory"}),
        "policy": "Retain numerical evidence, logical IDs, hashes and package versions; omit local paths, source snapshot dictionaries and mtimes. Source and public hashes differ intentionally.",
        "raw_report_sha256_by_logical_id": snapshots,
        "publisher_sha256": sha(Path(__file__)),
        "files": {p.relative_to(args.output).as_posix(): sha(p) for p in sorted(args.output.rglob("*")) if p.is_file()},
    }
    with (args.output/"publication_manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Published {len(manifest['files'])} compact files; private sources unchanged; RST status FAILED")


if __name__ == "__main__":
    main()
