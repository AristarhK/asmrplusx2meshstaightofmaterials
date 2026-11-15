#!/usr/bin/env python3
"""Verify the recovered ASMR++ artifacts and run one saved-policy episode.

Uses the current Python environment; never installs dependencies or trains.
All runtime files are written beneath a new, explicitly selected output folder.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        path = (ROOT / artifact["path"]).resolve()
        if ROOT not in path.parents:
            raise ValueError("Artifact path escapes the recovery package")
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != artifact["size_bytes"]:
            raise ValueError(f"Size mismatch: {artifact['path']}")
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {artifact['path']}")
    return manifest


def extract_source(archive, destination, expected_files):
    destination.mkdir()
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if len(members) != len(expected_files):
            raise ValueError("Unexpected number of source archive members")
        seen = set()
        for member in members:
            path = (destination / member.name).resolve()
            if destination not in path.parents or not member.isfile():
                raise ValueError(f"Unsafe source archive member: {member.name}")
            if member.name not in expected_files or member.name in seen:
                raise ValueError(f"Unexpected source file: {member.name}")
            seen.add(member.name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as incoming, path.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            path.chmod(member.mode & 0o777)
            if sha256(path) != expected_files[member.name]:
                raise ValueError(f"Source hash mismatch: {member.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path, help="New directory; existing paths are refused")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--environment-index", type=int, default=1)
    parser.add_argument("--penalty", type=float, default=0.001274)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    manifest = verify_artifacts()
    print(f"Verified {len(manifest['artifacts'])} artifacts; upstream {manifest['provenance']['upstream_commit']}", flush=True)
    if args.verify_only:
        return 0
    if args.output is None:
        parser.error("--output is required for inference")
    if args.threads < 1 or args.environment_index < 0:
        parser.error("threads must be positive and environment-index nonnegative")
    if sys.version_info[:2] != (3, 10):
        parser.error("Use the recovered ASMRpp_cpu Python 3.10 environment; verification alone works without it")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = output / "runtime"
    extract_source(ROOT / "artifacts/asmrplusplus_source.tar.gz", runtime, manifest["provenance"]["source_file_sha256"])
    checkpoint_dir = runtime / "reports/poisson_asmr_train10/log/rep_00/checkpoints"
    checkpoint_dir.mkdir(parents=True)
    for name in ("ckpt_final.pt", "ckpt_final_norm_params.pkl", "network_kwargs.npz"):
        shutil.copyfile(ROOT / "artifacts/checkpoints" / name, checkpoint_dir / name)
    shutil.copyfile(ROOT / "artifacts/config.yaml", checkpoint_dir.parent / "config.yaml")
    invocation = {
        "upstream_commit": manifest["provenance"]["upstream_commit"],
        "checkpoint_sha256": sha256(checkpoint_dir / "ckpt_final.pt"),
        "seed": args.seed, "environment_index": args.environment_index,
        "penalty": args.penalty, "threads": args.threads,
        "python_executable": sys.executable, "training_requested": False,
    }
    (output / "invocation.json").write_text(json.dumps(invocation, indent=2) + "\n")
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads),
        "OPENBLAS_NUM_THREADS": str(args.threads), "PYTHONHASHSEED": str(args.seed),
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(runtime),
        "NUMBA_CACHE_DIR": str(output / "cache/numba"),
        "MPLCONFIGDIR": str(output / "cache/matplotlib"), "MPLBACKEND": "Agg",
        "WANDB_MODE": "disabled",
    })
    command = [sys.executable, "-B", str(ROOT / "inference_worker.py"),
               "--source-root", str(runtime), "--output", str(output),
               "--seed", str(args.seed), "--environment-index", str(args.environment_index),
               "--penalty", str(args.penalty), "--threads", str(args.threads)]
    with (output / "inference.log").open("w") as log:
        process = subprocess.Popen(command, cwd=runtime, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        code = process.wait()
    # The pristine recovery package and its model remain unchanged by inference.
    verify_artifacts()
    if code:
        print(f"Inference failed ({code}); diagnostics: {output / 'inference.log'}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
