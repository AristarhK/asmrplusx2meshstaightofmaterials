"""Verify published panel SHA-256 with stdlib only; optionally check Git index bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory",type=Path)
    parser.add_argument("--git-index",action="store_true")
    args=parser.parse_args()
    root=args.run_directory.resolve()
    manifest=json.loads((root/"artifact_manifest.json").read_text(encoding="utf-8"))
    for name,expected in manifest["files"].items():
        path=(root/name.replace("\\","/")).resolve()
        if root not in path.parents:
            raise ValueError("Artifact path escapes run directory")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Artifact SHA-256 mismatch: {name}")
    if args.git_index:
        repository=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip()).resolve()
        paths=list(root.rglob("*"))
        for path in paths:
            if path.is_file():
                relative=path.relative_to(repository).as_posix()
                blob=subprocess.check_output(["git","cat-file","blob",":"+relative])
                if blob!=path.read_bytes():
                    raise ValueError(f"Git index changed artifact bytes: {relative}")
    print(f"Verified {len(manifest['files'])} artifact hashes" + (" and Git index bytes" if args.git_index else ""))


if __name__=="__main__":
    main()
