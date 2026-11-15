#!/usr/bin/env python3
"""One deterministic, four-refinement Poisson episode using recovered weights."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
from time import perf_counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--environment-index", type=int, required=True)
    parser.add_argument("--penalty", type=float, required=True)
    parser.add_argument("--threads", type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    import numpy as np
    import torch
    import yaml
    from src.algorithms import create_algorithm

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)
    config_path = args.source_root / "reports/poisson_asmr_train10/log/rep_00/config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["algorithm"]["checkpoint"] = {
        "experiment_name": "poisson_asmr_train10", "repetition": 0,
        "load_root_dir": str(args.source_root / "reports"),
    }
    config["algorithm"]["use_gpu"] = False
    config["recording"]["wandb"]["enabled"] = False
    algorithm = create_algorithm(config=config, seed=args.seed)
    algorithm.set_training_mode(False)
    if not 0 <= args.environment_index < len(algorithm.evaluation_environments):
        raise ValueError("No such evaluation environment")
    env = algorithm.evaluation_environments[args.environment_index]
    env.reset()
    env.set_element_penalty_lambda(args.penalty, from_position=False)

    def policy_digest():
        digest = hashlib.sha256()
        for name, tensor in sorted(algorithm.policy.state_dict().items()):
            array = tensor.detach().cpu().numpy()
            digest.update(name.encode())
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    before = policy_digest()
    states, meshes = [], []
    elapsed = 0.0

    def record(step, decision=0.0, fem=0.0):
        mesh = env.fem_problem.mesh
        points, cells = np.asarray(mesh.p).copy(), np.asarray(mesh.t).copy()
        mesh_digest = hashlib.sha256(points.tobytes() + cells.tobytes()).hexdigest()
        row = {
            "step": step, "elements": int(env.num_agents), "vertices": points.shape[1],
            "legacy_normalized_reference_error": float(np.asarray(env._get_remaining_error()).reshape(-1)[0]),
            "decision_s": decision, "fem_step_s": fem, "cumulative_s": elapsed,
            "mesh_sha256": mesh_digest,
        }
        if not np.isfinite(row["legacy_normalized_reference_error"]):
            raise ValueError("Non-finite error metric")
        states.append(row)
        meshes.append((points, cells))
        np.savez_compressed(args.output / f"mesh_step_{step}.npz", points=points, cells=cells,
                            scalar_solution=np.asarray(env.scalar_solution))
        print(f"step={step} N={row['elements']} error={row['legacy_normalized_reference_error']:.12f}", flush=True)

    record(0)
    done = False
    for step in range(1, int(env._max_timesteps) + 1):
        if done:
            raise RuntimeError("Episode ended earlier than the saved horizon")
        start = perf_counter()
        actions, _ = algorithm.policy_step(observation=env.last_observation)
        actions = actions.detach().cpu().numpy().reshape(-1, 1)
        decision = perf_counter() - start
        start = perf_counter()
        _, _, done, _ = env.step(action=actions)
        fem = perf_counter() - start
        elapsed += decision + fem
        record(step, decision, fem)
    if not done:
        raise RuntimeError("Episode did not finish at the saved horizon")
    after = policy_digest()
    if before != after or algorithm.policy.training:
        raise RuntimeError("Policy changed or remained in training mode")
    with (args.output / "trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(states[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(states)

    expected = json.loads((Path(__file__).parent / "artifacts/expected_case0.json").read_text())
    comparable = (args.seed == expected["seed"] and args.environment_index == expected["environment_index"]
                  and args.penalty == expected["penalty"])
    mismatches = []
    if comparable:
        if len(states) != len(expected["states"]):
            mismatches.append("different trajectory length")
        for actual, target in zip(states, expected["states"]):
            for key in ("step", "elements", "vertices"):
                if actual[key] != target[key]:
                    mismatches.append(f"step {target['step']}: {key}")
            if not np.isclose(actual["legacy_normalized_reference_error"], target["error"], rtol=1e-9, atol=1e-10):
                mismatches.append(f"step {target['step']}: error")
    versions = {}
    for name in ("numpy", "torch", "torch-geometric", "scikit-fem", "scipy", "meshpy", "PyYAML"):
        versions[name] = importlib.metadata.version(name)
    report = {
        "status": "passed" if not mismatches else "legacy_mismatch",
        "python": platform.python_version(), "versions": versions,
        "seed": args.seed, "environment_index": args.environment_index,
        "penalty": args.penalty, "threads": args.threads, "training_performed": False,
        "policy_sha256_before": before, "policy_sha256_after": after,
        "policy_unchanged": before == after,
        "legacy_case0_matches": not mismatches if comparable else None,
        "legacy_comparison_scope": "elements, vertices and legacy error at all five states; timing excluded",
        "legacy_mismatches": mismatches, "states": states,
        "limitation": "Single-case saved-policy reproduction; no new accuracy, speedup or elasticity benchmark.",
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    for ax, index, color, title in [(axes[0], 0, "#64748b", "Initial mesh"),
                                    (axes[1], -1, "#2563eb", "Saved policy: final mesh")]:
        points, cells = meshes[index]
        ax.triplot(Triangulation(points[0], points[1], cells.T), color=color, linewidth=0.65)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nN = {states[index]['elements']}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    axes[2].plot([s["elements"] for s in states],
                 [s["legacy_normalized_reference_error"] for s in states], "o-", color="#2563eb")
    axes[2].set(xlabel="Number of elements", ylabel="Legacy normalized reference error",
                title="Reproduced trajectory")
    axes[2].grid(alpha=0.25)
    fig.suptitle(f"ASMR++ / Poisson / seed {args.seed} / saved weights / no training")
    fig.savefig(args.output / "mesh_reproduction.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"status": report["status"], "legacy_case0_matches": report["legacy_case0_matches"],
                      "policy_unchanged": report["policy_unchanged"]}), flush=True)
    return 0 if not mismatches else 2


if __name__ == "__main__":
    raise SystemExit(main())
