# Transformer tank mesh adaptation in ANSYS

## Current priority

Develop neural adaptation of a finite-element mesh for the mechanical strength
analysis of a transformer tank in ANSYS. Application: the 2026 «Система» contest,
ERSO topic «Новейшие методы расчета баков на механическую прочность».

The nearest practical milestone is Python running the same ANSYS mechanical
model twice with different meshes and saving a comparable result table.
First recover the user's Windows automation and define the mechanical model.
The saved Poisson policy is research infrastructure; its applicability to
elasticity, transformer tanks or ANSYS has NOT been demonstrated.

## Start each task

1. Read README.md, docs/WINDOWS_HANDOFF.md, docs/ERSO_ANSYS_SCOPE.md and
   docs/ASMR_REPRODUCIBILITY.md. Consult the research log and roadmap for methodology.
2. Inspect actual cwd, Git status, branches, commits and available artifacts.
   Do not assume a path named research-hub, a previous SHA or branch is current.
3. Use the existing checkout. Do not create a worktree unless the user asks.
4. Check the saved recovery package without scientific dependencies:
   `python3 -B experiments/asmrpp/reproduce/reproduce.py --verify-only`.
   On Windows, use a confirmed Python interpreter; `py -3 -B` is one option.
5. Distinguish checksum verification, saved historical reproduction, a fresh
   inference, and an actual ANSYS solve. Report only what was executed.

## Preserved ASMR++ research

Location: experiments/asmrpp/.

- ASMR-01: PPO training, 1 iteration.
- ASMR-02: PPO training, 10 iterations.
- ASMR-03: penalty coefficient analysis.
- ASMR-04: Pareto analysis and mesh visualization.
- ASMR-05: comparison against classical ZZ-AMR.

Historical pilot on 10 Poisson tasks: mean normalized reference error
ASMR++ 0.4855736514 / ZZ 0.6812989644; lower ASMR++ error in 7/10 pairs.
Mean element counts: 230.1 / 224.1. The ZZ counts are capped by ASMR++ counts,
not exactly matched. Multi-seed, matched-DoF tests and reference audit remain open.

The recovery package includes the source snapshot, patch, resolved config,
checkpoints, normalizers, architecture, environment metadata and SHA-256 manifest.
Saved reports document reproduction of ONE historical case in two Linux processes.
Read the actual manifest/reports before making claims. A clean environment and
native Windows inference have not been validated. Do not recreate training to
compensate for an unavailable runtime.

## Working rules

1. Preserve historical CSV, figures, configs, scripts, model artifacts and completed
   runs. New outputs need a new directory; never overwrite completed results.
2. Do not retrain models or install heavy scientific dependencies without explicit
   authorization. Do not run an inference merely to check artifact integrity.
3. Never guess the old Windows script path, ANSYS API, port or installed version.
   Search the actual local files using docs/WINDOWS_HANDOFF.md. If absent,
   restoring the connection is a separate next task.
4. Keep geometry, units, materials, loads, boundary conditions and solver settings
   fixed between mesh comparisons. Record actual DoF, quality and failures.
5. Separate observations from hypotheses; document mathematical assumptions,
   error definitions, provenance and limitations in docs/.
6. Keep .gitattributes protection for experiments/asmrpp/: Git line-ending conversion
   would invalidate manifest hashes on Windows. Do not fix hashes to hide a mismatch.
7. Stage explicit files. Preserve user changes. Inspect staged paths and file sizes
   before committing. No force-push or PR merge without explicit instructions.
8. Do not commit runtime/cache directories, environments, credentials or unrelated
   personal files. Use ignored .local/ for local ANSYS models and scratch runs;
   publish selected authorized results with provenance in a separate run directory.
9. Record the final commit, synchronization state, commands checked and remaining
   blockers. Prepared instructions are not proof of publication or an ANSYS run.

## Environments and secondary directions

Windows: actual ANSYS Mechanical installation, VS Code/Python automation and
mechanical simulations. ACP is relevant only if the model requires composites.
Ubuntu: preserved ASMR++ runtime and future adaptation research.
Codex cloud: repository work and lightweight verification; no access to Windows
files or ANSYS unless explicitly connected.

AMBER, G-Adaptivity (graph-based r-adaptivity), PlasmaMesh, Firedrake/PETSc and
hybrid h/r/classical adaptation remain research directions. Preserve their context
in docs/MESH_RESEARCH_ROADMAP.md; they do not precede the transformer-tank milestone.
GitHub is the shared source of code, artifacts, instructions and research records.
