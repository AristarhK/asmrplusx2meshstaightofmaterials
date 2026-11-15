# Neural Mesh Adaptation Research

## Project goal
Research neural network methods for mesh generation
and adaptation in computational physics.

Long-term goal: develop a hybrid mesh adaptation
system for plasma physics and multiphysics simulations.

## Computing environments

Windows:
- ANSYS Mechanical
- ANSYS ACP
- Engineering FEM simulations

Ubuntu:
- ASMR++
- AMBER
- G-Adaptivity
- PlasmaMesh
- Firedrake / PETSc

GitHub is the central research repository.

## ASMR++ Research

Location: experiments/asmrpp/

Completed experiments:

ASMR-01: PPO training, 1 iteration.
ASMR-02: PPO training, 10 iterations.
ASMR-03: Penalty coefficient analysis.
ASMR-04: Pareto analysis and mesh visualization.
ASMR-05: Comparison against classical ZZ-AMR.

Final pilot results on 10 varying Poisson problems:

ASMR++ mean normalized error: 0.48557
ZZ-AMR mean normalized error: 0.68129

ASMR++ achieved lower error in 7/10 cases.

Mean mesh sizes:
ASMR++: 230.1
ZZ-AMR: 224.1

These are preliminary results.
Further multi-seed and matched-budget tests are needed.

## Important instructions for Codex

1. Read available experimental files before making claims.
2. Preserve original CSV files and experimental data.
3. Do not retrain models without authorization.
4. Do not overwrite completed experimental results.
5. Keep reproducibility and scientific rigor.
6. Document mathematical assumptions.
7. Record new experiments in docs/.
8. Distinguish measured results from hypotheses.
9. Avoid installing heavy scientific dependencies
   without explicit authorization.
10. Note that the full upstream ASMR++ source code
    and trained checkpoints may not be present here.

## Next research direction

G-Adaptivity: graph neural network based mesh relocation.

Compare:
- h-adaptivity
- r-adaptivity
- classical FEM adaptation
- hybrid neural/classical approaches
