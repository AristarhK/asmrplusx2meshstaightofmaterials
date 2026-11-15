# ASMR++ vs ZZ-AMR — benchmark notes
Number of paired PDE tasks: 10.
Adaptive steps: 4.
Policy: deterministic, saved ASMR++ PPO after 10 training iterations.
ZZ scoring: built-in ZienkiewiczZhuErrorHeuristic recovered-gradient estimator.
Budget: ZZ element count at every step <= ASMR++ count at that step.
Same sampled PDE, initial mesh, FEM solve and normalized reference error for each pair.
Reference metric is not necessarily exact analytical error.
Timing is exploratory: ZZ budget matching performs additional trial refinements;
therefore do not infer optimized solver wall-time or speedup from this script.

## Final-step paired comparison
PPO mean N: 230.10
ZZ mean N: 224.10
PPO mean error: 0.485574
ZZ mean error: 0.681299
PPO error smaller in 7/10 pairs.
PPO error gain (mean of paired percentages): 20.90%

Interpreting N-efficiency requires care because matched counts may not be exactly equal.
For a paper, run repeated seeds and compare at truly matched N or interpolated error(N).
