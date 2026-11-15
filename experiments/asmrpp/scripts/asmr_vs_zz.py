#!/usr/bin/env python3
"""ASMR++ vs classical Zienkiewicz-Zhu adaptive refinement.

Run from ASMRplusplus repository root with conda env ASMRpp_cpu:
    python asmr_vs_zz.py --cases 10

Same sampled Poisson PDE, initial mesh, FEM solver and reference error metric
for both methods on every case. ZZ chooses top recovered-gradient estimator
scores and does not exceed the ASMR++ element count at the same step.

Note: matching budgets by trial refinements is not an optimized ZZ implementation;
wall-time results are exploratory only.
"""
import argparse
import copy
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.algorithms import create_algorithm
from modules.swarm_environments.mesh_refinement.remeshing_heuristics import (
    ZienkiewiczZhuErrorHeuristic,
)
from modules.swarm_environments.util.torch_util import detach


def numeric_error(env):
    return float(np.asarray(env._get_remaining_error()).reshape(-1)[0])


def record(case, method, step, env, t_decision, t_step, t_cum):
    return dict(
        case=case,
        method=method,
        step=step,
        elements=int(env.num_agents),
        vertices=int(env.fem_problem.mesh.p.shape[1]),
        error=numeric_error(env),
        decision_s=float(t_decision),
        fem_step_s=float(t_step),
        cumulative_s=float(t_cum),
    )


def zz_actions_at_budget(env, target_elements, estimator):
    """Rank elements using a recovery estimator and select largest prefix under cap.

    scikit-fem mesh.refined() may add closure elements to retain conformity,
    therefore number of newly created elements is not 1-to-1 with marked ids.
    """
    n = int(env.num_agents)
    actions = np.zeros((n, 1), dtype=np.float32)
    if n >= target_elements:
        return actions, 0, n

    u = np.asarray(env.scalar_solution).reshape(-1)
    mesh = env.fem_problem.mesh
    scores = np.asarray(estimator._get_error_per_element(mesh=mesh, solution=u)).reshape(-1)
    if len(scores) != n:
        raise ValueError(f'ZZ returned {len(scores)} scores for {n} elements')
    scores = np.nan_to_num(scores, nan=-np.inf)
    order = np.argsort(scores, kind='stable')[::-1]

    def count_after(k):
        if k == 0:
            return n
        indices = np.asarray(order[:k], dtype=np.int64)
        return int(mesh.refined(indices).t.shape[1])

    left, right = 0, n
    while left < right:
        middle = (left + right + 1) // 2
        if count_after(middle) <= target_elements:
            left = middle
        else:
            right = middle - 1

    k = left
    if k:
        actions[order[:k], 0] = 1.0
    return actions, k, count_after(k)


def run_case(algorithm, base_env, case, penalty, estimator):
    # One sampled PDE, cloned BEFORE either method adapts: identical RHS,
    # boundary conditions, initial mesh and normalization reference.
    base_env.reset()
    base_env.set_element_penalty_lambda(penalty, from_position=False)
    ppo_env = copy.deepcopy(base_env)
    zz_env = copy.deepcopy(base_env)
    assert np.array_equal(ppo_env.fem_problem.mesh.t, zz_env.fem_problem.mesh.t)
    assert np.allclose(ppo_env.fem_problem.mesh.p, zz_env.fem_problem.mesh.p)
    assert np.isclose(numeric_error(ppo_env), numeric_error(zz_env))

    rows = [
        record(case, 'ASMR++ PPO', 0, ppo_env, 0, 0, 0),
        record(case, 'ZZ-AMR', 0, zz_env, 0, 0, 0),
    ]
    budgets = [int(ppo_env.num_agents)]
    ppo_cumulative = 0.0
    zz_cumulative = 0.0
    ppo_done, zz_done = False, False
    max_steps = int(ppo_env._max_timesteps)

    for step in range(1, max_steps + 1):
        # NN strategy: deterministic saved policy.
        if ppo_done:
            raise RuntimeError(f'ASMR++ became terminal unexpectedly at step {step}')
        start = perf_counter()
        actions, _ = algorithm.policy_step(observation=ppo_env.last_observation)
        actions = np.asarray(detach(actions)).reshape(-1, 1)
        decision = perf_counter() - start
        start = perf_counter()
        _, _, ppo_done, _ = ppo_env.step(action=actions)
        fem_time = perf_counter() - start
        ppo_cumulative += decision + fem_time
        budgets.append(int(ppo_env.num_agents))
        rows.append(record(case, 'ASMR++ PPO', step, ppo_env, decision, fem_time, ppo_cumulative))

        # Classical ZZ-AMR: recovered-gradient error indicator, not the fine-
        # grid reference error, which is only used for common evaluation.
        if zz_done:
            raise RuntimeError(f'ZZ became terminal unexpectedly at step {step}')
        start = perf_counter()
        actions_zz, k, predicted_n = zz_actions_at_budget(zz_env, budgets[-1], estimator)
        decision_zz = perf_counter() - start
        start = perf_counter()
        _, _, zz_done, _ = zz_env.step(action=actions_zz)
        fem_time_zz = perf_counter() - start
        zz_cumulative += decision_zz + fem_time_zz
        actual_n = int(zz_env.num_agents)
        if actual_n != predicted_n:
            raise AssertionError(f'Predicted ZZ elements {predicted_n}; actual {actual_n}')
        if actual_n > budgets[-1]:
            raise AssertionError(f'ZZ exceeded matched budget at step {step}')
        rows.append(record(case, 'ZZ-AMR', step, zz_env, decision_zz, fem_time_zz, zz_cumulative))
        print(f'  Step {step}: PPO N={budgets[-1]:4d}, eta={numeric_error(ppo_env):.5f}; '
              f'ZZ N={actual_n:4d}, eta={numeric_error(zz_env):.5f} (marked {k})', flush=True)

    return rows


def make_outputs(df, output, num_cases):
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / 'cases.csv', index=False)
    summary = (df.groupby(['method', 'step'], as_index=False)
               .agg(N_mean=('elements', 'mean'), N_std=('elements', 'std'),
                    error_mean=('error', 'mean'), error_std=('error', 'std'),
                    time_mean_s=('cumulative_s', 'mean')))
    summary.to_csv(output / 'summary.csv', index=False)

    stepmax = int(df['step'].max())
    last = df[df['step'] == stepmax].pivot(index='case', columns='method', values=['error','elements','cumulative_s'])
    paired = pd.DataFrame({
        'case': last.index,
        'PPO_N': last[('elements','ASMR++ PPO')].values,
        'ZZ_N': last[('elements','ZZ-AMR')].values,
        'PPO_error': last[('error','ASMR++ PPO')].values,
        'ZZ_error': last[('error','ZZ-AMR')].values,
        'PPO_runtime_s': last[('cumulative_s','ASMR++ PPO')].values,
        'ZZ_runtime_s': last[('cumulative_s','ZZ-AMR')].values,
    })
    paired['PPO_error_gain_vs_ZZ_pct'] = 100 * (paired['ZZ_error'] - paired['PPO_error']) / paired['ZZ_error'].replace(0, np.nan)
    paired.to_csv(output / 'final_paired.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method, data in summary.groupby('method'):
        ax.plot(data['N_mean'], data['error_mean'], '-o', label=method)
    ax.set(xlabel='Mean number of elements', ylabel='Mean normalized reference error',
           title=f'ASMR++ vs ZZ-AMR: matched budgets, {num_cases} Poisson cases')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'error_vs_elements.png', dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method, data in summary.groupby('method'):
        ax.plot(data['step'], data['error_mean'], '-o', label=method)
    ax.set(xlabel='Refinement step', ylabel='Mean normalized reference error',
           title=f'ASMR++ vs ZZ-AMR, {num_cases} matched Poisson cases')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'error_vs_step.png', dpi=220)
    plt.close(fig)

    summary_lines = [
        '# ASMR++ vs ZZ-AMR — benchmark notes',
        f'Number of paired PDE tasks: {num_cases}.',
        f'Adaptive steps: {stepmax}.',
        'Policy: deterministic, saved ASMR++ PPO after 10 training iterations.',
        'ZZ scoring: built-in ZienkiewiczZhuErrorHeuristic recovered-gradient estimator.',
        'Budget: ZZ element count at every step <= ASMR++ count at that step.',
        'Same sampled PDE, initial mesh, FEM solve and normalized reference error for each pair.',
        'Reference metric is not necessarily exact analytical error.',
        'Timing is exploratory: ZZ budget matching performs additional trial refinements;',
 'therefore do not infer optimized solver wall-time or speedup from this script.',
        '',
        '## Final-step paired comparison',
        f"PPO mean N: {paired['PPO_N'].mean():.2f}",
        f"ZZ mean N: {paired['ZZ_N'].mean():.2f}",
        f"PPO mean error: {paired['PPO_error'].mean():.6f}",
        f"ZZ mean error: {paired['ZZ_error'].mean():.6f}",
        f"PPO error smaller in {(paired['PPO_error'] < paired['ZZ_error']).sum()}/{num_cases} pairs.",
        f"PPO error gain (mean of paired percentages): {paired['PPO_error_gain_vs_ZZ_pct'].mean():.2f}%",
        '',
        'Interpreting N-efficiency requires care because matched counts may not be exactly equal.',
        'For a paper, run repeated seeds and compare at truly matched N or interpolated error(N).',
    ]
    (output / 'README_results.txt').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')

    print('\n' + '='*65)
    print('FINAL-STEP SUMMARY')
    print(summary[summary['step'] == stepmax].round(5).to_string(index=False))
    print(f"PPO error lower in {(paired['PPO_error'] < paired['ZZ_error']).sum()}/{num_cases} cases")
    print(f"Mean paired PPO gain (positive=better): {paired['PPO_error_gain_vs_ZZ_pct'].mean():.2f}%")
    print('Results saved:', output.resolve())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=int, default=10, help='Number of paired Poisson tasks')
    parser.add_argument('--penalty', type=float, default=0.001274)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output', type=Path, default=Path('reports/analysis/asmr_vs_zz'))
    args = parser.parse_args()
    if args.cases < 1:
        raise ValueError('Require --cases >= 1')

    cfg_path = Path('reports/poisson_asmr_train10/log/rep_00/config.yaml')
    if not cfg_path.exists():
        raise FileNotFoundError('Run script from ASMRplusplus root; missing ' + str(cfg_path))
    with cfg_path.open(encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['algorithm']['checkpoint'] = dict(
        experiment_name='poisson_asmr_train10',
        repetition=0,
        load_root_dir='./reports',
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    algorithm = create_algorithm(config=config, seed=args.seed)
    algorithm.set_training_mode(False)
    base_env = algorithm.evaluation_environments[1]
    if base_env.refinement_strategy not in ('absolute', 'absolute_discrete'):
        raise ValueError('This experiment needs elementwise absolute refinement actions')
    estimator = ZienkiewiczZhuErrorHeuristic(theta=0.5)

    all_rows = []
    for case in range(args.cases):
        print(f'\nCASE {case + 1}/{args.cases}', flush=True)
        all_rows.extend(run_case(algorithm, base_env, case, args.penalty, estimator))
    make_outputs(pd.DataFrame(all_rows), args.output, args.cases)


if __name__ == '__main__':
    main()
