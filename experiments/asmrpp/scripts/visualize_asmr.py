from pathlib import Path

import numpy as np
import torch
import yaml
import plotly.graph_objects as go

from src.algorithms import create_algorithm
from util.torch_util.torch_util import detach

# Каталоги
run_dir = Path("reports/poisson_asmr_train10/log/rep_00")
output = Path("reports/analysis/mesh_visualization")
output.mkdir(parents=True, exist_ok=True)

# Загружаем конфигурацию проведённого эксперимента
with open(run_dir / "config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Подключаем обученную модель
config["algorithm"]["checkpoint"] = {
    "experiment_name": "poisson_asmr_train10",
    "repetition": 0,
    "load_root_dir": "./reports"
}

np.random.seed(123)
torch.manual_seed(123)

algorithm = create_algorithm(config=config, seed=123)
algorithm.set_training_mode(False)

# Выбираем первую оценочную среду
env = algorithm.evaluation_environments[0]
env.reset()

# Фиксируем штраф
penalty = 0.001274
env.set_element_penalty_lambda(
    penalty,
    from_position=False
)

def save_state(step):
    traces, layout = env.render(
        mode="human",
        render_intermediate_steps=True
    )

    fig = go.Figure(data=traces, **layout)

    fig.write_html(
        str(output / f"mesh_step_{step}.html"),
        include_plotlyjs=True
    )

    error_fig = env._plot_error_per_element(
        normalize_by_element_volume=False
    )

    error_fig.write_html(
        str(output / f"error_step_{step}.html"),
        include_plotlyjs=True
    )

    print(
        f"Step {step}: "
        f"N={env.num_agents}, "
        f"error={float(env._get_remaining_error()):.6f}"
    )

# Начальное состояние
save_state(0)

# Детерминированная адаптация
done = False
step = 0

while not done:
    step += 1

    actions, values = algorithm.policy_step(
        observation=env.last_observation
    )

    observation, reward, done, info = env.step(
        action=detach(actions)
    )

    save_state(step)

print("\nВизуализация завершена.")
print("Результаты:", output.resolve())
