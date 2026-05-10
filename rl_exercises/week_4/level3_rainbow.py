"""
Level 3 runner for Rainbow-style DQN ablations.

Runs four CartPole configurations:
- Base DQN
- DQN + prioritized replay
- DQN + Double DQN
- DQN + prioritized replay + Double DQN
"""

from typing import Callable, Dict, List, Tuple

from pathlib import Path

import gymnasium as gym
import hydra
import matplotlib
import numpy as np
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from rliable import metrics
from rliable.library import get_interval_estimates

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from rl_exercises.week_4.dqn import DQNAgent, set_seed
from rl_exercises.week_4.level2_rliable import (
    aggregate_over_steps,
    align_curves,
    optimality_gap,
    save_curve_csv,
)
from rliable.plot_utils import plot_sample_efficiency_curve

Curve = List[Tuple[int, float]]


def agent_kwargs_from_cfg(
    cfg: DictConfig, seed: int, prioritized_replay: bool, double_dqn: bool
) -> dict:
    return dict(
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        hidden_dim=cfg.agent.hidden_dim,
        num_hidden_layers=cfg.agent.num_hidden_layers,
        seed=seed,
        prioritized_replay=prioritized_replay,
        double_dqn=double_dqn,
        per_alpha=cfg.agent.per_alpha,
        per_beta=cfg.agent.per_beta,
        per_eps=cfg.agent.per_eps,
    )


def save_comparison_plot(
    steps: np.ndarray,
    scores_by_algorithm: Dict[str, np.ndarray],
    aggregate_fn: Callable,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    estimates, intervals = get_interval_estimates(
        scores_by_algorithm,
        aggregate_fn,
        reps=2000,
    )
    algorithms = list(scores_by_algorithm.keys())
    plot_sample_efficiency_curve(
        steps,
        estimates,
        intervals,
        algorithms=algorithms,
        xlabel="Frames",
        ylabel=ylabel,
    )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_observations_template(
    output_dir: Path, seeds: List[int], cfg: DictConfig
) -> None:
    path = output_dir / "observations_l3.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Level 3 observations\n\n")
        f.write("I compared four Rainbow-style DQN ablations on CartPole-v1.\n")
        f.write(f"Seeds used: {seeds}\n\n")
        f.write("Configurations:\n")
        f.write("- Base DQN\n")
        f.write("- DQN + prioritized replay\n")
        f.write("- DQN + Double DQN\n")
        f.write("- DQN + prioritized replay + Double DQN\n\n")
        f.write("Shared hyperparameters:\n")
        f.write(f"- hidden_dim = {cfg.agent.hidden_dim}\n")
        f.write(f"- hidden_layers = {cfg.agent.num_hidden_layers}\n")
        f.write(f"- buffer_capacity = {cfg.agent.buffer_capacity}\n")
        f.write(f"- batch_size = {cfg.agent.batch_size}\n")
        f.write(f"- learning_rate = {cfg.agent.learning_rate}\n")
        f.write(f"- gamma = {cfg.agent.gamma}\n")
        f.write(f"- epsilon_decay = {cfg.agent.epsilon_decay}\n")
        f.write(f"- target_update_freq = {cfg.agent.target_update_freq}\n")
        f.write(f"- per_alpha = {cfg.agent.per_alpha}\n")
        f.write(f"- per_beta = {cfg.agent.per_beta}\n\n")
        f.write(
            "Use the RLiable plots in this folder to discuss whether prioritized replay, "
        )
        f.write(
            "Double DQN, or their combination improved sample efficiency or final score.\n"
        )


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig) -> None:
    seeds = [0, 1, 2, 3, 4]
    variants = {
        "Base DQN": (False, False),
        "DQN + PER": (True, False),
        "DQN + Double": (False, True),
        "DQN + PER + Double": (True, True),
    }

    output_dir = Path(get_original_cwd()) / "rl_exercises" / "week_4" / "output_l3"
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_by_algorithm: Dict[str, np.ndarray] = {}
    steps = None
    for algorithm, (prioritized_replay, double_dqn) in variants.items():
        curves: List[Curve] = []
        variant_dir = output_dir / algorithm.lower().replace(" + ", "_").replace(
            " ", "_"
        )
        variant_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            env = gym.make(cfg.env.name)
            set_seed(env, seed)
            agent = DQNAgent(
                env, **agent_kwargs_from_cfg(cfg, seed, prioritized_replay, double_dqn)
            )
            agent.train(cfg.train.num_frames, cfg.train.eval_interval)
            curves.append(agent.training_curve)
            save_curve_csv(agent.training_curve, variant_dir / f"seed_{seed}.csv")
            env.close()

        steps, scores = align_curves(curves, cfg.train.num_frames)
        scores_by_algorithm[algorithm] = scores
        np.save(variant_dir / "aligned_scores.npy", scores)

    assert steps is not None
    save_comparison_plot(
        steps,
        scores_by_algorithm,
        aggregate_over_steps(metrics.aggregate_iqm),
        "Rainbow ablation IQM training curves",
        "IQM mean reward",
        output_dir / "iqm_comparison.png",
    )
    save_comparison_plot(
        steps,
        scores_by_algorithm,
        aggregate_over_steps(metrics.aggregate_mean),
        "Rainbow ablation mean training curves",
        "Mean reward",
        output_dir / "mean_comparison.png",
    )
    save_comparison_plot(
        steps,
        scores_by_algorithm,
        aggregate_over_steps(metrics.aggregate_median),
        "Rainbow ablation median training curves",
        "Median reward",
        output_dir / "median_comparison.png",
    )
    normalized_scores = {
        algorithm: np.clip(scores / 500.0, 0.0, 1.0)
        for algorithm, scores in scores_by_algorithm.items()
    }
    save_comparison_plot(
        steps,
        normalized_scores,
        aggregate_over_steps(optimality_gap),
        "Rainbow ablation optimality gap",
        "Optimality gap",
        output_dir / "optimality_gap_comparison.png",
    )
    write_observations_template(output_dir, seeds, cfg)


if __name__ == "__main__":
    main()
