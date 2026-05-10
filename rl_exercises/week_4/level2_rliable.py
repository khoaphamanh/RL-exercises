"""
Level 2 runner for multi-seed DQN experiments with RLiable plots.
"""

from typing import Callable, List, Tuple

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
from rliable.plot_utils import plot_sample_efficiency_curve

Curve = List[Tuple[int, float]]


def agent_kwargs_from_cfg(cfg: DictConfig, seed: int) -> dict:
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
    )


def experiment_name(cfg: DictConfig) -> str:
    return (
        f"dqn_hidden{cfg.agent.hidden_dim}"
        f"_layers{cfg.agent.num_hidden_layers}"
        f"_buffer{cfg.agent.buffer_capacity}"
        f"_batch{cfg.agent.batch_size}"
        f"_lr{cfg.agent.learning_rate}"
        f"_gamma{cfg.agent.gamma}"
        f"_epsdecay{cfg.agent.epsilon_decay}"
        f"_target{cfg.agent.target_update_freq}"
    )


def save_curve_csv(curve: Curve, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("frame,mean_reward\n")
        for frame, mean_reward in curve:
            f.write(f"{frame},{mean_reward}\n")


def align_curves(curves: List[Curve], num_frames: int, n_points: int = 101) -> tuple:
    steps = np.linspace(0, num_frames, n_points, dtype=int)
    aligned_scores = []

    for curve in curves:
        frames = np.array([0] + [frame for frame, _ in curve], dtype=float)
        rewards = np.array([0.0] + [reward for _, reward in curve], dtype=float)
        aligned_scores.append(np.interp(steps, frames, rewards))

    return steps, np.array(aligned_scores)


def aggregate_over_steps(metric_fn: Callable[[np.ndarray], float]) -> Callable:
    return (
        lambda scores: np.array(  # noqa: E731
            [
                metric_fn(scores[:, eval_idx : eval_idx + 1])
                for eval_idx in range(scores.shape[-1])
            ]
        )
    )


def optimality_gap(scores: np.ndarray) -> float:
    return metrics.aggregate_optimality_gap(scores, gamma=1.0)


def save_rliable_plot(
    steps: np.ndarray,
    scores: np.ndarray,
    aggregate_fn: Callable,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    estimates, intervals = get_interval_estimates(
        {"DQN": scores},
        aggregate_fn,
        reps=2000,
    )
    plot_sample_efficiency_curve(
        steps,
        estimates,
        intervals,
        algorithms=["DQN"],
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
    path = output_dir / "observations_l2.txt"
    with path.open("w", encoding="utf-8") as f:
        f.write("Level 2 observations\n\n")
        f.write("I reran DQN on CartPole-v1 with multiple random seeds.\n")
        f.write(f"Seeds used: {seeds}\n\n")
        f.write("The plots and CSV files are saved under:\n")
        f.write(f"{output_dir}\n\n")
        f.write("Configuration:\n")
        f.write(f"- hidden_dim = {cfg.agent.hidden_dim}\n")
        f.write(f"- hidden_layers = {cfg.agent.num_hidden_layers}\n")
        f.write(f"- buffer_capacity = {cfg.agent.buffer_capacity}\n")
        f.write(f"- batch_size = {cfg.agent.batch_size}\n")
        f.write(f"- learning_rate = {cfg.agent.learning_rate}\n")
        f.write(f"- gamma = {cfg.agent.gamma}\n")
        f.write(f"- epsilon_decay = {cfg.agent.epsilon_decay}\n")
        f.write(f"- target_update_freq = {cfg.agent.target_update_freq}\n\n")
        f.write("What changed compared with plain averages:\n")
        f.write(
            "RLiable summarizes the result across seeds instead of relying on one run. "
            "The IQM curve is less sensitive to unusually good or bad seeds than a "
            "plain mean curve. The confidence intervals also make the uncertainty "
            "between seeds visible.\n\n"
        )
        f.write("Do I feel more confident in the result?\n")
        f.write(
            "Yes, because a single DQN run can be misleading. Running five seeds gives "
            "a better picture of typical behavior and shows how much variation there "
            "is between runs. I would still be careful about strong conclusions, "
            "because five seeds is useful but still a small sample.\n"
        )


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig) -> None:
    seeds = [0, 1, 2, 3, 4]
    output_dir = (
        Path(get_original_cwd())
        / "rl_exercises"
        / "week_4"
        / "output_l2"
        / experiment_name(cfg)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = []
    for seed in seeds:
        env = gym.make(cfg.env.name)
        set_seed(env, seed)
        agent = DQNAgent(env, **agent_kwargs_from_cfg(cfg, seed))
        agent.train(cfg.train.num_frames, cfg.train.eval_interval)
        curves.append(agent.training_curve)
        save_curve_csv(agent.training_curve, output_dir / f"seed_{seed}.csv")
        env.close()

    steps, scores = align_curves(curves, cfg.train.num_frames)
    np.save(output_dir / "aligned_scores.npy", scores)

    save_rliable_plot(
        steps,
        scores,
        aggregate_over_steps(metrics.aggregate_iqm),
        "DQN IQM training curve across 5 seeds",
        "IQM mean reward",
        output_dir / "iqm_training_curve.png",
    )
    save_rliable_plot(
        steps,
        scores,
        aggregate_over_steps(metrics.aggregate_mean),
        "DQN mean training curve across 5 seeds",
        "Mean reward",
        output_dir / "mean_training_curve.png",
    )
    save_rliable_plot(
        steps,
        scores,
        aggregate_over_steps(metrics.aggregate_median),
        "DQN median training curve across 5 seeds",
        "Median reward",
        output_dir / "median_training_curve.png",
    )

    normalized_scores = np.clip(scores / 500.0, 0.0, 1.0)
    save_rliable_plot(
        steps,
        normalized_scores,
        aggregate_over_steps(optimality_gap),
        "DQN optimality gap across 5 seeds",
        "Optimality gap",
        output_dir / "optimality_gap_curve.png",
    )

    write_observations_template(output_dir, seeds, cfg)


if __name__ == "__main__":
    main()
