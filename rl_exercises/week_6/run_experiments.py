"""Experiment runner for week 6 plots.

The default step budgets are intentionally modest so the script can be run on a
laptop. Increase the command-line budgets for more reliable final reports.
"""

from __future__ import annotations

from typing import Callable

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-week6"))

import gymnasium as gym
import matplotlib
import numpy as np
import torch
from rliable import metrics
from rliable.library import get_interval_estimates

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from rl_exercises.week_6.actor_critic import ActorCriticAgent, set_seed
from rl_exercises.week_6.ppo import PPOAgent
from rl_exercises.week_6.sac import SACAgent
from rliable.plot_utils import plot_sample_efficiency_curve

Curve = list[tuple[int, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("rl_exercises/week_6"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--l1-cartpole-steps", type=int, default=8000)
    parser.add_argument("--l1-lunar-steps", type=int, default=12000)
    parser.add_argument("--l2-steps", type=int, default=12000)
    parser.add_argument("--l3-steps", type=int, default=12000)
    parser.add_argument("--eval-interval", type=int, default=2000)
    return parser.parse_args()


def make_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_curve_csv(curve: Curve, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "average_return"])
        writer.writerows(curve)


def align_curves(
    curves: list[Curve], total_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    steps = np.array(sorted({step for curve in curves for step, _ in curve}))
    if len(steps) == 0:
        steps = np.array([total_steps])

    aligned = []
    for curve in curves:
        frames = np.array([0] + [step for step, _ in curve], dtype=float)
        rewards = np.array([0.0] + [reward for _, reward in curve], dtype=float)
        aligned.append(np.interp(steps, frames, rewards))
    return steps, np.asarray(aligned)


def aggregate_over_steps(metric_fn: Callable[[np.ndarray], float]) -> Callable:
    return (
        lambda scores: np.array(  # noqa: E731
            [metric_fn(scores[:, idx : idx + 1]) for idx in range(scores.shape[-1])]
        )
    )


def save_rliable_plot(
    curves_by_name: dict[str, list[Curve]],
    total_steps: int,
    title: str,
    output_path: Path,
) -> dict[str, np.ndarray]:
    steps = None
    score_dict = {}
    for name, curves in curves_by_name.items():
        alg_steps, scores = align_curves(curves, total_steps)
        if steps is None:
            steps = alg_steps
        else:
            scores = np.asarray(
                [np.interp(steps, alg_steps, seed_scores) for seed_scores in scores]
            )
        score_dict[name] = scores

    assert steps is not None
    estimates, intervals = get_interval_estimates(
        score_dict,
        aggregate_over_steps(metrics.aggregate_mean),
        reps=1000,
    )
    plot_sample_efficiency_curve(
        steps,
        estimates,
        intervals,
        algorithms=list(score_dict.keys()),
        xlabel="Environment steps",
        ylabel="Average return",
    )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return score_dict


def evaluate_discrete_policy(
    agent: ActorCriticAgent | PPOAgent,
    env_id: str,
    seed: int,
    episodes: int,
) -> float:
    eval_env = gym.make(env_id)
    returns = []
    for episode in range(episodes):
        state, _ = eval_env.reset(seed=seed + 1000 + episode)
        done = False
        total_return = 0.0
        while not done:
            if isinstance(agent, ActorCriticAgent):
                action, _ = agent.predict_action(state, evaluate=True)
            else:
                with torch.no_grad():
                    state_t = torch.as_tensor(state, dtype=torch.float32)
                    probs = agent.policy(state_t).squeeze(0)
                    action = int(torch.argmax(probs).item())
            state, reward, term, trunc, _ = eval_env.step(action)
            done = term or trunc
            total_return += float(reward)
        returns.append(total_return)
    eval_env.close()
    return float(np.mean(returns))


def train_actor_critic(
    env_id: str,
    baseline: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
) -> Curve:
    env = gym.make(env_id)
    set_seed(env, seed)
    agent = ActorCriticAgent(
        env,
        seed=seed,
        baseline_type=baseline,
        lr_actor=5e-4 if env_id == "CartPole-v1" else 5e-3,
        lr_critic=1e-3 if env_id == "CartPole-v1" else 1e-2,
    )
    curve: Curve = []
    step_count = 0
    while step_count < total_steps:
        state, _ = env.reset(seed=seed + step_count)
        done = False
        trajectory = []
        while not done and step_count < total_steps:
            action, logp = agent.predict_action(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            trajectory.append((state, action, float(reward), next_state, done, logp))
            state = next_state
            step_count += 1
            if step_count % eval_interval == 0:
                curve.append(
                    (
                        step_count,
                        evaluate_discrete_policy(agent, env_id, seed, eval_episodes),
                    )
                )
        agent.update_agent(trajectory)
    if not curve or curve[-1][0] != total_steps:
        curve.append(
            (total_steps, evaluate_discrete_policy(agent, env_id, seed, eval_episodes))
        )
    env.close()
    return curve


def train_ppo(
    env_id: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
    enhanced: bool,
) -> Curve:
    env = gym.make(env_id)
    set_seed(env, seed)
    agent = PPOAgent(
        env,
        seed=seed,
        target_kl=0.03 if enhanced else None,
        max_grad_norm=0.5 if enhanced else 10.0,
    )
    curve: Curve = []
    step_count = 0
    while step_count < total_steps:
        state, _ = env.reset(seed=seed + step_count)
        done = False
        trajectory = []
        while not done and step_count < total_steps:
            action, logp, ent, _ = agent.predict(state)
            next_state, reward, term, trunc, _ = env.step(action)
            done = term or trunc
            trajectory.append(
                (state, action, logp, ent, reward, float(done), next_state)
            )
            state = next_state
            step_count += 1
            if step_count % eval_interval == 0:
                curve.append(
                    (
                        step_count,
                        evaluate_discrete_policy(agent, env_id, seed, eval_episodes),
                    )
                )
        agent.update(trajectory)
    if not curve or curve[-1][0] != total_steps:
        curve.append(
            (total_steps, evaluate_discrete_policy(agent, env_id, seed, eval_episodes))
        )
    env.close()
    return curve


def train_sac(
    env_id: str,
    seed: int,
    total_steps: int,
    eval_interval: int,
    eval_episodes: int,
) -> Curve:
    env = gym.make(env_id)
    set_seed(env, seed)
    agent = SACAgent(
        env,
        seed=seed,
        learning_starts=min(1000, max(100, total_steps // 10)),
        batch_size=128,
    )
    eval_env = gym.make(env_id)
    curve: Curve = []
    state, _ = env.reset(seed=seed)
    for step in range(1, total_steps + 1):
        if step < agent.learning_starts:
            action = env.action_space.sample()
        else:
            action, _ = agent.predict_action(state)

        next_state, reward, term, trunc, _ = env.step(action)
        done = term or trunc
        agent.replay_buffer.add(state, action, float(reward), next_state, done)
        state = next_state
        if done:
            state, _ = env.reset(seed=seed + step)

        if step >= agent.learning_starts:
            agent.update_agent()

        if step % eval_interval == 0:
            mean_return, _ = agent.evaluate(eval_env, eval_episodes)
            curve.append((step, mean_return))

    if not curve or curve[-1][0] != total_steps:
        mean_return, _ = agent.evaluate(eval_env, eval_episodes)
        curve.append((total_steps, mean_return))
    eval_env.close()
    env.close()
    return curve


def main() -> None:
    args = parse_args()

    l1_dir = make_output_dir(args.output_dir / "output_l1")
    l2_dir = make_output_dir(args.output_dir / "output_l2")
    l3_dir = make_output_dir(args.output_dir / "output_l3")

    baselines = ["none", "avg", "value", "gae"]
    l1_cart: dict[str, list[Curve]] = {name: [] for name in baselines}
    l1_lunar: dict[str, list[Curve]] = {name: [] for name in baselines}

    for env_id, total_steps, target in [
        ("CartPole-v1", args.l1_cartpole_steps, l1_cart),
        ("LunarLander-v3", args.l1_lunar_steps, l1_lunar),
    ]:
        for baseline in baselines:
            for seed in args.seeds:
                print(f"L1 {env_id} baseline={baseline} seed={seed}")
                curve = train_actor_critic(
                    env_id,
                    baseline,
                    seed,
                    total_steps,
                    args.eval_interval,
                    args.eval_episodes,
                )
                target[baseline].append(curve)
                save_curve_csv(
                    curve,
                    l1_dir / env_id / baseline / f"seed_{seed}.csv",
                )

    save_rliable_plot(
        l1_cart,
        args.l1_cartpole_steps,
        "Actor-Critic baselines on CartPole-v1",
        l1_dir / "actor_critic_cartpole.png",
    )
    save_rliable_plot(
        l1_lunar,
        args.l1_lunar_steps,
        "Actor-Critic baselines on LunarLander-v3",
        l1_dir / "actor_critic_lunarlander.png",
    )

    l2_curves: dict[str, list[Curve]] = {
        "ActorCritic-GAE": [],
        "PPO-vanilla": [],
        "PPO-enhanced": [],
    }
    for seed in args.seeds:
        print(f"L2 ActorCritic-GAE seed={seed}")
        l2_curves["ActorCritic-GAE"].append(
            train_actor_critic(
                "LunarLander-v3",
                "gae",
                seed,
                args.l2_steps,
                args.eval_interval,
                args.eval_episodes,
            )
        )
        print(f"L2 PPO vanilla seed={seed}")
        l2_curves["PPO-vanilla"].append(
            train_ppo(
                "LunarLander-v3",
                seed,
                args.l2_steps,
                args.eval_interval,
                args.eval_episodes,
                enhanced=False,
            )
        )
        print(f"L2 PPO enhanced seed={seed}")
        l2_curves["PPO-enhanced"].append(
            train_ppo(
                "LunarLander-v3",
                seed,
                args.l2_steps,
                args.eval_interval,
                args.eval_episodes,
                enhanced=True,
            )
        )

    for name, runs in l2_curves.items():
        for seed, curve in zip(args.seeds, runs):
            save_curve_csv(curve, l2_dir / name / f"seed_{seed}.csv")
    save_rliable_plot(
        l2_curves,
        args.l2_steps,
        "PPO variants and Actor-Critic on LunarLander-v3",
        l2_dir / "ppo_vs_actor_critic_lunarlander.png",
    )

    l3_curves = {"SAC": []}
    for seed in args.seeds:
        print(f"L3 SAC Pendulum-v1 seed={seed}")
        l3_curves["SAC"].append(
            train_sac(
                "Pendulum-v1",
                seed,
                args.l3_steps,
                args.eval_interval,
                args.eval_episodes,
            )
        )
    for seed, curve in zip(args.seeds, l3_curves["SAC"]):
        save_curve_csv(curve, l3_dir / "SAC" / f"seed_{seed}.csv")
    save_rliable_plot(
        l3_curves,
        args.l3_steps,
        "SAC on Pendulum-v1",
        l3_dir / "sac_pendulum.png",
    )

    print(f"Saved week 6 outputs under {args.output_dir}")


if __name__ == "__main__":
    main()
