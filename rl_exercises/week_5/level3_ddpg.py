"""
Level 3 DDPG experiment for a continuous-control environment.

The script compares DDPG against a simple Gaussian REINFORCE baseline on
Pendulum-v1 and writes CSV logs plus a plot to rl_exercises/week_5/output_l3.
"""

from __future__ import annotations

from typing import Deque, List, Tuple

import argparse
import csv
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def set_seed(env: gym.Env, seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


class ContinuousReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = (
            deque(maxlen=capacity)
        )

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *(self.buffer[i] for i in indices)
        )
        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32),
            torch.as_tensor(np.asarray(actions), dtype=torch.float32),
            torch.as_tensor(np.asarray(rewards), dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32),
            torch.as_tensor(np.asarray(dones), dtype=torch.float32).unsqueeze(-1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_limit: float) -> None:
        super().__init__()
        self.action_limit = action_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.action_limit * self.net(state)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class DDPGAgent:
    def __init__(
        self,
        env: gym.Env,
        actor_lr: float,
        critic_lr: float,
        gamma: float,
        tau: float,
        buffer_capacity: int,
        batch_size: int,
        exploration_noise: float,
        seed: int,
    ) -> None:
        set_seed(env, seed)
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.exploration_noise = exploration_noise
        self.action_low = env.action_space.low.astype(np.float32)
        self.action_high = env.action_space.high.astype(np.float32)
        self.action_limit = float(np.max(np.abs(self.action_high)))

        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        self.actor = Actor(obs_dim, action_dim, self.action_limit)
        self.actor_target = Actor(obs_dim, action_dim, self.action_limit)
        self.critic = Critic(obs_dim, action_dim)
        self.critic_target = Critic(obs_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.buffer = ContinuousReplayBuffer(buffer_capacity)

    def predict_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).numpy()
        if not evaluate:
            action += np.random.normal(0.0, self.exploration_noise, size=action.shape)
        return np.clip(action, self.action_low, self.action_high)

    def update(self) -> None:
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * (1.0 - dones) * target_q

        q = self.critic(states, actions)
        critic_loss = F.mse_loss(q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * source_param.data)


class GaussianReinforcePolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_limit: float) -> None:
        super().__init__()
        self.action_limit = action_limit
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.action_limit * torch.tanh(self.net(state))
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std


class ContinuousReinforceAgent:
    def __init__(self, env: gym.Env, lr: float, gamma: float, seed: int) -> None:
        set_seed(env, seed)
        self.env = env
        self.gamma = gamma
        self.action_low = env.action_space.low.astype(np.float32)
        self.action_high = env.action_space.high.astype(np.float32)
        action_limit = float(np.max(np.abs(self.action_high)))
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        self.policy = GaussianReinforcePolicy(obs_dim, action_dim, action_limit)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def predict_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[np.ndarray, torch.Tensor | None]:
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        mean, std = self.policy(state_t)
        if evaluate:
            action = mean.squeeze(0).detach().numpy()
            return np.clip(action, self.action_low, self.action_high), None

        dist = torch.distributions.Normal(mean, std)
        action_t = dist.sample()
        log_prob = dist.log_prob(action_t).sum(dim=-1).squeeze(0)
        action = action_t.squeeze(0).detach().numpy()
        return np.clip(action, self.action_low, self.action_high), log_prob

    def update(self, log_probs: List[torch.Tensor], rewards: List[float]) -> None:
        returns = []
        running_return = 0.0
        for reward in reversed(rewards):
            running_return = reward + self.gamma * running_return
            returns.insert(0, running_return)

        returns_t = torch.as_tensor(returns, dtype=torch.float32)
        returns_t = (returns_t - returns_t.mean()) / (
            returns_t.std(unbiased=False) + 1e-8
        )
        log_probs_t = torch.stack(log_probs)
        loss = -(log_probs_t * returns_t).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train_ddpg(agent: DDPGAgent, episodes: int, max_steps: int) -> List[float]:
    returns = []
    for episode in range(1, episodes + 1):
        state, _ = agent.env.reset()
        episode_return = 0.0
        for _ in range(max_steps):
            action = agent.predict_action(state)
            next_state, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated
            agent.buffer.add(state, action, float(reward), next_state, done)
            agent.update()
            state = next_state
            episode_return += float(reward)
            if done:
                break
        returns.append(episode_return)
        if episode % 10 == 0:
            print(f"[DDPG] Ep {episode:4d} Return {episode_return:8.1f}")
    return returns


def train_reinforce(
    agent: ContinuousReinforceAgent, episodes: int, max_steps: int
) -> List[float]:
    returns = []
    for episode in range(1, episodes + 1):
        state, _ = agent.env.reset()
        log_probs: List[torch.Tensor] = []
        rewards: List[float] = []
        episode_return = 0.0
        for _ in range(max_steps):
            action, log_prob = agent.predict_action(state)
            next_state, reward, terminated, truncated, _ = agent.env.step(action)
            done = terminated or truncated
            if log_prob is not None:
                log_probs.append(log_prob)
            rewards.append(float(reward))
            state = next_state
            episode_return += float(reward)
            if done:
                break
        agent.update(log_probs, rewards)
        returns.append(episode_return)
        if episode % 10 == 0:
            print(f"[REINFORCE] Ep {episode:4d} Return {episode_return:8.1f}")
    return returns


def moving_average(values: List[float], window: int = 10) -> np.ndarray:
    values_arr = np.asarray(values, dtype=np.float32)
    if len(values_arr) < window:
        return values_arr
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values_arr, kernel, mode="valid")


def save_csv(path: Path, returns: List[float]) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["episode", "return"])
        for episode, value in enumerate(returns, start=1):
            writer.writerow([episode, value])


def save_plot(
    path: Path, ddpg_returns: List[float], reinforce_returns: List[float]
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(moving_average(ddpg_returns), label="DDPG, 10-episode moving average")
    plt.plot(
        moving_average(reinforce_returns),
        label="REINFORCE, 10-episode moving average",
    )
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Pendulum-v1: DDPG vs REINFORCE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_observations(
    path: Path, episodes: int, ddpg_returns: List[float], reinforce_returns: List[float]
) -> None:
    ddpg_last = float(np.mean(ddpg_returns[-10:]))
    reinforce_last = float(np.mean(reinforce_returns[-10:]))
    path.write_text(
        "\n".join(
            [
                "Level 3 Observations - DDPG",
                "",
                "Environment: Pendulum-v1",
                f"Episodes per method: {episodes}",
                "",
                "DDPG implementation:",
                "DDPG uses an actor network for deterministic continuous actions and a critic network for Q(s, a).",
                "The critic is trained with a Bellman target y = r + gamma * Q_target(s', actor_target(s')).",
                "The actor is trained to maximize the critic value, implemented as minimizing -Q(s, actor(s)).",
                "Target actor and critic networks are updated slowly with Polyak averaging.",
                "A replay buffer is used so transitions can be reused for many gradient updates.",
                "",
                "REINFORCE baseline:",
                "The baseline uses a Gaussian policy for continuous actions and updates once per episode.",
                "It directly uses sampled Monte Carlo returns, so the gradient estimate has high variance.",
                "",
                "Stability interpretation:",
                "DDPG is more stable than the original deterministic policy gradient idea because it borrows two important ideas from DQN: replay buffer sampling and target networks.",
                "The replay buffer reduces correlation between updates, and target networks make the Bellman target move more slowly.",
                "",
                "Experiment result:",
                f"Mean return over the last 10 DDPG episodes: {ddpg_last:.1f}",
                f"Mean return over the last 10 REINFORCE episodes: {reinforce_last:.1f}",
                "On Pendulum-v1, higher return is better because rewards are negative costs.",
                "The plot and CSV files in this directory contain the raw comparison curves.",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="rl_exercises/week_5/output_l3")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ddpg_env = gym.make(args.env_name)
    reinforce_env = gym.make(args.env_name)

    ddpg = DDPGAgent(
        env=ddpg_env,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.005,
        buffer_capacity=100000,
        batch_size=64,
        exploration_noise=0.1,
        seed=args.seed,
    )
    reinforce = ContinuousReinforceAgent(
        env=reinforce_env,
        lr=1e-3,
        gamma=0.99,
        seed=args.seed,
    )

    ddpg_returns = train_ddpg(ddpg, args.episodes, args.max_steps)
    reinforce_returns = train_reinforce(reinforce, args.episodes, args.max_steps)

    save_csv(output_dir / "ddpg_pendulum_returns.csv", ddpg_returns)
    save_csv(output_dir / "reinforce_pendulum_returns.csv", reinforce_returns)
    save_plot(
        output_dir / "ddpg_vs_reinforce_pendulum.png", ddpg_returns, reinforce_returns
    )
    write_observations(
        output_dir / "observations_l3.txt",
        args.episodes,
        ddpg_returns,
        reinforce_returns,
    )
    print(f"Wrote Level 3 outputs to {output_dir}")


if __name__ == "__main__":
    main()
