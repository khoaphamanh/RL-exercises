"""
Soft Actor-Critic for continuous-action Gymnasium environments.

This implementation uses the standard SAC ingredients: a stochastic tanh-Gaussian
actor, twin Q-functions, target Q-functions, entropy regularization, and replay.
"""

from __future__ import annotations

from typing import Deque, Tuple

from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent


def set_seed(env: gym.Env, seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


@dataclass
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, capacity: int = 100000) -> None:
        self.data: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = (
            deque(maxlen=capacity)
        )

    def __len__(self) -> int:
        return len(self.data)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.data.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> ReplayBatch:
        indices = np.random.choice(len(self.data), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *(self.data[i] for i in indices)
        )
        return ReplayBatch(
            states=torch.as_tensor(np.array(states), dtype=torch.float32),
            actions=torch.as_tensor(np.array(actions), dtype=torch.float32),
            rewards=torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1),
            next_states=torch.as_tensor(np.array(next_states), dtype=torch.float32),
            dones=torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1),
        )


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = int(np.prod(observation_space.shape))
        self.action_dim = int(np.prod(action_space.shape))

        self.fc1 = nn.Linear(self.state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, self.action_dim)
        self.log_std = nn.Linear(hidden_size, self.action_dim)

        action_scale = (action_space.high - action_space.low) / 2.0
        action_bias = (action_space.high + action_space.low) / 2.0
        self.register_buffer(
            "action_scale", torch.as_tensor(action_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.as_tensor(action_bias, dtype=torch.float32)
        )

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if states.dim() == 1:
            states = states.unsqueeze(0)
        x = states.view(states.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), min=-20.0, max=2.0)
        return mean, log_std

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(states)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        squashed = torch.tanh(z)
        action = squashed * self.action_scale + self.action_bias
        log_prob = normal.log_prob(z) - torch.log(1.0 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            mean, _ = self.forward(state_t)
            action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action.squeeze(0).cpu().numpy()


class QNetwork(nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Box,
        action_space: gym.spaces.Box,
        hidden_size: int = 256,
    ) -> None:
        super().__init__()
        state_dim = int(np.prod(observation_space.shape))
        action_dim = int(np.prod(action_space.shape))
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        x = torch.cat([states.view(states.size(0), -1), actions], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class SACAgent(AbstractAgent):
    def __init__(
        self,
        env: gym.Env,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        batch_size: int = 256,
        buffer_size: int = 100000,
        learning_starts: int = 1000,
        updates_per_step: int = 1,
        seed: int = 0,
        hidden_size: int = 256,
    ) -> None:
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("SACAgent requires a continuous Box action space.")

        set_seed(env, seed)
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.updates_per_step = updates_per_step
        self.seed = seed

        self.policy = GaussianPolicy(
            env.observation_space, env.action_space, hidden_size
        )
        self.q1 = QNetwork(env.observation_space, env.action_space, hidden_size)
        self.q2 = QNetwork(env.observation_space, env.action_space, hidden_size)
        self.target_q1 = QNetwork(env.observation_space, env.action_space, hidden_size)
        self.target_q2 = QNetwork(env.observation_space, env.action_space, hidden_size)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr_critic)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr_critic)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def predict_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[np.ndarray, dict]:
        if evaluate:
            return self.policy.deterministic(state), {}
        state_t = torch.as_tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action, _ = self.policy.sample(state_t)
        return action.squeeze(0).cpu().numpy(), {}

    def update_agent(self) -> Tuple[float, float, float] | None:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_actions, next_logp = self.policy.sample(batch.next_states)
            target_q = torch.min(
                self.target_q1(batch.next_states, next_actions),
                self.target_q2(batch.next_states, next_actions),
            )
            target = batch.rewards + self.gamma * (1.0 - batch.dones) * (
                target_q - self.alpha * next_logp
            )

        q1_loss = F.mse_loss(self.q1(batch.states, batch.actions), target)
        q2_loss = F.mse_loss(self.q2(batch.states, batch.actions), target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        actions, logp = self.policy.sample(batch.states)
        min_q = torch.min(
            self.q1(batch.states, actions), self.q2(batch.states, actions)
        )
        policy_loss = (self.alpha * logp - min_q).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

        return policy_loss.item(), q1_loss.item(), q2_loss.item()

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - self.tau)
            target_param.data.add_(self.tau * source_param.data)

    def evaluate(self, eval_env: gym.Env, num_episodes: int = 5) -> Tuple[float, float]:
        returns = []
        for episode in range(num_episodes):
            state, _ = eval_env.reset(seed=self.seed + episode)
            done = False
            total_return = 0.0
            while not done:
                action, _ = self.predict_action(state, evaluate=True)
                state, reward, term, trunc, _ = eval_env.step(action)
                done = term or trunc
                total_return += float(reward)
            returns.append(total_return)
        return float(np.mean(returns)), float(np.std(returns))

    def train(
        self,
        total_steps: int,
        eval_interval: int = 5000,
        eval_episodes: int = 5,
    ) -> None:
        eval_env = gym.make(self.env.spec.id)
        state, _ = self.env.reset(seed=self.seed)

        for step in range(1, total_steps + 1):
            if step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action, _ = self.predict_action(state)

            next_state, reward, term, trunc, _ = self.env.step(action)
            done = term or trunc
            self.replay_buffer.add(state, action, float(reward), next_state, done)
            state = next_state

            if done:
                state, _ = self.env.reset()

            losses = None
            if step >= self.learning_starts:
                for _ in range(self.updates_per_step):
                    losses = self.update_agent()

            if step % eval_interval == 0:
                mean_r, std_r = self.evaluate(eval_env, eval_episodes)
                loss_text = ""
                if losses is not None:
                    loss_text = (
                        f" Policy Loss {losses[0]:.3f}"
                        f" Q1 Loss {losses[1]:.3f}"
                        f" Q2 Loss {losses[2]:.3f}"
                    )
                print(
                    f"[Eval ] Step {step:6d} AvgReturn {mean_r:7.1f}"
                    f" +/- {std_r:5.1f}{loss_text}"
                )

        print("Training complete.")


@hydra.main(config_path="../configs/agent/", config_name="sac", version_base="1.1")
def main(cfg: DictConfig) -> None:
    env = gym.make(cfg.env.name)
    agent = SACAgent(
        env,
        lr_actor=cfg.agent.lr_actor,
        lr_critic=cfg.agent.lr_critic,
        gamma=cfg.agent.gamma,
        tau=cfg.agent.tau,
        alpha=cfg.agent.alpha,
        batch_size=cfg.agent.batch_size,
        buffer_size=cfg.agent.buffer_size,
        learning_starts=cfg.agent.learning_starts,
        updates_per_step=cfg.agent.updates_per_step,
        seed=cfg.seed,
        hidden_size=cfg.agent.hidden_size,
    )
    agent.train(
        cfg.train.total_steps,
        cfg.train.eval_interval,
        cfg.train.eval_episodes,
    )


if __name__ == "__main__":
    main()
