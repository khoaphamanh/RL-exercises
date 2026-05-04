from __future__ import annotations

from typing import Any, DefaultDict

from collections import defaultdict

import gymnasium as gym
import numpy as np
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_3.epsilon_greedy_policy import EpsilonGreedyPolicy

State = Any


class TDLambdaAgent(AbstractAgent):
    """SARSA(lambda) agent with accumulating eligibility traces."""

    def __init__(
        self,
        env: gym.Env,
        policy: EpsilonGreedyPolicy,
        alpha: float = 0.5,
        gamma: float = 1.0,
        lambda_: float = 0.8,
    ) -> None:
        assert alpha > 0, "Learning rate has to be greater than 0"
        assert 0 <= gamma <= 1, "Gamma should be in [0, 1]"
        assert 0 <= lambda_ <= 1, "Lambda should be in [0, 1]"

        self.env = env
        self.policy = policy
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.n_actions = env.action_space.n
        self.Q: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )
        self.E: DefaultDict[Any, np.ndarray] = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float)
        )

    def predict_action(
        self, state: np.array, info: dict = {}, evaluate: bool = False
    ) -> Any:  # type: ignore # noqa
        """Predict the action for a given state."""
        return self.policy(self.Q, state, evaluate=evaluate), info

    def save(self, path: str) -> Any:  # type: ignore
        """Save the Q table."""
        np.save(path, dict(self.Q))  # type: ignore

    def load(self, path) -> Any:  # type: ignore
        """Load the Q table."""
        loaded_q = np.load(path, allow_pickle=True).item()
        self.Q = defaultdict(
            lambda: np.zeros(self.n_actions, dtype=float),
            loaded_q,
        )
        self.reset_traces()

    def reset_traces(self) -> None:
        """Clear eligibility traces at episode boundaries."""
        self.E.clear()

    def update_agent(self, batch) -> float:  # type: ignore
        """Update all eligible state-action pairs from the latest transition."""
        state, action, reward, next_state, done, _ = batch[0]
        next_action = 0 if done else self.policy(self.Q, next_state)
        return self.SARSA_lambda(state, action, reward, next_state, next_action, done)

    def SARSA_lambda(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_action: int,
        done: bool,
    ) -> float:
        """Perform an accumulating-trace SARSA(lambda) update."""
        q_value = self.Q[state][action]
        next_q_value = 0.0 if done else self.Q[next_state][next_action]
        td_error = reward + self.gamma * next_q_value - q_value

        self.E[state][action] += 1.0
        for trace_state, trace_values in list(self.E.items()):
            self.Q[trace_state] += self.alpha * td_error * trace_values
            if done:
                trace_values.fill(0.0)
            else:
                trace_values *= self.gamma * self.lambda_

        if done:
            self.reset_traces()

        return self.Q[state][action]


class RandomWalkTDLambdaEnv(gym.Env):
    """Seven-state random walk example for TD(lambda)-style experiments."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self, n_states: int = 7, horizon: int = 20, seed: int | None = None
    ) -> None:
        assert n_states >= 3, "Random walk needs at least two terminals and one start"
        self.n_states = n_states
        self.horizon = horizon
        self.current_steps = 0
        self.start_state = n_states // 2
        self.position = self.start_state
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Discrete(n_states)
        self.action_space = gym.spaces.Discrete(2)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_steps = 0
        self.position = self.start_state
        return self.position, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict[str, Any]]:
        action = int(action)
        if not self.action_space.contains(action):
            raise RuntimeError(f"{action} is not a valid action (needs to be 0 or 1)")

        self.current_steps += 1
        self.position += -1 if action == 0 else 1
        self.position = int(np.clip(self.position, 0, self.n_states - 1))
        terminated = self.position in (0, self.n_states - 1)
        truncated = self.current_steps >= self.horizon
        reward = 1.0 if self.position == self.n_states - 1 else 0.0
        return self.position, reward, terminated, truncated, {}

    def render(self, mode: str = "human") -> None:
        print(f"[RandomWalkTDLambda] pos={self.position}")
