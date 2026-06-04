from typing import Any, Dict, List, Tuple

import numpy as np
from rl_exercises.agent import AbstractBuffer


class ReplayBuffer(AbstractBuffer):
    """
    Simple FIFO replay buffer.

    Stores tuples of (state, action, reward, next_state, done, info),
    and evicts the oldest when capacity is exceeded.
    """

    def __init__(self, capacity: int) -> None:
        """
        Parameters
        ----------
        capacity : int
            Maximum number of transitions to store.
        """
        super().__init__()
        self.capacity = capacity
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.next_states: List[np.ndarray] = []
        self.dones: List[bool] = []
        self.infos: List[Dict] = []

    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        """
        Add a single transition to the buffer.

        If the buffer is full, the oldest transition is removed.

        Parameters
        ----------
        state : np.ndarray
            Observation before action.
        action : int or float
            Action taken.
        reward : float
            Reward received.
        next_state : np.ndarray
            Observation after action.
        done : bool
            Whether episode terminated/truncated.
        info : dict
            Gym info dict (can store extras).
        """

        # remove oldest transition if buffer is full laut first in first out (FIFO) policy
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)
            self.infos.pop(0)

        # append state, action, reward, next_state, done, info to their respective lists
        self.states.append(state)
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_states.append(next_state)
        self.dones.append(bool(done))
        self.infos.append(info)

    def sample(
        self, batch_size: int = 32
    ) -> List[Tuple[Any, Any, float, Any, bool, Dict]]:
        """
        Uniformly sample a batch of transitions.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.

        Returns
        -------
        List of transitions as (state, action, reward, next_state, done, info).
        """
        # TODO: randomly choose `batch_size` unique indices from [0, len(self.states))
        idxs = np.random.choice(
            len(self.states), size=min(batch_size, len(self.states)), replace=False
        )
        return [
            (
                self.states[i],
                self.actions[i],
                self.rewards[i],
                self.next_states[i],
                self.dones[i],
                self.infos[i],
            )
            for i in idxs
        ]

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self.states)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Replay buffer with proportional prioritized sampling.

    Sampling probability is P(i) = p_i^alpha / sum_k p_k^alpha. Each sampled
    transition carries its buffer index and importance-sampling weight in info,
    so agents can update priorities after computing TD errors.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.priorities: List[float] = []

    def add(
        self,
        state: np.ndarray,
        action: int | float,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict,
    ) -> None:
        if len(self.states) >= self.capacity:
            self.priorities.pop(0)

        super().add(state, action, reward, next_state, done, info)
        max_priority = max(self.priorities, default=1.0)
        self.priorities.append(max_priority)

    def sample(
        self, batch_size: int = 32
    ) -> List[Tuple[Any, Any, float, Any, bool, Dict]]:
        size = len(self.states)
        sample_size = min(batch_size, size)
        priorities = np.asarray(self.priorities, dtype=np.float64)
        scaled_priorities = priorities**self.alpha
        probs = scaled_priorities / scaled_priorities.sum()
        idxs = np.random.choice(size, size=sample_size, replace=False, p=probs)

        weights = (size * probs[idxs]) ** (-self.beta)
        weights /= weights.max()

        batch = []
        for idx, weight in zip(idxs, weights):
            info = dict(self.infos[idx])
            info["_per_index"] = int(idx)
            info["_per_weight"] = float(weight)
            batch.append(
                (
                    self.states[idx],
                    self.actions[idx],
                    self.rewards[idx],
                    self.next_states[idx],
                    self.dones[idx],
                    info,
                )
            )
        return batch

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[int(idx)] = float(priority) + self.eps


# note
"""
transition = (state, action, reward, next_state, done, info)
capacity = max number of transitions to store

#steps to steps
take one action in environment
get one new transition
add it to replay buffer
sample random batch of 32 from replay buffer
train once
repeat

# example 
step 1: buffer has 1 transition, no training yet
step 2: buffer has 2 transitions, no training yet
...
step 32: buffer has 32 transitions, sample 32, train once
step 33: buffer has 33 transitions, sample 32, train once
step 34: buffer has 34 transitions, sample 32, train once
...
step 1000: buffer has 1000 transitions, sample 32, train once
step 1001: remove oldest, add newest, buffer still has 1000, sample 32, train once

"""
