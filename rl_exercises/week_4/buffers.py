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
