"""
Deep Q-Learning implementation.
"""

from typing import Any, Dict, List, Tuple

from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import ReplayBuffer
from rl_exercises.week_4.networks import QNetwork


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent(AbstractAgent):
    """
    Deep Q‐Learning agent with ε‐greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
    ) -> None:
        """
        Initialize replay buffer, Q‐networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini‐batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target‐network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.env = env
        set_seed(env, seed)

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # main Q‐network and frozen target
        self.q = QNetwork(obs_dim, n_actions)
        self.target_q = QNetwork(obs_dim, n_actions)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0  # for ε decay and target sync
        self.training_curve: List[Tuple[int, float]] = []

    def epsilon(self) -> float:
        """
        Compute current ε by exponential decay.

        Returns
        -------
        float
            Exploration rate.
        """
        # ε = ε_final + (ε_start - ε_final) * exp(-total_steps / ε_decay)
        # Currently, it is constant and returns the starting value ε
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * np.exp(
            -self.total_steps / self.epsilon_decay
        )

    def predict_action(
        self, state: np.ndarray, info: Dict[str, Any] = {}, evaluate: bool = False
    ) -> Tuple[int, Dict]:
        """
        Choose action via ε‐greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(Q).

        Returns
        -------
        action : int
        info_out : dict
            Empty dict (compatible with interface).
        """
        if evaluate:
            # TODO: select purely greedy action from Q(s)
            # purely greedy
            t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                qvals = self.q(t)
            action = int(torch.argmax(qvals, dim=1).item())
        else:
            # ε-greedy
            if np.random.rand() < self.epsilon():
                # TODO: sample random action
                action = int(self.env.action_space.sample())
            else:
                # TODO: select purely greedy action from Q(s)
                action = int(
                    self.q(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    .argmax(dim=1)
                    .item()
                )

        return action

    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            MSE loss value.
        """
        # unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)
        s = torch.tensor(np.array(states), dtype=torch.float32)
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)
        r = torch.tensor(np.array(rewards), dtype=torch.float32)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)
        mask = torch.tensor(np.array(dones), dtype=torch.float32)

        # current Q estimates for taken actions
        # TODO: pass batched states through self.q and gather Q(s,a)
        pred = self.q(s).gather(1, a).squeeze(1)

        # TODO: compute TD target with frozen network
        with torch.no_grad():
            max_next_q = self.target_q(s_next).max(dim=1).values
            target = r + self.gamma * (1.0 - mask) * max_next_q

        loss = nn.MSELoss()(pred, target)

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # occasionally sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1
        return float(loss.item())

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []

        # enviroment steps
        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # store and step
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                # TODO: sample batch from replay buffer
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                ep_reward = 0.0
                # logging
                if len(recent_rewards) % 10 == 0:
                    # TODO: compute avg over last eval_interval episodes and print
                    avg = np.mean(recent_rewards[-10:])
                    self.training_curve.append((frame, float(avg)))
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        print("Training complete.")


def save_training_curve(
    training_curve: List[Tuple[int, float]],
    output_dir: str | Path,
    filename: str = "learning_curve",
    title: str = "DQN training curve",
) -> None:
    if not training_curve:
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / f"{filename}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("frame,mean_reward\n")
        for frame, mean_reward in training_curve:
            f.write(f"{frame},{mean_reward}\n")

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    frames, mean_rewards = zip(*training_curve)
    plt.figure(figsize=(8, 5))
    plt.plot(frames, mean_rewards)
    plt.xlabel("Frames")
    plt.ylabel("Mean reward")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path / f"{filename}.png", dpi=150)
    plt.close()


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)

    # 2) TODO: map config → agent kwargs
    agent_kwargs = dict(
        buffer_capacity=cfg.agent.buffer_capacity,
        batch_size=cfg.agent.batch_size,
        lr=cfg.agent.learning_rate,
        gamma=cfg.agent.gamma,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        epsilon_decay=cfg.agent.epsilon_decay,
        target_update_freq=cfg.agent.target_update_freq,
        seed=cfg.seed,
    )

    # 3) TODO:instantiate & train
    agent = DQNAgent(env, **agent_kwargs)
    agent.train(cfg.train.num_frames, cfg.train.eval_interval)

    experiment_name = (
        f"dqn_buffer{cfg.agent.buffer_capacity}"
        f"_batch{cfg.agent.batch_size}"
        f"_lr{cfg.agent.learning_rate}"
        f"_gamma{cfg.agent.gamma}"
        f"_epsdecay{cfg.agent.epsilon_decay}"
        f"_target{cfg.agent.target_update_freq}"
        f"_seed{cfg.seed}"
    )
    plot_title = (
        "DQN training curve\n"
        f"buffer={cfg.agent.buffer_capacity}, "
        f"batch={cfg.agent.batch_size}, "
        f"lr={cfg.agent.learning_rate}, "
        f"gamma={cfg.agent.gamma}\n"
        f"eps_decay={cfg.agent.epsilon_decay}, "
        f"target_update={cfg.agent.target_update_freq}, "
        f"seed={cfg.seed}"
    )
    output_dir = Path(get_original_cwd()) / "rl_exercises" / "week_4" / "output"
    save_training_curve(agent.training_curve, output_dir, experiment_name, plot_title)


if __name__ == "__main__":
    main()


"""
# environment cart pol v1
state = [x, x_velocity, angle, angular_velocity] obs_dim = 4
action = [left, right] n_actions = 2


How state updates happen

The simulator integrates these equations over time.

At every timestep:

1. Position update

x(t+1) = x(t) + τ * x_velocity(t)

2. Velocity update

x_velocity(t+1) = x_velocity(t) + τ * x_acceleration(t)

3. Angle update

theta(t+1) = theta(t) + τ * angular_velocity(t)

4. Angular velocity update

angular_velocity(t+1) =
    angular_velocity(t) + τ * angular_acceleration(t)

Where:

τ = timestep (usually 0.02 seconds)


# case one model DQN
Network trained by gradient descent, updated every step
input s,
output Q (s,:), 
choose action a and Q(s,a), 
enviroment returns r, s', transition (s, a, r, s'),
feed to net s', get Q(s',:),
calculate TD target y = Q (s,a) = r + γ max_a' Q'(s', a')  (from bellman optimality equation)
caöculate loss = MSE(Q(s,a), y)


# model 
online_network Q_o: trained by gradient descent, updated every step
target_network Q_t: frozen copy of Q, updated every C steps

step 1: init Q_o, Q_t same weights, init buffers
step 2: choose random action a, observe r, s', store (s, a, r, s') in buffer
step 3: sample random batch from buffer
step 4: forward pass batch through Q_o(s,a) to get Q(s,a)
step 5: choose action a using greedy, get s',r from enviroment
step 6: feed s' to Q_o to get Q_o'(s',:), choose action a' and get Q_o'(s', a')
step 7: feed s' to Q_t to get Q_t'(s',:), choose action a' and get Q_t'(s', a')
step 8: calculate TD target y = r + γ Q_t'(s', a')
step 9: calculate loss = MSE(Q_o(s,a), y)
step 10: gradient descent on Q_o
step 11: every C steps, update Q_t = Q_o
"""
