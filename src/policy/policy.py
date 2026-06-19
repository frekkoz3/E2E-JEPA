"""
E2E-Jepa

Team Rocket:
@capsia37
@enricosavorgnan
@frekkoz3

The file implements a class Policy that takes care of
- evaluating q-values given a state,
- selecting the best action,
- applying epsilon-greedy scheduling
- more
"""
import os
import sys
import yaml
import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from src.policy.algorithms import *
from src.game.snake import *
from src.policy.losses import *


class EpsilonGreedy:
    """Epsilon-greedy policy"""

    def __init__(self,
                 epsilon_start : float | int = 1.,
                 coeff : float | int = 0.999,
                 epsilon_end : float | int = 0.,
                 **kwargs):
        """
        Applies epsilon-greedy scheduling to balance exploration and exploitation during training.

        Parameters
        ----------
        epsilon_start : float | int, optional
            Initial value of epsilon (the exploration rate), by default 1.0
        coeff : float | int, optional
            Decay coefficient for epsilon, by default 0.999
        epsilon_end : float | int, optional
            Minimum value of epsilon, by default 0.0
            When reached epsilon will stop decaying and will remain constant at this value.
        """
        self.eps = epsilon_start
        self.coeff = coeff
        self.limit = epsilon_end


    def step(self):
        """Updates the value of epsilon according to the decay coefficient and the minimum limit."""
        if self.eps > self.limit:
            self.eps *= self.coeff
        else:
            self.eps = self.limit
        return self.eps



class Policy:
    """Policy class that combines a network architecture and an epsilon-greedy strategy for action selection."""

    def __init__(self,
                 environment : str = "Snake",
                 network : str = "DQN",
                 epsilon_strategy : str = "EpsilonGreedy",
                 optimizer : str = "Adam",
                 loss : str = "MSELoss",
                 n_iters : int = 1,
                 device : str = "cpu",
                 horizon : int = 1,
                 reward_discount : float | int = 0.99,
                 **kwargs):
        """
        Initializes the policy with a specified network architecture and epsilon-greedy strategy.

        Parameters
        ----------
        network : str, optional
            The name of the network architecture to use for approximating Q-values, by default "DQN"
        epsilon_strategy : str, optional
            The name of the epsilon-greedy strategy to use for exploration, by default "EpsilonGreedy"
        **kwargs:
            Additional keyword arguments to pass to the network and epsilon strategy constructors.
        """
        # Environment attributes
        self.environment = None

        # Policy attributes
        self.network = None
        self.epsilon_strategy = None
        self.horizon = horizon
        self.reward_discount = reward_discount

        # Training attributes
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.n_iterations = n_iters
        self.device = device

        # Set architecture
        self.set_environment(environment, **kwargs)
        self.set_network(network, **kwargs)
        if kwargs.get("load_network_path"):
            self.load_network(path = str(kwargs.get("load_network_path")))
        if self.device != "cpu":
            self.network.to(self.device)
        self.set_epsilon_strategy(epsilon_strategy, **kwargs)
        self.set_optimizer(optimizer, **kwargs)
        self.set_loss(loss, **kwargs)

        self.q_values = None


    def set_environment(self, environment : str, **kwargs):
        """Sets the environment for the policy."""
        maps = {
            "Snake": SnakeEnv,
        }

        # Assertions
        assert environment in maps, (f"Environment '{environment}' is not supported. "
                                     f"Supported environments are: {list(maps.keys())}.")
        assert "render_mode" in kwargs, f"Environment requires 'render_mode' parameter."
        assert "difficulty" in kwargs, f"Environment requires 'difficulty' parameter."
        assert "observation_type" in kwargs, f"Environment requires 'observation_type' parameter."

        self.environment = maps[environment](**kwargs)


    def set_network(self, network : str, **kwargs):
        """Sets the network architecture for approximating Q-values."""
        maps = {
            "DQN": DQN,
            "ConvDQN": ConvDQN,
            "AttentionDQN": AttentionDQN,
            "ConvPPO": ConvPPO,
            "AttentionPPO": AttentionPPO
        }

        # Assertions
        assert network in maps, f"Network '{network}' is not supported. Supported networks are: {list(maps.keys())}."
        if network == "DQN":
            assert "input_dim" in kwargs and "output_dim" in kwargs, \
                f"DQN requires 'input_dim' and 'output_dim' parameters."
            assert "num_hidden_layer" in kwargs and "dim_hidden_layer" in kwargs, \
                f"DQN requires 'num_hidden_layer' and 'dim_hidden_layer' parameters."
        elif network == "ConvDQN" or network == "ConvPPO":
            assert "input_dim" in kwargs and "output_dim" in kwargs, \
                f"ConvDQN requires 'input_dim' and 'output_dim' parameters."
            assert "num_conv_layer" in kwargs and "conv_layer_params" in kwargs, \
                f"ConvDQN requires 'num_conv_layer' and 'conv_layer_params' parameters."
            assert "num_fc_layer" in kwargs and "dim_fc_layer" in kwargs, \
                f"ConvDQN requires 'num_fc_layer' and 'dim_fc_layer' parameters."
        elif network == "AttentionDQN" or network == "AttentionPPO":
            assert "input_dim" in kwargs and "output_dim" in kwargs, \
                f"AttentionDQN requires 'input_dim' and 'output_dim' parameters."
            assert "num_attention_layer" in kwargs and "attention_layer_params" in kwargs, \
                f"AttentionDQN requires 'num_attention_layer' and 'attention_layer_params' parameters."
            assert "num_fc_layer" in kwargs and "dim_fc_layer" in kwargs, \
                f"AttentionDQN requires 'num_fc_layer' and 'dim_fc_layer' parameters."

        self.network = maps[network](**kwargs)


    def set_epsilon_strategy(self, epsilon_strategy : str, **kwargs):
        """Sets the epsilon-greedy strategy for exploration."""
        maps = {
            "EpsilonGreedy": EpsilonGreedy,
            "EpsilonConstant": lambda x,**kwargs: EpsilonGreedy(epsilon_start = kwargs.get("epsilon_start", 1.0),
                                                              coeff = 1.0,
                                                              epsilon_end = kwargs.get("epsilon_end", 0.0))
        }

        # Assertions
        assert epsilon_strategy in maps, \
            f"Epsilon strategy '{epsilon_strategy}' is not supported. Supported strategies are: {list(maps.keys())}."
        assert "epsilon_start" in kwargs and "coeff" in kwargs and "epsilon_end" in kwargs, \
            f"EpsilonGreedy requires 'epsilon_start', 'coeff', and 'epsilon_end' parameters."

        self.epsilon_strategy = maps[epsilon_strategy](**kwargs)


    def set_optimizer(self, optimizer : str, **kwargs):
        """Sets the optimizer for training the network."""
        maps = {
            "Adam": torch.optim.Adam,
            "SGD": torch.optim.SGD
        }

        # Assertions
        assert optimizer in maps, \
            f"Optimizer '{optimizer}' is not supported. Supported optimizers are: {list(maps.keys())}."
        assert "lr_init" in kwargs, f"Optimizer requires 'lr' parameter."

        self.optimizer = maps[optimizer](self.network.parameters(), lr=kwargs["lr_init"])
        if "lr_scheduler" in kwargs:
            scheduler = kwargs.get("lr_scheduler")
            if scheduler == "StepLR":
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                 step_size=kwargs.get("lr_step_size", 10),
                                                                 gamma=kwargs.get("lr_gamma", 0.1))
            elif scheduler == "ExponentialLR":
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer,
                                                                        gamma=kwargs.get("lr_gamma", 0.9))
            else:
                raise ValueError(f"Scheduler '{scheduler}' is not supported. "
                                 f"Supported schedulers are: ['StepLR', 'ExponentialLR'].")


    def set_loss(self, loss : str, **kwargs):
        """Sets the loss function for training the network."""
        maps = {
            "MSELoss": torch.nn.MSELoss,
            "PPOLoss": PPOLoss,
            "A2CLoss": A2CLoss
        }

        # Assertions
        assert loss in maps, \
            f"Loss function '{loss}' is not supported. Supported loss functions are: {list(maps.keys())}."

        self.loss = maps[loss](**kwargs)


    def get_reward(self, state, action) -> float:
        """
        TODO: Consider to remove this method if not needed,
            since in practice the reward is set by the environment and not computed by the policy.

        Returns the reward for a given couple (state, action).
        Here, we consider as a reward the Q-Value of the selected action with addiction of white noise.
        In practice, the reward is set by the environment.

        Parameters
        ----------
        state : torch.Tensor
            The current state of the environment.
        action : int
            The action taken in the current state.

        Returns
        -------
        reward : float
            The reward associated with the (state, action) pair.
        """
        with torch.no_grad():
            q_values = self.network(state)
            reward = q_values[0, action].item() + np.random.normal(0, 0.1)  # Adding white noise
        return reward


    def get_action(self, state, greedy : bool = False):
        """
        Selects an action based on the current policy and epsilon-greedy strategy.

        Parameters
        ----------
        state : torch.Tensor
            The current state of the environment.
        greedy : bool, optional, default False
            If True, selects the action with the highest Q-value (exploitation).
            If False, applies epsilon-greedy strategy to select an action.
        """
        state_tensor = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32, device=self.device)
        if state_tensor.dim() == 2: # Add sequence/channel dim if missing
            state_tensor = state_tensor.unsqueeze(0)

        is_policy_based = hasattr(self.network, 'actor_head')

        # --- PPO / A2C (On-Policy) ---
        if is_policy_based:
            with torch.no_grad():
                dist, value = self.network(state_tensor)
                if greedy:
                    action = torch.argmax(dist.logits, dim=-1)
                else:
                    action = dist.sample()
            return action.item(), (dist.log_prob(action), value.squeeze(-1))

        # --- DQN (Off-Policy) ---
        else:
            if not greedy and np.random.rand() < self.epsilon_strategy.eps:
                action = np.random.randint(0, self.network.output.out_features)
            else:
                with torch.no_grad():
                    q_values = self.network(state_tensor)
                    action = torch.argmax(q_values, dim=-1).item()
            return action, (None, None)


    def save_network(self, path : str):
        """Saves the model parameters to the specified path."""
        torch.save(self.network.state_dict(), path)


    def load_network(self, path : str):
        """Loads the model parameters from the specified path."""
        self.network.load_state_dict(torch.load(path))


    def forward(self, state):
        """
        TODO: Consider to remove this method if not needed
        Simply applies the forward() method of the network.
        """
        self.q_values = self.network(state)
        return self.q_values


    def train(self, batch_or_state):
        """
        Unified training entry point.
        - If DQN: Expects a batch tuple (states, actions, rewards, next_states, dones).
        - If PPO/A2C: Expects a single starting state to begin on-policy rollout.
        """
        is_policy_based = hasattr(self.network, 'actor_head')

        self.network.train()
        if is_policy_based:
            return self._train_on_policy(start_state=batch_or_state)
        else:
            return self._train_off_policy(batch=batch_or_state)

    def _train_off_policy(self, batch):
        """DQN Optimization via Bellman Equation and Replay Buffer."""
        states, actions, rewards, next_states, dones = batch

        # 1. Compute Current Q-values
        q_values = self.network(states)
        current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. Compute Target Q-values (Bootstrap)
        with torch.no_grad():
            # NOTE: For stability, use a cloned self.target_network.
            # Falling back to self.network here for structural demonstration.
            next_q_values = self.network(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            target_q = rewards + (self.reward_discount * max_next_q * (1 - dones))

        # 3. Optimize
        td_loss, _ = self.loss(current_q, target_q)

        self.optimizer.zero_grad()
        td_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        # Update Epsilon
        self.epsilon_strategy.step()

        return {"loss": td_loss.item(), "mean_q": current_q.mean().item()}


    def _train_on_policy(self, start_state):
        """PPO/A2C Optimization via Trajectory Rollout and GAE."""
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []
        state = start_state

        # 1. Trajectory Rollout (Data Collection)
        for _ in range(self.horizon):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                dist, value = self.network(state_tensor)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            next_state, reward, done, truncated, _ = self.environment.step(action.item())

            states.append(state_tensor)
            actions.append(action)
            rewards.append(reward)
            values.append(value.squeeze(-1))
            log_probs.append(log_prob)
            dones.append(done or truncated)

            state = next_state
            if done or truncated:
                state, _ = self.environment.reset()

        # Concatenate history
        states = torch.cat(states)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        values = torch.cat(values)
        log_probs = torch.cat(log_probs)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # 2. Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            last_state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, last_value = self.network(last_state_tensor)
            last_value = last_value.squeeze(-1)

        returns = torch.zeros_like(rewards)
        gae = 0
        lam = 0.95 # GAE lambda parameter
        for t in reversed(range(self.horizon)):
            if t == self.horizon - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = last_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t+1]

            delta = rewards[t] + self.reward_discount * next_val * next_non_terminal - values[t]
            gae = delta + self.reward_discount * lam * next_non_terminal * gae
            returns[t] = gae + values[t]

        # 3. Optimization Epochs
        epochs = self.n_iterations if isinstance(self.loss, PPOLoss) else 1 # PPO reuses data; A2C strictly steps once
        total_loss_history = []

        for _ in range(epochs):
            dist, new_values = self.network(states)

            # Branch based on loss function signature
            if isinstance(self.loss, PPOLoss):
                loss = self.loss.compute(dist, new_values, actions, returns, log_probs.detach())
            else:
                loss = self.loss.compute(dist, new_values, actions, returns)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()
            total_loss_history.append(loss.item())

        if self.scheduler:
            self.scheduler.step() # Note: Step scheduler per rollout, not per environment frame

        return {"loss": np.mean(total_loss_history), "mean_reward": rewards.mean().item()}



if __name__ == "__main__":

    input_dim = 16
    output_dim = 4

    # --- DQN SETUP ---
    # network = "DQN"
    # num_hidden_layer = 2
    # dim_hidden_layer = [64, 64]

    # --- ConvDQN SETUP ---
    # network = "ConvDQN"
    # num_conv_layer = 2
    # conv_layer_params = [
    #     {"out_channels": 8, "kernel_size": 3, "stride": 1},
    #     {"out_channels": 16, "kernel_size": 3, "stride": 1}
    # ]
    # num_fc_layer = 2
    # dim_fc_layer = [192, 32]

    # --- AttentionDQN SETUP ---
    network = "AttentionDQN"
    num_attention_layer = 2
    attention_layer_params = [
        {"num_heads": 2, "dim_feedforward": 32},
        {"num_heads": 4, "dim_feedforward": 64}
    ]
    num_fc_layer = 2
    dim_fc_layer = [16, 32]

    epsilon_strategy = "EpsilonGreedy"
    epsilon_start = 1.0
    coeff = 0.99
    epsilon_end = 0.01

    # --- DQN Policy
    # policy = Policy(network=network,
    #                 epsilon_strategy=epsilon_strategy,
    #                 input_dim=input_dim,
    #                 output_dim=output_dim,
    #                 num_hidden_layer=num_hidden_layer,
    #                 dim_hidden_layer=dim_hidden_layer,
    #                 epsilon_start=epsilon_start,
    #                 coeff=coeff,
    #                 epsilon_end=epsilon_end)

    # --- ConvDQN Policy
    # policy = Policy(network=network,
    #                 epsilon_strategy=epsilon_strategy,
    #                 input_dim=input_dim,
    #                 output_dim=output_dim,
    #                 num_conv_layer=num_conv_layer,
    #                 conv_layer_params=conv_layer_params,
    #                 num_fc_layer=num_fc_layer,
    #                 dim_fc_layer=dim_fc_layer,
    #                 epsilon_start=epsilon_start,
    #                 coeff=coeff,
    #                 epsilon_end=epsilon_end)

    # --- AttentionDQN Policy
    policy = Policy(network=network,
                    epsilon_strategy=epsilon_strategy,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    num_attention_layer=num_attention_layer,
                    attention_layer_params=attention_layer_params,
                    num_fc_layer=num_fc_layer,
                    dim_fc_layer=dim_fc_layer,
                    epsilon_start=epsilon_start,
                    coeff=coeff,
                    epsilon_end=epsilon_end)

    state = torch.randn(1, 1, input_dim)  # Example state
    q_values = policy.train(state)
    action = policy.get_action(state, greedy = True)
    print(f"Q-values: {q_values}, Selected action: {action}")
