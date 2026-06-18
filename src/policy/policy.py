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

from src.policy.networks import *


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



class Policy():
    """Policy class that combines a network architecture and an epsilon-greedy strategy for action selection."""

    def __init__(self,
                 network : str = "DQN",
                 epsilon_strategy : str = "EpsilonGreedy",
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
        self.network = None
        self.epsilon_strategy = None
        self.set_network(network, **kwargs)
        self.set_epsilon_strategy(epsilon_strategy, **kwargs)

        self.q_values = None


    def set_network(self, network : str, **kwargs):
        """Sets the network architecture for approximating Q-values."""
        maps = {
            "DQN": DQN,
            "ConvDQN": ConvDQN,
            "AttentionDQN": AttentionDQN
        }

        # Assertions
        assert network in maps, f"Network '{network}' is not supported. Supported networks are: {list(maps.keys())}."
        if network == "DQN":
            assert "input_dim" in kwargs and "output_dim" in kwargs, \
                f"DQN requires 'input_dim' and 'output_dim' parameters."
            assert "num_hidden_layer" in kwargs and "dim_hidden_layer" in kwargs, \
                f"DQN requires 'num_hidden_layer' and 'dim_hidden_layer' parameters."
        elif network == "ConvDQN":
            assert "input_dim" in kwargs and "output_dim" in kwargs, \
                f"ConvDQN requires 'input_dim' and 'output_dim' parameters."
            assert "num_conv_layer" in kwargs and "conv_layer_params" in kwargs, \
                f"ConvDQN requires 'num_conv_layer' and 'conv_layer_params' parameters."
            assert "num_fc_layer" in kwargs and "dim_fc_layer" in kwargs, \
                f"ConvDQN requires 'num_fc_layer' and 'dim_fc_layer' parameters."
        elif network == "AttentionDQN":
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
            "EpsilonGreedy": EpsilonGreedy
        }

        # Assertions
        assert epsilon_strategy in maps, \
            f"Epsilon strategy '{epsilon_strategy}' is not supported. Supported strategies are: {list(maps.keys())}."
        assert "epsilon_start" in kwargs and "coeff" in kwargs and "epsilon_end" in kwargs, \
            f"EpsilonGreedy requires 'epsilon_start', 'coeff', and 'epsilon_end' parameters."

        self.epsilon_strategy = maps[epsilon_strategy](**kwargs)


    def train(self, state):
        """Simply applies the forward() method of the network."""
        self.q_values = self.network(state)
        return self.q_values


    def get_action(self, state, greedy : bool = False):
        """Selects an action based on the current policy and epsilon-greedy strategy."""
        if np.random.rand() < self.epsilon_strategy.eps and not greedy:
            # Explore: select a random action
            action = np.random.randint(0, self.network.output.out_features)
        else:
            # Exploit: select the action with the highest Q-value
            with torch.no_grad():
                q_values = self.network(state)
                action = torch.argmax(q_values).item()

        return action


    def save_network(self, path : str):
        """Saves the model parameters to the specified path."""
        torch.save(self.network.state_dict(), path)


    def load_network(self, path : str):
        """Loads the model parameters from the specified path."""
        self.network.load_state_dict(torch.load(path))



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
