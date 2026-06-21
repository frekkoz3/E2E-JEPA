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
from collections import deque
import copy

import yaml
import argparse

import numpy as np
import torch

import gymnasium as gym

from src.policy.algorithms import *
from src.game.snake import *
from src.policy.losses import *
from src.utils.utils import *



class EpsilonGreedy:
    """Epsilon-greedy policy"""

    def __init__(self,
                 epsilon_start : float | int = 1.,
                 epsilon_coeff : float | int = 0.999,
                 epsilon_end : float | int = 0.,
                 **kwargs):
        """
        Applies epsilon-greedy scheduling to balance exploration and exploitation during training.

        Parameters
        ----------
        epsilon_start : float | int, optional
            Initial value of epsilon (the exploration rate), by default 1.0
        epsilon_coeff : float | int, optional
            Decay coefficient for epsilon, by default 0.999
        epsilon_end : float | int, optional
            Minimum value of epsilon, by default 0.0
            When reached epsilon will stop decaying and will remain constant at this value.
        """
        self.eps = epsilon_start
        self.coeff = epsilon_coeff
        self.limit = epsilon_end


    def step(self):
        """Updates the value of epsilon according to the decay coefficient and the minimum limit."""
        if self.eps > self.limit:
            self.eps *= self.coeff
        else:
            self.eps = self.limit
        return self.eps



class EpsilonConstant(EpsilonGreedy):
    """Constant epsilon-greedy policy"""

    def __init__(self, **kwargs):
        """
        Initializes a constant epsilon-greedy policy.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments (not used in this class).
        """
        super().__init__(epsilon_start=kwargs.get("epsilon_start", 1.0),
                         epsilon_coeff=1.0,
                         epsilon_end=kwargs.get("epsilon_end", 0.0))



class Policy:
    """Policy class that combines a network architecture and an epsilon-greedy strategy for action selection."""

    def __init__(self,
                 environment : str = "Snake",
                 network : str = "DQN",
                 epsilon_strategy : str = "EpsilonGreedy",
                 optimizer : str = "Adam",
                 loss : str = "MSELoss",
                 n_epochs : int = 1,
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
        self.mini_batch_size = kwargs.get("mini_batch_size", 8)
        self.reward_discount = reward_discount

        # Training attributes
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.n_epochs = n_epochs
        self.device = device

        # Set architecture
        self._set_environment(environment, **kwargs)
        self._set_network(network, **kwargs)
        if kwargs.get("model_path"):
            self.load_network(path = str(kwargs.get("model_path")))
        if self.device != "cpu":
            self.network.to(self.device)
        self._set_epsilon_strategy(epsilon_strategy, **kwargs)
        self._set_optimizer(optimizer, **kwargs)
        self._set_loss(loss, **kwargs)

        self.q_values = None

        if isinstance(self.network, (DQN, AttentionDQN, ConvDQN)):
            self.replay_buffer = deque(maxlen=50000)
            self.batch_size = kwargs.get("mini_batch_size", 64)
            self.target_network = copy.deepcopy(self.network)
            self.target_net_update_freq = kwargs.get("target_update_freq", 25)
            self.update_count = 0


    def _set_environment(self, environment : str, **kwargs):
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

        num_envs = kwargs.get("n_environments", 1)
        print(f"NUM ENVS: {num_envs}")
        self.environment = gym.vector.SyncVectorEnv([lambda: maps[environment](**kwargs) for _ in range(num_envs)]) if num_envs > 1 else maps[environment](**kwargs)


    def _set_network(self, network : str, **kwargs):
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

        assert "input_dim" in kwargs and "output_dim" in kwargs, \
            f"The network requires 'input_dim' and 'output_dim' parameters."
        if network == "DQN":
            assert "num_hidden_layer" in kwargs and "dim_hidden_layer" in kwargs, \
                f"DQN requires 'num_hidden_layer' and 'dim_hidden_layer' parameters."
        elif network == "ConvDQN" or network == "ConvPPO":
            assert "num_conv_layer" in kwargs and "conv_layer_params" in kwargs, \
                f"ConvDQN requires 'num_conv_layer' and 'conv_layer_params' parameters."
            assert "num_fc_layer" in kwargs and "dim_fc_layer" in kwargs, \
                f"ConvDQN requires 'num_fc_layer' and 'dim_fc_layer' parameters."
        elif network == "AttentionDQN" or network == "AttentionPPO":
            assert "num_attention_layer" in kwargs and "attention_layer_params" in kwargs, \
                f"AttentionDQN requires 'num_attention_layer' and 'attention_layer_params' parameters."
            assert "num_fc_layer" in kwargs and "dim_fc_layer" in kwargs, \
                f"AttentionDQN requires 'num_fc_layer' and 'dim_fc_layer' parameters."

        self.network = maps[network](**kwargs)


    def _set_epsilon_strategy(self, epsilon_strategy : str, **kwargs):
        """Sets the epsilon-greedy strategy for exploration."""
        maps = {
            "EpsilonGreedy": EpsilonGreedy,
            "EpsilonConstant": EpsilonConstant
        }

        # Assertions
        assert epsilon_strategy in maps, \
            f"Epsilon strategy '{epsilon_strategy}' is not supported. Supported strategies are: {list(maps.keys())}."
        assert "epsilon_start" in kwargs and "epsilon_coeff" in kwargs and "epsilon_end" in kwargs, \
            f"EpsilonGreedy requires 'epsilon_start', 'coeff', and 'epsilon_end' parameters."

        self.epsilon_strategy = maps[epsilon_strategy](**kwargs)


    def _set_optimizer(self, optimizer : str, **kwargs):
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


    def _set_loss(self, loss : str, **kwargs):
        """Sets the loss function for training the network."""
        maps = {
            "MSELoss": MSELoss,
            "PPOLoss": PPOLoss,
            "A2CLoss": A2CLoss
        }

        # Assertions
        assert loss in maps, \
            f"Loss function '{loss}' is not supported. Supported loss functions are: {list(maps.keys())}."

        print(f"Using loss function: {loss}")
        self.loss = maps[loss](**kwargs)


    def _format_state(self, state):
        """Formats the state from the environment to match the expected input shape of the network."""
        state_tensor = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32, device=self.device)

        # 1. Batched 2D grids: (num_envs, 20, 20) -> (num_envs, 400)
        if state_tensor.dim() == 3 and state_tensor.shape[1:] == (20, 20) and isinstance(self.network, (ConvPPO, AttentionPPO)):
            state_tensor = state_tensor.view(state_tensor.shape[0], -1)

        # 2. Flatten the 2D grid from the environment (20, 20) -> (1, 400)
        elif state_tensor.shape == (20, 20):
            state_tensor = state_tensor.flatten().unsqueeze(0)

        # 3. Add a Channel dimension specifically for Convolutional and Attention Networks -> (1, 1, 400)
        if not isinstance(self.network, DQN) and state_tensor.dim() == 2:
            state_tensor = state_tensor.unsqueeze(1)

        return state_tensor


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
        state = self._format_state(state)

        is_on_policy = isinstance(self.network, (ConvPPO, AttentionPPO))

        # --- PPO / A2C (On-Policy) ---
        if is_on_policy:
            with torch.no_grad():
                dist, value = self.network(state)
                if greedy:
                    action = torch.argmax(dist.logits, dim=-1)
                else:
                    action = dist.sample()
            return action.cpu().numpy(), (dist.log_prob(action), value.squeeze(-1))

        # --- DQN (Off-Policy) ---
        else:
            if not greedy and np.random.rand() < self.epsilon_strategy.eps:
                action = np.random.randint(0, self.network.output.out_features)
            else:
                with torch.no_grad():
                    q_values = self.network(state)
                    action = torch.argmax(q_values, dim=-1).item()
            return action, (None, None)


    def save_network(self, path : str):
        """Saves the model parameters to the specified path."""
        torch.save(self.network.state_dict(), path)


    def load_network(self, path : str):
        """Loads the model parameters from the specified path."""
        self.network.load_state_dict(torch.load(path, map_location=self.device))


    def forward(self, state):
        """
        TODO: Consider to remove this method if not needed
        Simply applies the forward() method of the network.
        """
        self.q_values = self.network(state)
        return self.q_values


    @torch.no_grad()
    def _get_rollout(self, state):
        """Runs a rollout in the environment, given an initial state"""
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

        # Rollout loop
        for step in range(self.horizon):
            state = self._format_state(state)

            action, (log_p, value) = self.get_action(state, greedy = False)

            next_state, reward, done, truncated, _ = self.environment.step(action)
            stop = done | truncated

            states.append(state)
            actions.append(torch.tensor(action, dtype=torch.int64, device=self.device))
            rewards.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
            values.append(value)
            dones.append(torch.tensor(stop, dtype=torch.float32, device=self.device))
            log_probs.append(log_p)

            # 4. Handle State Transition
            state = next_state

        states = torch.stack(states, dim=0)
        actions = torch.stack(actions, dim=0)
        rewards = torch.stack(rewards, dim=0)
        values = torch.stack(values, dim=0)
        dones = torch.stack(dones, dim=0)
        log_probs = torch.stack(log_probs, dim=0)

        last_state = self._format_state(state)
        _, last_value = self.network(last_state)
        last_value = last_value.squeeze(-1).squeeze(-1).unsqueeze(0)
        values = torch.cat( [values, last_value], dim=0)

        return states, actions, rewards, values, log_probs, dones


    def _get_buffer(self, state):
        """Computes a single trajectory buffer for Off-Policy DQN."""
        state = self._format_state(state)
        buffer = []
        is_terminal = False

        while not is_terminal:
            # Action is a scalar integer for a single environment
            action, _ = self.get_action(state, greedy=False)

            # Step returns scalars and standard Python booleans
            next_state, reward, done, truncated, _ = self.environment.step(action)
            is_terminal = done or truncated

            next_state_formatted = self._format_state(next_state)

            buffer.append((state, action, reward, next_state_formatted, is_terminal))
            state = next_state_formatted

        # Unpack the buffer
        states, actions, rewards, next_states, dones = zip(*buffer)

        # states and next_states are tuples of tensors of shape [1, 1, 400]
        # Concatenating them along dim=0 creates a valid 3D tensor -> [batch_size, 1, 400]
        states = torch.cat(states, dim=0)
        next_states = torch.cat(next_states, dim=0)

        # actions, rewards, and dones are tuples of standard Python scalars.
        # Convert them directly to 1D tensors.
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        batch = (states, actions, rewards, next_states, dones)

        return batch


    def train(self, state):
        """
        Training entry point.
        - If DQN: Expects a batch tuple (states, actions, rewards, next_states, dones).
        - If PPO/A2C: Expects a single starting state to begin on-policy rollout.
        """
        is_on_policy = isinstance(self.network, (AttentionPPO, ConvPPO))

        self.network.train()
        if is_on_policy:
            return self._train_on_policy(init_state=state)
        else:
            return self._train_off_policy(init_state=state)


    def _train_off_policy(self, init_state):
        import random

        # 1. Collect the trajectory and unpack it
        states, actions, rewards, next_states, dones = self._get_buffer(init_state)

        # 2. Push individual transitions to the persistent replay buffer
        for i in range(len(states)):
            self.replay_buffer.append((states[i], actions[i], rewards[i], next_states[i], dones[i]))

        # 3. Do not train if the buffer does not have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0, "mean_value": 0.0}

        # 4. Sample a random, decorrelated mini-batch
        batch = random.sample(self.replay_buffer, self.batch_size)

        b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)
        b_states = torch.stack(b_states).to(self.device)
        b_actions = torch.stack(b_actions).to(self.device)
        b_rewards = torch.stack(b_rewards).to(self.device)
        b_next_states = torch.stack(b_next_states).to(self.device)
        b_dones = torch.stack(b_dones).to(self.device)

        # 5. Compute Q-values and optimize
        q_values = self.network(b_states)
        current_q = q_values.gather(1, b_actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(b_next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            target_q = b_rewards + (self.reward_discount * max_next_q * (1 - b_dones))

        td_loss = self.loss(current_q, target_q)

        self.optimizer.zero_grad()
        td_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()
        self.epsilon_strategy.step()

        if (self.update_count + 1) % self.target_net_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.update_count += 1

        return {"loss": td_loss.item(), "mean_value": current_q.mean().item()}


    def _train_on_policy(self, init_state):
        """PPO/A2C Optimization via Trajectory Rollout and GAE."""

        # Compute rollout
        states, actions, rewards, values, log_probs, dones = self._get_rollout(state = init_state)

        # Compute advantages and returns
        returns, advantages = self.loss.compute_advantages(last_state_value = values[-1],
                                                          rewards = rewards,
                                                          values = values,
                                                          dones = dones)

        batch_size = states.shape[0] * states.shape[1]

        states = states.view(batch_size, *states.shape[2:])
        actions = actions.view(batch_size)
        old_log_probs = log_probs.view(batch_size).detach()
        mini_batch_size = self.mini_batch_size

        # 3. Optimization Epochs
        # PPO: # epochs = self.n_epochs
        # A2C: # epochs = 1
        epochs = self.n_epochs if isinstance(self.loss, PPOLoss) else 1
        loss_history = []
        distribution_history = []

        self.network.train()
        for _ in range(epochs):

            indices = torch.randperm(batch_size, device=self.device)

            # Iterate over the batch in chunks of mini_batch_size
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_indices = indices[start:end]

                # Slice mini-batch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]

                # Forward pass on mini-batch
                distribution, new_values = self.network(mb_states)

                # Compute PPO loss
                loss = self.loss.compute(dist = distribution,
                                         values = new_values,
                                         actions = mb_actions,
                                         returns = mb_returns,
                                         advantages = mb_advantages,
                                         old_log_probs = mb_old_log_probs)

                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                loss_history.append(loss.item())
                distribution_history.append(distribution.probs.mean(dim=0).cpu().detach().numpy())

        if self.scheduler:
            self.scheduler.step() # Note: Step scheduler per rollout, not per environment frame

        return {"loss": np.mean(loss_history), "mean_reward": rewards.mean().item(), "last_distribution": distribution_history[-1]}



def main(config_path, train_flag):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = flat_config(config)

    # Initialize Policy
    policy = Policy(**config)

    # Reset environment and get initial state
    initial_state, _ = policy.environment.reset()

    if train_flag:
        print("Starting training...")
        n_episodes = config.get("n_episodes", 100)
        save_model = config.get("save_model", False)
        checkpoint = config.get("checkpoint", 10)
        folder_path = config.get("save_path", f"./src/policy/models/")

        for episode in range(n_episodes):
            state = policy.environment.reset()[0]
            done = False

            info = policy.train(state)

            if episode % 10 == 0:
                if isinstance(policy.network, (ConvPPO, AttentionPPO)):
                    print(f"Episode {episode + 1}/{n_episodes} \t\t Loss: {info['loss']:.4f}, "
                      f"Mean Reward: {info['mean_reward']:.4f}, Distribution: {info['last_distribution']}")
                else:
                    print(f"Episode {episode + 1}/{n_episodes} \t\t Loss: {info['loss']:.4f}, "
                      f"Mean Value: {info['mean_value']:.4f}, Epsilon: {policy.epsilon_strategy.eps:.4f}")
            if save_model and episode % checkpoint == 0:
                policy.save_network(path=f"{folder_path}ep_{episode+1}-{n_episodes}.pth")

        if config.get("save_model", False):
            policy.save_network(path=f"{folder_path}_final.pth")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy Training Modules")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--train", action="store_true", help="Flag to indicate training mode.")
    args = parser.parse_args()

    main(config_path=args.config, train_flag=args.train)
