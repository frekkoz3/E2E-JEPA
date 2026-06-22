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
from typing import Dict, Any, Tuple
import yaml
import argparse
import copy
import random
import numpy as np
import torch
from collections import deque
import gymnasium as gym

from src.policy.algorithms import *
from src.policy.epsilon import *
from src.policy.losses import *

from src.game.snake import *
from src.utils.utils import *



class Policy:
    """Policy class that combines a network architecture and an epsilon-greedy strategy for action selection."""

    def __init__(self,
                 environment : str | None = "Snake",
                 network : str = "DQN",
                 epsilon_strategy : str = "EpsilonGreedy",
                 optimizer : str = "Adam",
                 loss : str = "MSELoss",
                 device : str = "cpu",
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
        self.reward_discount = reward_discount

        # Training attributes
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.device = device

        # Set architecture
        if environment:
            self._set_environment(environment, **kwargs)
        self._set_network(network, **kwargs)
        if kwargs.get("model_path", None) is not None:
            self.load_network(path = str(kwargs.get("model_path")))
        if self.device != "cpu":
            self.network.to(self.device)
        self._set_epsilon_strategy(epsilon_strategy, **kwargs)
        self._set_optimizer(optimizer, **kwargs)
        self._set_loss(loss, **kwargs)


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
        self.environment = gym.vector.SyncVectorEnv([lambda: maps[environment](**kwargs) for _ in range(num_envs)]) \
            if num_envs > 1 else maps[environment](**kwargs)


    def _set_network(self, network : str, **kwargs):
        """Sets the network architecture for approximating Q-values."""
        maps = {
            "DQN": DQN,
            "ConvDQN": ConvDQN,
            "ConvDQN2D": ConvDQN2D,
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
        elif network == "ConvDQN" or network == "ConvPPO" or network == "ConvDQN2D":
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

        self.loss = maps[loss](**kwargs)


    def save_network(self, path : str):
        """Saves the model parameters to the specified path."""
        print(path)
        torch.save(self.network.state_dict(), path)


    def load_network(self, path : str):
        """Loads the model parameters from the specified path."""
        self.network.load_state_dict(torch.load(path, map_location=self.device))


    def _format_state(self, state):
        """Formats the state from the environment to match the expected input shape of the network."""
        pass


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
        pass


    def train(self, state, n_trajectories):
        """
        Training entry point.
        - If DQN: Expects a batch tuple (states, actions, rewards, next_states, dones).
        - If PPO/A2C: Expects a single starting state to begin on-policy rollout.
        """
        pass


    def trainer(self, **kwargs):
        """Manages a training session"""
        assert self.environment is not None, "Environment must be set for training"

        n_trajectories = kwargs.get("n_trajectories", 100)
        save_model = kwargs.get("save_model", False)
        checkpoint = kwargs.get("checkpoint", 10)
        folder_path = kwargs.get("save_path", f"./src/policy/models/")

        for trajectory in range(n_trajectories):
            state = self.environment.reset()[0]

            self.train(state, n_trajectories = n_trajectories)
            if save_model and trajectory % checkpoint == 0:
                self.save_network(path=f"{folder_path}ep_{trajectory+1}-{n_trajectories}.pth")

        if save_model:
            self.save_network(path=f"{folder_path}final.pth")



class PolicyPPO(Policy):
    """Subclass for PPO-based policies"""

    def __init__(self,  **kwargs):

        assert kwargs.get("network", "ConvPPO") in ["ConvPPO", "AttentionPPO"], \
            f"Invalid network type: {kwargs.get('network')}. Must be 'ConvPPO' or 'AttentionPPO'."
        assert kwargs.get("loss", "PPOLoss") in ["PPOLoss", "A2CLoss"], \
            f"Invalid loss type: {kwargs.get('loss')}. Must be 'PPOLoss' or 'A2CLoss'."
        assert kwargs.get("epsilon_strategy", "EpsilonConstant") == "EpsilonConstant", \
            f"Invalid epsilon strategy: {kwargs.get('epsilon_strategy')}. Must be 'EpsilonConstant' for PolicyPPO."

        super().__init__(**kwargs)

        self.n_inner_epochs = kwargs.get("n_inner_epochs", 4)
        self.mini_batch_size = kwargs.get("mini_batch_size", 8)
        self.horizon = kwargs.get("horizon", 1)


    @torch.no_grad()
    def _get_rollout(self, state):
        """"Runs a rollout in the environment, given an initial state"""
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
        if isinstance(self.environment, gym.vector.VectorEnv):
            last_value = last_value.squeeze(-1).squeeze(-1).unsqueeze(0)
        else:
            last_value = last_value.squeeze(-1).unsqueeze(0)
        values = torch.cat( [values, last_value], dim=0)

        return states, actions, rewards, values, log_probs, dones


    def _format_state(self, state : Any) -> torch.Tensor:
        """
        Formats the input state into a suitable format for the PPO network:
                [batch_size, n_channels, raw_pixels] = [1, 1, 400]
        """
        state_tensor = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32, device=self.device)

        # (num_envs, 20, 20) -> (num_envs, 1, 400)
        if state_tensor.dim() == 3 and state_tensor.shape[1:] == (20, 20):
            state_tensor = state_tensor.view(state_tensor.shape[0], 1, 400)

        # (20, 20) -> (1, 1, 400)
        elif state_tensor.shape == (20, 20):
            state_tensor = state_tensor.flatten().unsqueeze(0).unsqueeze(0)


        return state_tensor

    @torch.no_grad()
    def get_action(self,
                   state : torch.Tensor | Tuple[torch.Tensor, ...],
                   greedy = False) -> Tuple[torch.Tensor | Any, tuple[Any, Any]]:
        """Selects an action based on the current state using a policy derived from the PPO algorithm."""
        dist, value = self.network(state)

        if greedy:
            action_tensor = torch.argmax(dist.logits, dim=-1)
        else:
            action_tensor = dist.sample()

        log_p = dist.log_prob(action_tensor)
        if isinstance(self.environment, gym.vector.VectorEnv):
            action_out = action_tensor.cpu().numpy()
        else:
            action_out = action_tensor.item()

        return action_out, (log_p, value.squeeze(-1))


    def train(self, init_state, n_trajectories : int):
        """PPO full training loop"""
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
        # PPO: # epochs = self.n_inner_epochs
        # A2C: # epochs = 1
        epochs = self.n_inner_epochs if isinstance(self.loss, PPOLoss) else 1
        loss_history = []
        distribution_history = []

        self.network.train()
        for epoch in range(epochs):

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

        info = {"loss": np.mean(loss_history), "mean_reward": rewards.mean().item(), "last_distribution": distribution_history[-1]}
        if epoch % 10 == 0:
            print(f"Episode {epoch + 1}/{n_trajectories} \t\t Loss: {info['loss']:.4f}, Mean Value: {info['mean_value']:.4f}, Epsilon: {self.epsilon_strategy.eps:.4f}")

        return info


    def update_parameters(self, trajectory : torch.Tensor | Tuple[torch.Tensor, ...]) -> Dict[str, float | Any]:
        """
        Updates the parameters of the architecture based on policy gradient optimization.

        Parameters
        ----------
        trajectory : torch.Tensor | Tuple[torch.Tensor, ...]
            A trajectory of states, actions, rewards, and other relevant information used for computing the policy gradient update.

        Returns
        -------
        loss : Dict[str,  float]
            A dictionary containing the computed losses for monitoring and analysis.

        Notes
        -----
        The update is slower than the one for DQN, since it requires (in e2e_jepa.update_parameters() ) to compute the full trajectory of states, actions, next states, rewards, dones.
        """
        z_states, _, z_next_states, actions, rewards, dones, log_probs, values = trajectory

        # Compute Advantages and Returns using GAE
        returns, advantages = self.loss.compute_advantages(last_state_value = values[-1], rewards = rewards, values = values, dones = dones)

        loss_history = []
        dist_history = []

        self.network.train()
        for _ in range(self.n_inner_epochs):
            dist, value = self.network(z_states)
            loss = self.loss(dist = dist, values = value, actions = actions, returns = returns, advantages = advantages, old_log_probs = log_probs)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()

            loss_history.append(loss.item())
            dist_history.append(dist.probs.mean(dim=0).cpu().etach().numpy())

        return {"loss": np.mean(loss_history), "mean_distribution": np.mean(dist_history, axis=0)}



class PolicyDQN(Policy):
    """Subclass for DQN-based policies"""

    def __init__(self,**kwargs):

        assert kwargs.get("network", "ConvDQN") in ["ConvDQN", "ConvDQN2D", "AttentionDQN", "DQN"], \
            f"Invalid network type: {kwargs.get('network')}. Must be 'ConvDQN' or 'AttentionDQN' or 'DQN'."
        assert kwargs.get("loss", "MSELoss") == "MSELoss", \
            f"Invalid loss type: {kwargs.get('loss')}. Must be 'MSELoss'."
        assert kwargs.get("epsilon_strategy", "EpsilonConstant") in ["EpsilonConstant", "EpsilonGreedy"], \
            f"Invalid epsilon strategy: {kwargs.get('epsilon_strategy')}. Must be 'EpsilonConstant' or 'EpsilonGreedy'."

        super().__init__(**kwargs)

        self.target_network = copy.deepcopy(self.network)
        self.target_net_update_freq = kwargs.get("target_net_update_freq", 25)
        self.epoch = 0

        self.buffer_size = kwargs.get("buffer_size", 10000)
        self.buffer = deque(maxlen=self.buffer_size)
        self.batch_size = kwargs.get("batch_size", 64)

    @torch.no_grad()
    def _full_buffer(self):
        """Computes a buffer of size self.buffer_size of trajectories for Off-Policy DQN."""

        while len(self.buffer) < self.buffer_size:

            state = self.environment.reset()[0]

            state = self._format_state(state)
            is_terminal = False

            while not is_terminal:
                # Action is a scalar integer for a single environment
                action, _ = self.get_action(state, greedy=False)

                # Step returns scalars and standard Python booleans
                next_state, reward, done, truncated, _ = self.environment.step(action)
                is_terminal = done or truncated

                next_state_formatted = self._format_state(next_state)

                # convert to tensors and store in buffer
                action = torch.tensor(action, dtype=torch.int64, device=self.device)
                reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
                is_terminal = torch.tensor(is_terminal, dtype=torch.float32, device=self.device)

                self.buffer.append((state, action, reward, next_state_formatted, is_terminal))
                state = next_state_formatted

        # shuffle the buffer to ensure decorrelation
        random.shuffle(self.buffer)


    def _format_state(self, state : Any, bracket = True) -> torch.Tensor:
        """Formats the input state into a suitable format for the PPO network."""
        state_tensor = state if isinstance(state, torch.Tensor) else torch.tensor(state, dtype=torch.float32, device=self.device)

        # Find 1 for head, 2 for food.
        if bracket:
            head_indices = (state_tensor == 2).nonzero(as_tuple=True)   # fixed swap
            food_indices = (state_tensor == 1).nonzero(as_tuple=True)

            head_y = head_indices[-2][0].float() if len(head_indices[-2]) > 0 else torch.tensor(0., device=self.device)
            head_x = head_indices[-1][0].float() if len(head_indices[-1]) > 0 else torch.tensor(0., device=self.device)
            food_y = food_indices[-2][0].float() if len(food_indices[-2]) > 0 else torch.tensor(0., device=self.device)
            food_x = food_indices[-1][0].float() if len(food_indices[-1]) > 0 else torch.tensor(0., device=self.device)

            # Relative food position (generalises better than absolute food coords)
            delta_x = food_x - head_x
            delta_y = food_y - head_y

            # Current direction as one-hot over {UP, DOWN, LEFT, RIGHT}
            direction_map = {(0, -1): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3}
            dir_idx = direction_map.get(self.environment.direction, 0)
            dir_onehot = torch.zeros(4, dtype=torch.float32, device=self.device)
            dir_onehot[dir_idx] = 1.0

            # 8-dim state: [head_x, head_y, Δx, Δy, dir_UP, dir_DOWN, dir_LEFT, dir_RIGHT]
            state_brack = torch.cat([
                torch.stack([head_x, head_y, delta_x, delta_y]),
                dir_onehot
            ]).unsqueeze(0)   # shape [1, 8]

            return state_brack

        if isinstance(self.network, ConvDQN2D):
            # Conv2d expects (batch_size, in_channels, H, W)
            if state_tensor.shape == (20, 20):
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(0) # Becomes (1, 1, 20, 20)
            elif state_tensor.dim() == 3:
                state_tensor = state_tensor.unsqueeze(1) # Becomes (batch_size, 1, 20, 20)
            return state_tensor

        # 2. Flatten the 2D grid from the environment (20, 20) -> (1, 400)
        if state_tensor.shape == (20, 20):
            state_tensor = state_tensor.flatten().unsqueeze(0)

        if not isinstance(self.network, DQN) and state_tensor.dim() == 2:
            state_tensor = state_tensor.unsqueeze(1)

        return state_tensor

    @torch.no_grad()
    def get_action(self,
                   state : torch.Tensor | Tuple[torch.Tensor, ...],
                   greedy : bool = False) -> Tuple[torch.Tensor | Any, tuple[Any, Any]]:
        """Selects an action based on the current state using an epsilon-greedy strategy."""
        if not greedy and np.random.rand() < self.epsilon_strategy.eps:
            q_values = None
            action = np.random.randint(0, self.network.output.out_features)
        else:
            q_values = self.network(state)
            action = torch.argmax(q_values, dim=-1).item()
        return action, (q_values.cpu().detach().numpy() if q_values is not None else None)


    def train(self, init_state, n_trajectories):
        """Collect one episode, add every transition to the rolling buffer,
        then do one gradient step per transition (if buffer is warm)."""

        state = self._format_state(self.environment.reset()[0])
        is_terminal = False
        loss_sum, step_count = 0.0, 0

        while not is_terminal:
            action, _ = self.get_action(state, greedy=False)
            next_state_raw, reward, done, truncated, _ = self.environment.step(action)
            is_terminal = done or truncated
            next_state = self._format_state(next_state_raw)

            self.buffer.append((
                state,
                torch.tensor(action,       dtype=torch.int64,   device=self.device),
                torch.tensor(reward,       dtype=torch.float32, device=self.device),
                next_state,
                torch.tensor(float(is_terminal), dtype=torch.float32, device=self.device),
            ))
            state = next_state

            # Only train once buffer is warm
            if len(self.buffer) < self.batch_size:
                continue

            batch = random.sample(self.buffer, self.batch_size)
            b_states, b_actions, b_rewards, b_next_states, b_dones = zip(*batch)

            b_states      = torch.cat(b_states).to(self.device)
            b_next_states = torch.cat(b_next_states).to(self.device)
            b_actions     = torch.stack(b_actions).to(self.device)
            b_rewards     = torch.stack(b_rewards).to(self.device)
            b_dones       = torch.stack(b_dones).to(self.device)

            q_values  = self.network(b_states)
            current_q = q_values.gather(1, b_actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q  = self.target_network(b_next_states)
                target_q = b_rewards + self.reward_discount * next_q.max(dim=1)[0] * (1 - b_dones)

            loss = self.loss(current_q, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()

            loss_sum   += loss.item()
            step_count += 1

        self.epsilon_strategy.step()
        if self.scheduler:
            self.scheduler.step()

        if (self.epoch + 1) % self.target_net_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        self.epoch += 1

        mean_loss = loss_sum / step_count if step_count > 0 else 0.0

        info = {
            "loss": mean_loss,
            # "mean_value": current_q.mean().item()
        }
        if self.epoch % 50 == 0:
            self.buffer.clear()

        if self.epoch % 50 == 0:
             # Clear the buffer every 50 episodes to ensure fresh data collection and prevent overfitting to old transitions
            print(
                f"Episode {self.epoch + 1}/{n_trajectories}"
                f" | Loss: {info['loss']:.4f}"
                # f" | MeanQ: {info['mean_value']:.4f}"
                f" | Epsilon: {self.epsilon_strategy.eps:.4f}"
            )
        return info


    def update_parameters(self,
                          init_state : torch.Tensor | Tuple[torch.Tensor, ...],
                          next_state : torch.Tensor | Tuple[torch.Tensor],
                          rewards : torch.Tensor,
                          dones : torch.Tensor,
                          ) -> Dict[str, float | int]:
        """
        Computes a TD learning step for the DQN architecture.

        Parameters
        ----------
        init_state : torch.Tensor | Tuple[torch.Tensor, ...]
            The initial state or a tuple of states from which to compute the DQN update.
        next_state : torch.Tensor | Tuple[torch.Tensor]
            The next state or a tuple of next states corresponding to the initial states.
        rewards : torch.Tensor
            The rewards received after taking actions in the initial states.
        dones : torch.Tensor
            A tensor indicating whether the episodes have terminated after taking actions in the initial states.

        Returns
        -------
        loss : Dict[str, float]
            A dictionary containing the computed losses for monitoring and analysis.
        """
        # Compute Q-Values for the initial state
        q_values = self.network(init_state)
        online_q_values = q_values.gather(1, torch.argmax(q_values, dim=-1, keepdim=True)).squeeze(-1)

        # Compute Target Q-Values for the next state using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state)
            max_next_q_values, _ = torch.max(next_q_values, dim=-1)
            print(f"{max_next_q_values.device}, {dones.device}")
            target_q_values = rewards + self.reward_discount * max_next_q_values * (1 - dones)
            

        # Compute the loss (MSE) between online and target Q-Values
        loss = self.loss(online_q_values, target_q_values)

        # Optimize parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.epsilon_strategy.step()

        # Update target network
        self.epoch += 1
        if self.epoch % self.target_net_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        return loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy Training Modules")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--train", action="store_true", help="Flag to indicate training mode.")
    args = parser.parse_args()

    config_path, train_flag = args.config, args.train

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config = flat_config(config)

    # Initialize Policy
    if config.get("network") in ["AttentionPPO", "ConvPPO"]:
        policy = PolicyPPO(**config)
    else:
        policy = PolicyDQN(**config)

    # Reset environment and get initial state
    initial_state, _ = policy.environment.reset()

    if train_flag:
        print("Starting training...")
        policy.trainer(**config)
