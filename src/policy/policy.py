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
import yaml
import argparse

import numpy as np
import torch

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
        self.reward_discount = reward_discount

        # Training attributes
        self.optimizer = None
        self.scheduler = None
        self.loss = None
        self.n_epochs = n_epochs
        self.device = device

        # Set architecture
        self.set_environment(environment, **kwargs)
        self.set_network(network, **kwargs)
        if kwargs.get("model_path"):
            self.load_network(path = str(kwargs.get("model_path")))
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


    def set_epsilon_strategy(self, epsilon_strategy : str, **kwargs):
        """Sets the epsilon-greedy strategy for exploration."""
        maps = {
            "EpsilonGreedy": EpsilonGreedy,
            "EpsilonConstant": lambda **kwargs: EpsilonGreedy(epsilon_start = kwargs.get("epsilon_start", 1.0),
                                                              epsilon_coeff= 1.0,
                                                              epsilon_end = kwargs.get("epsilon_end", 0.0))
        }

        # Assertions
        assert epsilon_strategy in maps, \
            f"Epsilon strategy '{epsilon_strategy}' is not supported. Supported strategies are: {list(maps.keys())}."
        assert "epsilon_start" in kwargs and "epsilon_coeff" in kwargs and "epsilon_end" in kwargs, \
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
            "MSELoss": MSELoss,
            "PPOLoss": PPOLoss,
            "A2CLoss": A2CLoss
        }

        # Assertions
        assert loss in maps, \
            f"Loss function '{loss}' is not supported. Supported loss functions are: {list(maps.keys())}."

        print(f"Using loss function: {loss}")
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
        # 1. Flatten the 2D grid from the environment (20, 20) -> (400)
        if state_tensor.shape == (20, 20):
            state_tensor = state_tensor.flatten()

        # 2. Add Batch dimension (400) -> (1, 400)
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # 3. Add Channel dimension specifically for Convolutional and Attention Networks -> (1, 1, 400)
        if isinstance(self.network, (ConvDQN, ConvPPO, AttentionDQN, AttentionPPO)) and state_tensor.dim() == 2:
            state_tensor = state_tensor.unsqueeze(1)

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

    @torch.no_grad()
    def _get_rollout(self, state):
        """Runs a rollout in the environment, given an initial state"""
        states, actions, rewards, values, log_probs, dones = [], [], [], [], [], []

        # Deal with batch dimensions
        # Final shape: [batch_size, channel, pixels]
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        if state.shape == (20, 20):
            state = state.flatten()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.dim() == 2: # Require a batch dimension
            state = state.unsqueeze(1)

        # Rollout loop
        for step in range(self.horizon):

            distribution, value = self.network(state)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            next_state, reward, done, truncated, _ = self.environment.step(action.item())
            stop = done or truncated

            states.append(state)
            actions.append(action)
            rewards.append(torch.tensor([reward], dtype=torch.float32, device=self.device))
            values.append(value.squeeze(-1))
            dones.append(torch.tensor([stop], dtype=torch.float32, device=self.device))
            log_probs.append(log_prob)

            state = next_state
            if stop:
                state, _ = self.environment.reset()

            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            if state.shape == (20, 20):
                state = state.flatten()
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if state.dim() == 2: # Require a batch dimension
                state = state.unsqueeze(1)



        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0)
        values = torch.cat(values, dim=0)
        dones = torch.cat(dones, dim=0)
        log_probs = torch.cat(log_probs, dim=0)

        # Value of final state for GAE bootstrapping
        _, last_value = self.network(state)
        values = torch.cat((values, last_value.squeeze(-1)), dim=0)

        return states, actions, rewards, values, log_probs, dones


    def train(self, batch_or_state):
        """
        Training entry point.
        - If DQN: Expects a batch tuple (states, actions, rewards, next_states, dones).
        - If PPO/A2C: Expects a single starting state to begin on-policy rollout.
        """
        is_policy_based = hasattr(self.network, 'actor_head')

        self.network.train()
        if is_policy_based:
            return self._train_on_policy(init_state=batch_or_state)
        else:
            return self._train_off_policy(batch=batch_or_state)


    def _train_off_policy(self, batch):
        """DQN Optimization via Bellman Equation and Replay Buffer."""
        states, actions, rewards, next_states, dones = batch

        # 1. Flatten the 2D grids: [batch_size, 20, 20] -> [batch_size, 400]
        if states.dim() == 3 and states.shape[1:] == (20, 20):
            states = states.view(states.shape[0], -1)
            next_states = next_states.view(next_states.shape[0], -1)

        # 2. Channel/Sequence dimension for Conv/Attention Networks -> [batch_size, 1, 400]
        if isinstance(self.network, (ConvDQN, AttentionDQN)) and states.dim() == 2:
            states = states.unsqueeze(1)
            next_states = next_states.unsqueeze(1)

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
        td_loss = self.loss(current_q, target_q)
        print(f"TD Loss: {td_loss.item():.4f}, Mean Q: {current_q.mean().item():.4f}, Epsilon: {self.epsilon_strategy.eps:.4f}")

        self.optimizer.zero_grad()
        td_loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        # Update Epsilon
        self.epsilon_strategy.step()

        return {"loss": td_loss.item(), "mean_q": current_q.mean().item()}


    def _train_on_policy(self, init_state):
        """PPO/A2C Optimization via Trajectory Rollout and GAE."""

        # Compute rollout
        states, actions, rewards, values, log_probs, dones = self._get_rollout(state = init_state)

        # Compute advantages and returns
        returns, advantages = self.loss.compute_advantages(last_state_value = values[-1],
                                                          rewards = rewards,
                                                          values = values,
                                                          dones = dones)

        # 3. Optimization Epochs
        # PPO: # epochs = self.n_epochs
        # A2C: # epochs = 1
        epochs = self.n_epochs if isinstance(self.loss, PPOLoss) else 1
        total_loss_history = []

        for _ in range(epochs):
            distribution, new_values = self.network(states)

            # Branch based on loss function signature
            loss = self.loss.compute(dist = distribution,
                                     values = new_values,
                                     actions = actions,
                                     returns = returns,
                                     advantages = advantages,
                                     old_log_probs = log_probs.detach())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
            self.optimizer.step()
            total_loss_history.append(loss.item())

        if self.scheduler:
            self.scheduler.step() # Note: Step scheduler per rollout, not per environment frame

        return {"loss": np.mean(total_loss_history), "mean_reward": rewards.mean().item()}



def main(config_path, train_flag):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Flatten the nested dictionary manually
    config = flat_config(config)

    # Initialize Policy
    policy = Policy(**config)

    # Example usage: Reset environment and get initial state
    initial_state, _ = policy.environment.reset()
    action, _ = policy.get_action(initial_state)
    print(f"Initial action selected: {action}")

    if train_flag:
        print("Starting training...")
        n_episodes = config.get("n_episodes", 100)
        save_model = config.get("save_model", False)
        checkpoint = config.get("checkpoint", 10)
        folder_path = config.get("save_path", f"./src/policy/models/")
        for episode in range(n_episodes):
            print(f"Episode {episode + 1}/{config.get('n_episodes', 100)}")
            state = policy.environment.reset()[0]
            done = False

            # Policy Training Loop
            if isinstance(policy.network, (ConvPPO, AttentionPPO)):
                policy.train(state)

                if save_model and episode % checkpoint == 0:
                    policy.save_network(path=f"{folder_path}ep_{episode+1}-{n_episodes}.pth")

            else:
                # For DQN, we need a replay buffer.
                buffer = []
                while not done:
                    action, _ = policy.get_action(state)
                    next_state, reward, done, truncated, _ = policy.environment.step(action)
                    buffer.append((state, action, reward, next_state, done or truncated))
                    state = next_state

                # Convert buffer to batch tensors
                states, actions, rewards, next_states, dones = zip(*buffer)
                states = torch.tensor(np.array(states), dtype=torch.float32, device=policy.device)
                actions = torch.tensor(np.array(actions), dtype=torch.int64, device=policy.device)
                rewards = torch.tensor(np.array(rewards), dtype=torch.float32, device=policy.device)
                next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=policy.device)
                dones = torch.tensor(np.array(dones), dtype=torch.float32, device=policy.device)
                batch = (states, actions, rewards, next_states, dones)

                policy.train(batch)

        if config.get("save_model", False):
            policy.save_network(path=f"{folder_path}_final.pth")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy Training Modules")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--train", action="store_true", help="Flag to indicate training mode.")
    args = parser.parse_args()

    main(config_path=args.config, train_flag=args.train)
