import torch
import torch.nn as nn
import torch.nn.functional as F


class MSELoss(nn.MSELoss):
    """Overrides torch.nn.MSELoss to accept dummy **kwargs for compatibility"""

    def __init__(self, **kwargs):
        super().__init__()



class A2CLoss:
    """Common Advantage-Actor Critic Loss"""

    def __init__(self,
                 actor_coef : float | int = 0.01,
                 critic_coef : int | float = 0.5,
                 **kwargs):
        self.actor_coef = actor_coef
        self.critic_coef = critic_coef


    def compute(self,
                dist,
                values,
                actions,
                returns,
                **kwargs):
        """Compute the A2C loss given the distribution, values, actions, and returns."""
        values = values.squeeze(-1)

        # 2. Compute Advantage (Returns - Baseline Value)
        advantages = returns - values.detach()

        # 3. Compute metrics
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # 4. Compute Losses
        actor_loss = -(log_probs * advantages).mean()
        critic_loss = F.mse_loss(values, returns)

        # Total Loss (Minimize actor loss, minimize critic error, maximize entropy)
        total_loss = actor_loss + (self.critic_coef * critic_loss) - (self.actor_coef * entropy)

        return total_loss

    @torch.no_grad()
    def compute_advantages(self, last_state_value, rewards, values, dones):
        """
        Compute Advantages and normalize them.
        """
        horizon = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        advantage = torch.zeros_like(rewards)

        for t in reversed(range(horizon)):
            next_non_terminal = 1.0 - dones[t]
            next_value = last_state_value if t == horizon-1 else values[t+1]

            delta = rewards[t] + self.discount * next_value * next_non_terminal - values[t]
            advantage = self. gae_coeff * self.discount * next_non_terminal * advantage + delta
            returns[t] = advantage + values[t]

        return returns, advantage


class PPOLoss:
    """Common PPO Loss"""

    def __init__(self,
                 actor_coeff : float | int = 0.01,
                 critic_coeff : float | int  = 0.5,
                 gae_coeff : float | int = 0.95,
                 clip_ratio : float | int = 0.2,
                 discount : float | int = 0.99
                 **kwargs):
        self.actor_coeff = actor_coeff
        self.critic_coeff = critic_coeff
        self.gae_coeff = gae_coeff
        self.clip_ratio = clip_ratio
        self.discount = discount

    @torch.no_grad()
    def compute_advantages(self, last_state_value, rewards, values, dones):
        """
        Compute Advantages and normalize them.
        """
        horizon = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        advantage = torch.zeros_like(rewards)

        for t in reversed(range(horizon)):
            next_non_terminal = 1.0 - dones[t]
            next_value = last_state_value if t == horizon-1 else values[t+1]

            delta = rewards[t] + self.discount * next_value * next_non_terminal - values[t]
            advantage = self. gae_coeff * self.discount * next_non_terminal * advantage + delta
            returns[t] = advantage + values[t]

        # normalize advantage
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        return returns, advantage


    def compute(self,
                dist,
                values,
                actions,
                returns,
                old_log_probs,
                **kwargs):
        """Compute the PPO loss given the distribution, values, returns, and old log probabilities."""
        values = values.squeeze(-1)

        # 2. Compute Advantage and Normalize it
        advantages = returns - values.detach()
        if advantages.shape[0] > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Calculate Policy Ratio
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 4. Compute Clipped Surrogate Objective
        surrogate_1 = ratio * advantages
        surrogate_2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages

        # PPO Actor loss is the negative minimum of the two surrogates
        actor_loss = -torch.min(surrogate_1, surrogate_2).mean()

        # 5. Critic and Entropy Losses
        critic_loss = F.mse_loss(values, returns)
        entropy = dist.entropy().mean()

        # Total Loss
        total_loss = actor_loss + (self.critic_coeff * critic_loss) - (self.actor_coeff * entropy)

        return total_loss