import torch
import torch.nn.functional as F

class A2CLoss:
    """Common Advantage-Actor Critic Loss"""

    def __init__(self, **kwargs):
        pass


    def compute(self,
                dist,
                values,
                actions,
                returns,
                entropy_coef=0.01,
                critic_coef=0.5):
        """Compute the PPO loss given the distribution, values, actions, and returns."""
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
        total_loss = actor_loss + (critic_coef * critic_loss) - (entropy_coef * entropy)

        return total_loss



class PPOLoss:
    """Common PPO Loss"""

    def __init__(self, **kwargs):
        pass


    def compute(self,
                dist,
                values,
                actions,
                returns,
                old_log_probs,
                clip_ratio=0.2,
                entropy_coef=0.01,
                critic_coef=0.5):
        """Compute the PPO loss given the distribution, values, returns, and old log probabilities."""
        values = values.squeeze(-1)

        # 2. Compute Advantage and Normalize it
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Calculate Policy Ratio
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 4. Compute Clipped Surrogate Objective
        surrogate_1 = ratio * advantages
        surrogate_2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages

        # PPO Actor loss is the negative minimum of the two surrogates
        actor_loss = -torch.min(surrogate_1, surrogate_2).mean()

        # 5. Critic and Entropy Losses
        critic_loss = F.mse_loss(values, returns)
        entropy = dist.entropy().mean()

        # Total Loss
        total_loss = actor_loss + (critic_coef * critic_loss) - (entropy_coef * entropy)

        return total_loss