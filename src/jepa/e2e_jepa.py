r"""
  _____ ____  _____          _ _____ ____   _    
 | ____|___ \| ____|        | | ____|  _ \ / \   
 |  _|   __) |  _| _____ _  | |  _| | |_) / _ \  
 | |___ / __/| |__|_____| |_| | |___|  __/ ___ \ 
 |_____|_____|_____|     \___/|_____|_| /_/   \_\
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from typing import Dict, Any, Tuple

from src.game.snake import SnakeEnv
from src.policy.algorithms import ConvPPO, AttentionPPO
from src.policy.regularizers import *
from src.policy.policy import Policy


def save_results(where: str, predictor: nn.Module, encoder: nn.Module, policy_net: nn.Module):
    torch.save({
        "predictor": predictor.state_dict(), 
        "encoder": encoder.state_dict(), 
        "policy_net": policy_net.state_dict()
    }, where)

def load_results(where: str, predictor: nn.Module, encoder: nn.Module, policy_net: nn.Module):
    ldr = torch.load(where, weights_only=False, map_location="cpu")
    predictor.load_state_dict(ldr["predictor"])
    encoder.load_state_dict(ldr["encoder"])
    policy_net.load_state_dict(ldr["policy_net"])

# Experience Replay Buffer for Online Trajectories
class OnlineTrajectoryBuffer:
    """Stores online transitions and serves randomized mini-batches 
    to break temporal correlation during joint optimization."""
    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, x_t, a_t, r_t, x_tp1, done):
        self.buffer.append((x_t, a_t, r_t, x_tp1, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        x_t, a_t, r_t, x_tp1, done = zip(*batch)
        
        # Stack individual steps into batched tensors
        return (
            torch.stack(x_t),
            torch.stack(a_t),
            torch.tensor(r_t, dtype=torch.float32).unsqueeze(-1),
            torch.stack(x_tp1),
            torch.tensor(done, dtype=torch.float32).unsqueeze(-1)
        )

    def refresh(self):
        self.buffer = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.buffer)


# E2E-JEPA
class E2EJEPA:
    def __init__(
        self,
        env : SnakeEnv,
        encoder: nn.Module,
        predictor: nn.Module,
        policy: Policy,
        action_dim: int,
        embed_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 20000,
        coupled_dynamic = False,
        horizon : int = 1,
        alpha: Regularizer = LinearRegularizer(reg_weight_start=0.01, reg_weight_end=0.02, reg_weight_step=1),
        beta : Regularizer | None = None,
        pol_loss_regularizer: Regularizer = PropToOtherLossChangeRegularizer()
    ):
        self.env = env
        self.encoder = encoder
        self.predictor = predictor
        self.policy = policy
        self.action_dim = action_dim
        self.gamma = gamma
        self.horizon = horizon
        
        self.buffer = OnlineTrajectoryBuffer(capacity=buffer_capacity)

        self.alpha = alpha
        self.beta = beta
        self.gamma = pol_loss_regularizer

        if coupled_dynamic:
            self.optimizer = torch.optim.AdamW(
                list(self.encoder.parameters()) +
                list(self.predictor.parameters()) +
                list(self.policy.network.parameters()),
                lr=lr
            )
        else:
            self.optimizer = torch.optim.AdamW(
                list(self.encoder.parameters()) +
                list(self.predictor.parameters()),
                lr= lr
            )

        # SIGReg projections base vector
        self.register_buffer("u_m", F.normalize(torch.randn(embed_dim, 32), p=2, dim=0))


    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)


    def get_action(self, state: torch.Tensor, greedy = False) -> Tuple[torch.Tensor | Any, tuple[Any, Any]]:
        """Phase-Based Exploration: Generates actions on live frames."""
        action, info = self.policy.get_action(state=state, greedy=greedy)
        return action, info


    def compute_trajectory(self, z_t: torch.Tensor, horizon: int=1):
        """Computes a trajectory in the embedded space"""
        z_states, x_next_states, z_next_states, actions, rewards, dones, log_probs, values = [], [], [], [], [], [], [], []

        for _ in range(horizon):
            a_t, (log_prob, value) = self.get_action(state=z_t, greedy=False)
            x_tp1, r_t, done, _, _ = self.env.step(a_t)

            if done:
                x_tp1 = self.env.death_state()

            z_tp1 = self.encoder(x_tp1)

            z_states.append(z_t)
            x_next_states.append(x_tp1)
            z_next_states.append(z_tp1)
            actions.append(a_t)
            rewards.append(r_t)
            dones.append(done)
            log_probs.append(log_prob)
            values.append(value)

            if done:
                x_t, _ = self.env.reset()
                break

            z_t = z_tp1

        # map to tensors
        z_states = torch.stack(z_states)
        x_next_states = torch.stack(x_next_states)
        z_next_states = torch.stack(z_next_states)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)
        log_probs = torch.tensor(log_probs, dtype=torch.float32).unsqueeze(-1)
        values = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)
        batch = (z_states, x_next_states, z_next_states, actions, rewards, dones, log_probs, values)

        return batch


    def compute_sigreg(self, z: torch.Tensor) -> torch.Tensor:
            """Sketched-Isotropic-Gaussian Regularizer."""
            h = torch.matmul(z, self.u_m.to(z.device))
            h_mean = h.mean(dim=0, keepdim=True)
            h_std = h.std(dim=0, keepdim=True) + 1e-6
            return torch.mean((h_std - 1.0) ** 2) + torch.mean(h_mean ** 2)


    def update_parameters(self, batch_size: int, device : str = "cuda") -> Dict[str, float]:
        """Samples from active trajectory memory and performs backpropagation."""
        if len(self.buffer) < batch_size:
            return {} # Not enough data collected yet

        self.encoder.train()
        self.predictor.train()

        # This is completely decoupled

        # Sample online trajectory combinations
        x_t, a_t, r_t, x_tp1, done = self.buffer.sample(batch_size)
        x_t = x_t.to(device=device)
        a_t = a_t.to(device=device)
        r_t = r_t.to(device=device)
        x_tp1 = x_tp1.to(device=device)
        done = done.to(device=device)
    
        # Latent Mappings
        z_t = self.encoder(x_t)[:, 0, :]
        z_tp1_target = self.encoder(x_tp1)[:, 0, :]
        # Add a sequence dimension: [32, 64] -> [32, 1, 64]
        z_t_seq = z_t.unsqueeze(1) 
        # Pass through predictor and remove the sequence dimension: [32, 1, 64] -> [32, 64]
        z_tp1_pred = self.predictor(z_t_seq, a_t).squeeze(1)

        # Collapsing Check
        print(f"Batch {0}, Element {0}: z_t : {z_t[0][0].cpu().detach().numpy()} \t\t "
              f"z_tp1_target : {z_tp1_target[0][0].cpu().detach().numpy()} \t\t "
              f"z_tp1_pred : {z_tp1_pred[0][0].cpu().detach().numpy()}\n")

        # Prediction Loss
        loss_pred = F.mse_loss(z_tp1_pred, z_tp1_target.detach())

        # Anti-Collapse Loss
        loss_sigreg = self.compute_sigreg(z_t)


        if isinstance(self.policy.network, (AttentionPPO, ConvPPO)):
            # to handle differently
            trajectory = self.compute_trajectory(z_t, horizon=self.horizon)
            loss_policy = self.policy.update_parameters(trajectory = trajectory)
        else:
            reg_coeff = self.gamma.step(loss_target = loss_pred.detach() ) if self.gamma else 1.0
            loss_policy = self.policy.update_parameters(init_state = z_t.unsqueeze(1).detach(),
                                                        next_state = z_tp1_target.unsqueeze(1).detach(),
                                                        rewards = r_t,
                                                        dones = done,
                                                        reg_coeff = reg_coeff)

        # Total multi-task execution loss
        # Since the losses are actually decoupled
        # We can just sum them as they are
        total_loss = (
            loss_pred + 
            self.alpha.step(loss_reg = loss_sigreg, loss_target = loss_pred) * loss_sigreg
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        x_t = x_t.to(device="cpu")
        a_t = a_t.to(device="cpu")
        r_t = r_t.to(device="cpu")
        x_tp1 = x_tp1.to(device="cpu")
        done = done.to(device="cpu")

        return {"total_loss": total_loss.item(),
                "pred_loss": loss_pred.item(),
                "policy_loss":  loss_policy,
                "sigreg_loss": loss_sigreg.item()
                }

if __name__ == "__main__":
    pass