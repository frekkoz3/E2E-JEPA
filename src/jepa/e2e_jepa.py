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


# E2E-JEPA Trainer
class ActiveE2EJEPATrainer:
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        policy_network: nn.Module,
        action_dim: int,
        latent_dim: int = 192,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 20000,
        coupled_dynamic = False
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.policy_network = policy_network
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.buffer = OnlineTrajectoryBuffer(capacity=buffer_capacity)
        
        self.optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) +
            list(self.predictor.parameters()) +
            list(self.policy_network.parameters()),
            lr=lr
        )

        # SIGReg projections base vector
        self.register_buffer("u_m", F.normalize(torch.randn(latent_dim, 32), p=2, dim=0))


    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)


    def get_action(self, x_t: torch.Tensor, epoch: int, total_epochs: int) -> int:
        """Phase-Based Exploration: Generates actions on live frames."""
        self.encoder.eval()
        self.policy_network.eval()
        
        # Linear decay for epsilon exploration schedule 
        epsilon = max(0.1, 1.0 - (epoch / (total_epochs * 0.7))) # this converge to 0.1 when the actual epoch is 0.63 of the total epochs
        
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad(): # This is the decoupled version
            z_t = self.encoder(x_t.unsqueeze(0)) # Add batch dim
            q_values = self.policy_network(z_t)
            return torch.argmax(q_values, dim=-1).item()

    def compute_sigreg(self, z: torch.Tensor) -> torch.Tensor:
        """Sketched-Isotropic-Gaussian Regularizer."""
        h = torch.matmul(z, self.u_m.to(z.device))
        h_mean = h.mean(dim=0, keepdim=True)
        h_std = h.std(dim=0, keepdim=True) + 1e-6
        return torch.mean((h_std - 1.0) ** 2) + torch.mean(h_mean ** 2)

    def update_parameters(self, batch_size: int, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Samples from active trajectory memory and performs backpropagation."""
        if len(self.buffer) < batch_size:
            return {} # Not enough data collected yet

        self.encoder.train()
        self.predictor.train()
        self.policy_network.train()

        # This is completely decoupled

        # Sample online trajectory combinations
        x_t, a_t, r_t, x_tp1, done = self.buffer.sample(batch_size)

        # Regularization parameter
        alpha = 0.1
    
        # Latent Mappings
        z_t = self.encoder(x_t) # to take the cls token
        z_tp1_target = self.encoder(x_tp1) # to take the cls token
        z_tp1_pred = self.predictor(z_t, a_t) # to project in the correct dimension

        # Prediction Loss
        loss_pred = F.mse_loss(z_tp1_pred, z_tp1_target.detach())

        # Anti-Collapse Loss
        loss_sigreg = self.compute_sigreg(z_t)

        # Policy DQN Loss (TD-Error)
        q_values = self.policy_network(z_t)
        state_action_values = q_values.gather(1, torch.argmax(a_t, dim=-1, keepdim=True))

        with torch.no_grad():
            next_q_values = self.policy_network(z_tp1_target)
            max_next_q = next_q_values.max(1, keepdim=True)[0]
            expected_q = r_t + (self.gamma * max_next_q * (1 - done))

        loss_policy = F.smooth_l1_loss(state_action_values, expected_q)

        # Total multi-task execution loss
        # Since the losses are actually decoupled
        # We can just some them as they are
        total_loss = (
            loss_pred + 
            loss_policy + 
            alpha * loss_sigreg
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"total_loss": total_loss.item(), "pred_loss": loss_pred.item(), "policy_loss": loss_policy.item()}

if __name__ == "__main__":
    pass