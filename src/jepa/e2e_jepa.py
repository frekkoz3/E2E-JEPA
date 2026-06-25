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

# SIGReg parameters
DEFAULT_SIGREG_KNOTS = 17
DEFAULT_SIGREG_NUM_PROJ = 128

# SIGReg
class SIGReg(nn.Module):
    """
    Sketched-Isotropic-Gaussian Regularizer (SIGReg)
    as described in LeWM (Maes et al., 2026).
    """
    def __init__(self, embed_dim: int, num_projections: int = DEFAULT_SIGREG_NUM_PROJ, num_nodes: int = DEFAULT_SIGREG_KNOTS, device : str = "cuda"):
        super().__init__()

        self.device = device

        # 1. Generate M unit-norm directions (u_m)
        # Shape: (embed_dim, num_projections)
        u_m = torch.randn(embed_dim, num_projections).to(device=device)
        u_m = torch.nn.functional.normalize(u_m, p=2, dim=0)
        self.register_buffer('u_m', u_m)

        # 2. Quadrature nodes uniformly distributed in [0.2, 4]
        # Shape: (num_nodes,)
        self.num_nodes = num_nodes
        t_nodes = torch.linspace(0.2, 4.0, num_nodes, device=device)
        self.register_buffer('t_nodes', t_nodes)

        # Trapezoid integration step size
        self.dt = (4.0 - 0.2) / (num_nodes - 1)

        # 3. Target Characteristic Function for standard N(0, 1)
        # phi_0(t) = exp(-t^2 / 2)
        phi_0 = torch.exp(- (t_nodes ** 2) / 2.0)
        self.register_buffer('phi_0', phi_0)

        # 4. Weighting function w(t)
        # Using w(t) = exp(-t^2 / 2) as an example weight
        w_t = torch.exp(- (t_nodes ** 2) / 2.0)
        self.register_buffer('w_t', w_t)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: Latent embeddings tensor of shape (Batch, Seq, Embed_Dim) or (N, Embed_Dim)
        """
        # Flatten temporal/batch dimensions to treat all states as a single set of points
        if z.dim() > 2:
            z = z.reshape(-1, z.size(-1))

        N = z.size(0)

        # Step 1: Project embeddings onto the M random directions
        # h^(m) = Z * u^(m) -> Shape: (N, num_projections)
        h = torch.matmul(z, self.u_m)

        # Step 2: Compute Empirical Characteristic Function (ECF)
        # We need the product of t_nodes and h for the exponential e^(i * t * h)
        # th shape: (num_nodes, N, num_projections)
        th = torch.einsum('k,nm->knm', self.t_nodes, h.to(device=z.device))

        # Real and Imaginary parts of ECF over the N samples
        # ECF_real and ECF_imag shape: (num_nodes, num_projections)
        ecf_real = torch.mean(torch.cos(th), dim=1)
        ecf_imag = torch.mean(torch.sin(th), dim=1)

        # Step 3: Compute the squared difference |phi_N(t) - phi_0(t)|^2
        # phi_0 is unsqueezed to broadcast over the num_projections dimension
        phi_0_k = self.phi_0.unsqueeze(1)
        diff_sq = (ecf_real - phi_0_k) ** 2 + ecf_imag ** 2

        # Step 4: Apply weighting function and integrate using Trapezoidal rule
        integrand = self.w_t.unsqueeze(1) * diff_sq

        # integral shape: (num_projections,)
        integral = torch.trapz(integrand, dx=self.dt, dim=0)

        # Step 5: Average the test statistic over the M projections
        loss = torch.mean(integral)

        return loss

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
        pol_loss_regularizer: Regularizer = PropToOtherLossChangeRegularizer(),
        device = "cuda"
    ):
        self.env = env
        self.encoder = encoder
        self.predictor = predictor
        self.sigreg = SIGReg(embed_dim=embed_dim, device=device)
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


    def predict(self, ctx_emb, ctx_act):
        """Predicts future states autoregressively."""

        # 1. Dimension Fix: Convert discrete integer actions to one-hot floats
        # Transforms (B, T) -> (B, T, action_dim) to match transformer.py's action_proj
        if ctx_act.dim() == 2:
            ctx_act = F.one_hot(ctx_act.long(), num_classes=self.action_dim).float()
        elif ctx_act.dim() == 3 and ctx_act.shape[-1] == 1:
            ctx_act = F.one_hot(ctx_act.squeeze(-1).long(), num_classes=self.action_dim).float()

        # The new transformer.py natively handles causal masking internally
        return self.predictor(ctx_emb, ctx_act)


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

    def update_parameters(self, batch_size: int, device : str = "cuda") -> Dict[str, float]:
        """Samples from active trajectory memory and performs backpropagation."""
        if len(self.buffer) < batch_size:
            return {} # Not enough data collected yet

        self.encoder.train()
        self.predictor.train()

        # This is completely decoupled

        x_seq, a_seq, r_seq, done_seq = self.buffer.sample(batch_size)
        x_seq = x_seq.to(device=device)
        a_seq = a_seq.to(device=device)
        r_seq = r_seq.to(device=device)
        done_seq = done_seq.to(device=device)

        B, T = x_seq.shape[0], x_seq.shape[1]
        context_length = T-1

        z_seq = self.encoder(x_seq)

        context_embedding = z_seq[:, :context_length]
        context_action = a_seq[:, :context_length]
        target_embedding = z_seq[:, 1:context_length+1].detach()

        prediction_embedding = self.predict(context_embedding, context_action)

        # Prediction Loss
        loss_pred = F.mse_loss(prediction_embedding, target_embedding)

        # Anti-Collapse Loss
        loss_sigreg = self.sigreg(context_embedding.transpose(0, 1))

        z_t_policy = context_embedding[:, -1].unsqueeze(1).detach()
        z_tp1_target = target_embedding[:, -1].unsqueeze(1).detach()
        r_t = r_seq[:, -1]
        done_t = done_seq[:, -1]

        if isinstance(self.policy.network, (AttentionPPO, ConvPPO)):
            # to handle differently
            trajectory = self.compute_trajectory(z_t, horizon=self.horizon)
            loss_policy = self.policy.update_parameters(trajectory = trajectory)
        else:
            reg_coeff = self.gamma.step(loss_target = loss_pred.detach() ) if self.gamma else 1.0
            loss_policy = self.policy.update_parameters(init_state = z_t_policy,
                                                        next_state = z_tp1_target,
                                                        rewards = r_t,
                                                        dones = done_t,
                                                        reg_coeff = reg_coeff)

        # Total multi-task execution loss
        # Since the losses are actually decoupled
        # We can just sum them as they are
        alpha_val = self.alpha.step(loss_reg = loss_sigreg, loss_target = loss_pred)
        total_loss = (
            loss_pred + 
            alpha_val * loss_sigreg
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