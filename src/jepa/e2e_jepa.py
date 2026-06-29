r"""
  _____ ____  _____          _ _____ ____   _    
 | ____|___ \| ____|        | | ____|  _ \ / \   
 |  _|   __) |  _| _____ _  | |  _| | |_) / _ \  
 | |___ / __/| |__|_____| |_| | |___|  __/ ___ \ 
 |_____|_____|_____|     \___/|_____|_| /_/   \_\
"""
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import random
from collections import deque
from typing import Dict, Any, Tuple

from src.game.snake import SnakeEnv
from src.policy.algorithms import *
from src.policy.regularizers import *
from src.policy.policy import Policy


def save_results(where: str,
                 predictor: nn.Module,
                 encoder: nn.Module,
                 projector : nn.Module,
                 policy_net: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler : torch.optim.lr_scheduler._LRScheduler,
                 policy_optimizer : torch.optim.Optimizer,
                 policy_scheduler : torch.optim.lr_scheduler._LRScheduler,
                 epsilon_start : float | int,
                 ):
    torch.save({
        "predictor": predictor.state_dict(), 
        "encoder": encoder.state_dict(),
        "projector": projector.state_dict(),
        "policy_net": policy_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "pol_optimizer": policy_optimizer.state_dict(),
        "pol_scheduler": policy_scheduler.state_dict(),
        "epsilon_start": epsilon_start,
    }, where)

def load_results(where: str,
                 predictor: nn.Module,
                 encoder: nn.Module,
                 projector : nn.Module,
                 policy_net: nn.Module,
                 optimizer : torch.optim.Optimizer,
                 scheduler : torch.optim.lr_scheduler._LRScheduler,
                 policy_optimizer : torch.optim.Optimizer,
                 policy_scheduler : torch.optim.lr_scheduler._LRScheduler,
                 policy_eps,
                ):
    ldr = torch.load(where, weights_only=False, map_location="cpu")
    predictor.load_state_dict(ldr["predictor"])
    encoder.load_state_dict(ldr["encoder"])
    projector.load_state_dict(ldr["projector"])
    policy_net.load_state_dict(ldr["policy_net"])
    optimizer.load_state_dict(ldr["optimizer"])
    scheduler.load_state_dict(ldr["scheduler"])
    policy_optimizer.load_state_dict(ldr["pol_optimizer"])
    policy_scheduler.load_state_dict(ldr["pol_scheduler"])
    policy_eps.eps = ldr["epsilon_start"]


# Experience Replay Buffer for Online Trajectories
class OnlineTrajectoryBuffer:
    """Stores online transitions and serves randomized mini-batches 
    to break temporal correlation during joint optimization."""
    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, x_t, a_t, r_t, done):
        self.buffer.append((x_t, a_t, r_t, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        x_t, a_t, r_t, done = zip(*batch)

        # Stack individual steps into batched tensors
        return (
            torch.stack(x_t),
            torch.stack(a_t),
            torch.tensor(r_t, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(done, dtype=torch.float32).unsqueeze(-1)
        )


    def refresh(self):
        self.buffer = deque(maxlen=self.capacity)


    def sequential_sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Returns a sequential batch of transitions for temporal analysis."""
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough samples in buffer to sample sequentially.")

        start_idx = self.seq_idx if len(self.buffer) >= self.seq_idx + batch_size else self.seq_idx + batch_size % len(self.buffer)
        batch = list(self.buffer)[start_idx:start_idx + batch_size]
        x_t, a_t, r_t, done = zip(*batch)
        self.seq_idx = (self.seq_idx + batch_size)

        return (
            torch.stack(x_t),
            torch.stack(a_t),
            torch.tensor(r_t, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(done, dtype=torch.float32).unsqueeze(-1)
        )


    def sample_sequences(self, seq_len: int, batch_size: int, device : str = "cpu") -> Tuple[torch.Tensor, ...] | None:
        """Samples sequences of transitions for training."""
        if len(self.buffer) < seq_len:
            return None

        states, actions, rewards, dones = [], [], [], []
        while len(states) < batch_size:
            start_idx = random.randint(0, len(self.buffer) - seq_len)

            s_seq, a_seq, r_seq, d_seq = [], [], [], []
            for i in range(start_idx, start_idx+seq_len):
                s, a, r, done = self.buffer[i]
                if done == 1.0 and i != start_idx + seq_len - 1:
                    break
                s_seq.append(s)
                a_seq.append(a)
                r_seq.append(torch.tensor(r, dtype=torch.float32) if not isinstance(r, torch.Tensor) else r)
                d_seq.append(torch.tensor(done, dtype=torch.float32) if not isinstance(done, torch.Tensor) else done)

            if len(s_seq) == seq_len:
                states.append(torch.stack(s_seq))
                actions.append(torch.stack(a_seq))
                rewards.append(torch.stack(r_seq))
                dones.append(torch.stack(d_seq))

        return (
            torch.stack(states).to(device),  # Shape: (batch_size, seq_len, ...)
            torch.stack(actions).to(device),  # Shape: (batch_size, seq_len, ...)
            torch.stack(rewards).to(device),  # Shape: (batch_size, seq_len, 1)
            torch.stack(dones).to(device)     # Shape: (batch_size, seq_len, 1)
        )

    def __len__(self):
        return len(self.buffer)

# SIGReg parameters
DEFAULT_SIGREG_KNOTS = 17
DEFAULT_SIGREG_NUM_PROJ = 128

class SIGReg(nn.Module):
	
    """Sketch isotropic Gaussian regularizer, adapted from LeWM."""

    def __init__(self, device : str, knots: int = DEFAULT_SIGREG_KNOTS, num_proj: int = DEFAULT_SIGREG_NUM_PROJ):
        super().__init__()
        self.num_proj = num_proj
        self.device = device

        t = torch.linspace(0, 3, knots, dtype=torch.float32, device=device)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32, device=device)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim != 3:
            raise ValueError(f"SIGReg expects a 3D tensor, got shape {tuple(proj.shape)}")

        projections = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        projections = projections / projections.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)

        x_t = (proj @ projections).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(dim=-3) - self.phi).square() + x_t.sin().mean(dim=-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
    
# E2E-JEPA
class E2EJEPA:
    def __init__(
            self,
            env : SnakeEnv,
            encoder: nn.Module,
            predictor: nn.Module,
            projector: nn.Module,
            policy: Policy,
            action_dim: int,
            embed_dim: int,
            optimizer_name : str = "AdamW",
            lr_init : float = 1e-4,
            lr_scheduler : str = "ExponentialLR",
            lr_gamma : float = 0.9,
            buffer_capacity: int = 20000,
            coupled_dynamic = False,
            horizon : int = 1,
            alpha: Regularizer = LinearRegularizer(reg_weight_start=0.1, reg_weight_end=0.1, reg_weight_step=0),
            beta : Regularizer | None = None,
            pol_loss_regularizer: Regularizer = PropToOtherLossChangeRegularizer(),
            device = "cuda"
    ):

        self.env = env

        self.encoder = encoder
        self.predictor = predictor
        self.projector = projector

        self.sigreg = SIGReg(device=device)
        self.policy = policy

        self.action_dim = action_dim

        self.horizon = horizon

        self.alpha = alpha
        self.beta = beta
        self.gamma = pol_loss_regularizer


        assert optimizer_name in ["SGD", "Adam", "AdamW"], f"Unsupported optimizer: {optimizer_name}. Supported are 'SGD', 'Adam', 'AdamW'"
        assert lr_scheduler in ["ExponentialLR", "StepLR"], f"Unsupported scheduler: {lr_scheduler}. Supported are 'ExponentialLR', 'StepLR'"
        if coupled_dynamic:
            self.optimizer = eval(optimizer_name)(
                list(self.encoder.parameters()) +
                list(self.predictor.parameters()) +
                list(self.policy.network.parameters()),
                lr=lr_init
            )
        else:
            self.optimizer = eval(optimizer_name)(
                list(self.encoder.parameters()) +
                list(self.predictor.parameters()),
                lr= lr_init
            )
        self.scheduler = eval(lr_scheduler)(self.optimizer, gamma=lr_gamma)

        # SIGReg projections base vector
        self.buffer = OnlineTrajectoryBuffer(capacity=buffer_capacity)
        self.register_buffer("u_m", F.normalize(torch.randn(embed_dim, 32), p=2, dim=0))


    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)


    def get_action(self, state: torch.Tensor, greedy = False) -> Tuple[torch.Tensor | Any, tuple[Any, Any]]:
        """Phase-Based Exploration: Generates actions on live frames."""
        action, info = self.policy.get_action(state=state, greedy=greedy)
        return action, info


    def encode(self, x_seq):
        """
        Encodes a sequence of frames into latent embeddings.
        x_seq input shape: (Batch, Time, Channels, Height, Width)
        """
        B, T = x_seq.shape[0], x_seq.shape[1]

        # 1. Force the flatten: (B, T, C, H, W) -> (B*T, C, H, W)
        x_flat = x_seq.reshape(B * T, *x_seq.shape[2:])

        # 2. Pass the 4D tensor to the encoder
        # Note: Depending on your encoder, you might need to ensure x_flat is float
        output = self.encoder(x_flat.float())

        # 3. Extract CLS token (assuming index 0) and reshape back to (B, T, Embed_Dim)
        z_flat = output[:, 0, :]

        # 4. Project embeddings
        z_proj = self.projector(z_flat)

        z_seq = z_proj.view(B, T, -1)

        return z_seq


    def predict(self, ctx_emb, ctx_act):
        """Predicts future states autoregressively."""

        # 1. Dimension Fix: Convert discrete integer actions to one-hot floats
        # Transforms (B, T) -> (B, T, action_dim) to match transformer.py's action_proj
        if ctx_act.dim() == 2:
            ctx_act = F.one_hot(ctx_act.long(), num_classes=self.action_dim).float()

        return self.predictor(ctx_emb, ctx_act)


    def compute_trajectory(self, z_t: torch.Tensor, horizon: int=1):
        """Computes a trajectory in the embedded space"""
        z_states, x_next_states, z_next_states, actions, rewards, dones, log_probs, values = [], [], [], [], [], [], [], []

        for _ in range(horizon):
            a_t, (log_prob, value) = self.get_action(state=z_t, greedy=False)
            x_tp1, r_t, done, _, _ = self.env.step(a_t)

            if done:
                x_tp1 = self.env.death_state()

            z_tp1 = self.encode(x_tp1)

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


    def autoregressive_rollout(self, z_seq : torch.Tensor, context_actions : torch.Tensor, future_actions) -> torch.Tensor:
        """
        Takes the states and autoregressively predicts the future ones.
        Future states will have size: [Batch Size; Self.Horizon; Embedding Dimension]
        """
        # Concatenate actions:
        actions = torch.cat([context_actions, future_actions], dim=1)
        history_size = z_seq.size(1)

        predictions = []
        for t in range(self.horizon):
            # Sliding Window
            z_in = z_seq[:, -history_size:] # At the first round, it will be the whole Tensor
            a_in = actions[:, t : t + history_size]

            # Predict
            z_pred = self.predict(z_in, a_in)

            # Keep just the last element and concatenate it to the states' sequence
            z_next = z_pred[:, -1:]
            predictions.append(z_next)
            z_seq = torch.cat([z_seq, z_next], dim=1)

        return torch.cat(predictions, dim=1) 


    def update_parameters(self, batch) -> Dict[str, float]:
        """Samples from active trajectory memory and performs backpropagation."""
        self.encoder.train()
        self.predictor.train()

        x_seq, a_seq, r_seq, done_seq = batch

        z_seq = self.encode(x_seq)

        context_embedding = z_seq[:, :-self.horizon]
        context_action = a_seq[:, :-self.horizon]
        target_embedding = z_seq[:, -self.horizon:].detach()
        target_action = a_seq[:, -self.horizon:]

        if self.horizon > 1:
            prediction_embedding = self.autoregressive_rollout(context_embedding, context_action, target_action)
        else:
            prediction_embedding = self.predict(context_embedding, context_action)[:, -1:, ...]

        # Prediction Loss
        loss_pred = F.mse_loss(prediction_embedding, target_embedding)

        # Anti-Collapse Loss
        loss_sigreg = self.sigreg(context_embedding.transpose(0, 1))

        # Embedding average norm and variance
        avg_norm = context_embedding.norm(dim=-1).mean()
        var = context_embedding.var(dim=-1).mean()

        if isinstance(self.policy.network, (AttentionPPO, ConvPPO)):
            raise NotImplementedError("Policy update for AttentionPPO and ConvPPO is not implemented yet.")
        else:
            # Prepare inputs for policy update
            if isinstance(self.policy.network, DQN):
                z_t_policy = context_embedding[:, -1].detach()      # [B, D]
                z_tp1_policy = target_embedding[:, 0].detach()      # [B, D]
                a_t = a_seq[:, -1].detach()                         # [B, 4] --- 4 because of OneHot
                r_t = r_seq[:, -1].detach()                         # [B]
                done_t = done_seq[:, -1].detach()                   # [B]
            elif isinstance(self.policy.network, AttentionDQN):
                # Attention DQN expects sequences
                # Only predicts one step in the future!
                z_t_policy = context_embedding.detach()             # [B, SeqLen, D]
                z_tp1_policy = torch.cat(
                    (
                        context_embedding[:, 1:],                   # [B, SeqLen-1, D]
                        target_embedding[:, :1]                     # [B, 1, D]
                    ), dim=1).detach()                              # [B, SeqLen, D]
                a_t = context_action.detach()                       # [B, SeqLen, 4]
                r_t = r_seq[:, -1].detach()                         # [B]
                done_t = done_seq[:, -1].detach()                   # [B]

            reg_coeff = self.gamma.step(loss_target = loss_pred.detach() ) if self.gamma else 1.0
            loss_policy = self.policy.update_parameters(init_state = z_t_policy,
                                                        next_state = z_tp1_policy,
                                                        actions = a_t,
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
        # eventually clip:
        # torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {"total_loss": total_loss.item(),
                "pred_loss": loss_pred.item(),
                "policy_loss":  loss_policy,
                "sigreg_loss": loss_sigreg.item(),
                "avg_norm": avg_norm.item(),
                "variance": var.item()
                }

if __name__ == "__main__":
    pass