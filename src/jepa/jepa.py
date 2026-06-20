import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


DEFAULT_OBS_SHAPE = (750, 700, 3)
DEFAULT_NUM_ACTIONS = 4
DEFAULT_LATENT_DIM = 64
DEFAULT_HIDDEN_DIM = 128
DEFAULT_ACTION_EMBED_DIM = 16
DEFAULT_HISTORY_SIZE = 3
DEFAULT_SIGREG_KNOTS = 17
DEFAULT_SIGREG_NUM_PROJ = 128
DEFAULT_PREDICTION_WEIGHT = 1.0
DEFAULT_SIGREG_WEIGHT = 0.05
DEFAULT_ACTOR_WEIGHT = 1.0

class SIGReg(nn.Module):
	"""Sketch isotropic Gaussian regularizer, adapted from LeWM."""

	def __init__(self, knots: int = DEFAULT_SIGREG_KNOTS, num_proj: int = DEFAULT_SIGREG_NUM_PROJ):
		super().__init__()
		self.num_proj = num_proj

		t = torch.linspace(0, 3, knots, dtype=torch.float32)
		dt = 3 / (knots - 1)
		weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
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


class RGBEncoder(nn.Module):
	"""Encoder for Snake RGB images."""

	def __init__(self, obs_shape: tuple[int, int, int] = DEFAULT_OBS_SHAPE, hidden_dim: int = DEFAULT_HIDDEN_DIM, latent_dim: int = DEFAULT_LATENT_DIM):
		super().__init__()
		pass

	def forward(self, obs: torch.Tensor) -> torch.Tensor:
		obs = obs.float()
		
		pass


class ActionEncoder(nn.Module):
	"""Embed discrete actions into a learned action space, equivalent to LeWM Embedder."""

	def __init__(self, num_actions: int = DEFAULT_NUM_ACTIONS, embed_dim: int = DEFAULT_ACTION_EMBED_DIM):
		super().__init__()
		self.num_actions = num_actions
		self.embed = nn.Sequential(
			nn.Linear(num_actions, embed_dim*4),
			nn.SiLU(),
			nn.Linear(embed_dim*4, embed_dim),
		)

	def forward(self, actions: torch.Tensor) -> torch.Tensor:
		actions = actions.float()
		return self.embed(actions)


class LatentPredictor(nn.Module):
	"""Latent predictor."""

	def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM, action_dim: int = DEFAULT_ACTION_EMBED_DIM, hidden_dim: int = DEFAULT_HIDDEN_DIM):
		super().__init__()
		pass

	def forward(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
		pass


class JEPA(nn.Module):
	"""End-to-end JEPA core."""

	def __init__(
		self,
		encoder: nn.Module | None = None,
		predictor: nn.Module | None = None,
		action_encoder: nn.Module | None = None,
		actor: nn.Module | None = None,
		projector: nn.Module | None = None,
		pred_proj: nn.Module | None = None,
		obs_shape: tuple[int, int] = DEFAULT_OBS_SHAPE,
		num_actions: int = DEFAULT_NUM_ACTIONS,
		latent_dim: int = DEFAULT_LATENT_DIM,
		hidden_dim: int = DEFAULT_HIDDEN_DIM,
		action_embed_dim: int = DEFAULT_ACTION_EMBED_DIM,
		history_size: int = DEFAULT_HISTORY_SIZE,
		prediction_weight: float = DEFAULT_PREDICTION_WEIGHT,
		sigreg_weight: float = DEFAULT_SIGREG_WEIGHT,
		actor_weight: float = DEFAULT_ACTOR_WEIGHT,
		sigreg: SIGReg | None = None,
	):
		super().__init__()

		self.obs_shape = obs_shape
		self.num_actions = num_actions
		self.latent_dim = latent_dim
		self.history_size = history_size
		self.prediction_weight = prediction_weight
		self.sigreg_weight = sigreg_weight
		self.actor_weight = actor_weight

		self.encoder = encoder or RGBEncoder(obs_shape=obs_shape, hidden_dim=hidden_dim, latent_dim=latent_dim)
		self.action_encoder = action_encoder or ActionEncoder(num_actions=num_actions, embed_dim=action_embed_dim)
		self.predictor = predictor or LatentPredictor(latent_dim=latent_dim, action_dim=action_embed_dim, hidden_dim=hidden_dim)
		self.actor = actor or Actor(latent_dim=latent_dim, num_actions=num_actions, hidden_dim=hidden_dim)
		self.projector = projector or nn.Identity()
		self.pred_proj = pred_proj or nn.Identity()
		self.sigreg = sigreg or SIGReg()

	def encode(self, info):
		"""Encode observations and actions into embeddings."""

		obs = info["pixels"].float()

		info["emb"] = self.projector(self.encoder(obs))

		if "action" in info:
			info["act_emb"] = self.action_encoder(info["action"])

		return info

	def predict(self, emb, act_emb):
		"""Predict next latent states."""

		return self.pred_proj(self.predictor(emb, act_emb))

	def rollout(self, info, action_sequence, history_size: int | None = None):
		"""Roll out latent predictions for a set of candidate action plans."""

		pass

	def criterion(self, info_dict):
		"""Compute the cost between predicted embeddings and goal embeddings."""

		pred_emb = info_dict["predicted_emb"]
		goal_emb = info_dict["goal_emb"]

		goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)
		cost = F.mse_loss(
			pred_emb[..., -1:, :],
			goal_emb[..., -1:, :].detach(),
			reduction="none",
		).sum(dim=tuple(range(2, pred_emb.ndim)))
		return cost

	def get_cost(self, info_dict, action_candidates):
		""" Compute the cost of action candidates given an info dict with goal and initial state."""

		pass

	def losses(self, info, target_action: torch.Tensor | None = None, history_size: int | None = None):
		"""Compute online losses."""

		encoded = self.encode(info)
		emb = encoded["emb"]
		act_emb = encoded["act_emb"]

		ctx_len = history_size or self.history_size
		ctx_len = min(ctx_len, emb.size(1) - 1)

		pred_emb = self.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
		target_emb = emb[:, 1 : ctx_len + 1]

		pred_loss = F.mse_loss(pred_emb, target_emb)
		sigreg_loss = self.sigreg(emb)

		losses = {
			"pred_loss": pred_loss,
			"sigreg_loss": sigreg_loss,
			"loss": self.prediction_weight * pred_loss + self.sigreg_weight * sigreg_loss,
		}

		if target_action is not None:
			actor_loss = 0 # something that depends on actor and policy
			losses["actor_loss"] = actor_loss
			losses["loss"] = losses["loss"] + self.actor_weight * actor_loss

		return losses