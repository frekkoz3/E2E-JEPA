r"""
  _____ ____  _____          _ _____ ____   _    
 | ____|___ \| ____|        | | ____|  _ \ / \   
 |  _|   __) |  _| _____ _  | |  _| | |_) / _ \  
 | |___ / __/| |__|_____| |_| | |___|  __/ ___ \ 
 |_____|_____|_____|     \___/|_____|_| /_/   \_\
"""
import torch
import torch.nn.functional as F
from torch import nn

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

# ---------------------------------- #
# CHANGE 1: Add causal Mask
# ---------------------------------- #
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module"""

    def __init__(self,
                 dim : int,
                 num_heads :  int,
                 dropout : float | int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x, causal : bool = False):
        if causal:
            # Apply a lower-traingular causal mask
            T = x.size(1)
            attention_mask = nn.Transformer.generate_square_subsequent_mask(T, device = x.device, dtype=x.dtype)
            return self.attn(x, x, x, attn_mask=attention_mask, is_causal=True)[0]
        return self.attn(x, x, x)[0]


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, in_channels=3, embed_dim=768, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, C, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding"""

    def __init__(self, embed_dim, seq_len):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len+1, embed_dim))

    def forward(self, x):
        return x + self.pos_embedding

# ---------------------------------- #
# CHANGE 2: Add causal parameter
# ---------------------------------- #
class TransformerEncoderBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, num_heads, mlp_dim, dropout=0.0, adaLN_modulation=False):
        super().__init__()

        self.do_adaLN_modulation = adaLN_modulation

        self.attn = MultiHeadAttention(dim, num_heads=num_heads, dropout=dropout)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        if adaLN_modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
            )

            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
    # ---------------------------------- #
    # CHANGE 2: Add causal parameter
    # ---------------------------------- #
    def forward(self, x, c = None, causal : bool = False):
        if self.do_adaLN_modulation:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=-1)
            )

            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), causal = causal)
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        else:
            x = x + self.attn(self.norm1(x), causal = causal)
            x = x + self.mlp(self.norm2(x))
        
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim : int,
        hidden_dim : int,
        output_dim : int,
        depth : int,
        num_heads : int,
        mlp_dim : int ,
        cond_dim : int= 1,
        dropout : float = 0.0,
        use_adaLN : bool = False,
        no_last_layer_norm=False
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim) if not no_last_layer_norm else nn.Identity()

        self.norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim != hidden_dim else nn.Identity()
        self.output_proj = nn.Linear(hidden_dim, output_dim) if hidden_dim != output_dim else nn.Identity()

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads, mlp_dim, dropout, use_adaLN)
            for _ in range(depth)])

    def forward(self, x, c=None, causal = False):

        x = self.input_proj(x)

        if c is not None:
            if c.dim() == 2:
                c = c.unsqueeze(1).expand(-1, x.size(1), -1)  # Expand to match sequence length
            c = self.cond_proj(c)
            # c = c[:, None, :]

        for block in self.layers:
            x = block(x, c, causal = causal)

        x = self.norm(x)
        x = self.output_proj(x)
        return x
    
class VisualTransformer(nn.Module):
    """Visual encoder with patch embedding and transformer backbone."""

    def __init__(self, img_size, embed_dim, mlp_dim, patch_size=16, num_heads=8, depth=6, no_last_layer_norm=False):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels=3, embed_dim=embed_dim, patch_size=patch_size)
        self.pos_embed = PositionalEncoding(embed_dim=embed_dim, seq_len=(img_size[0]//patch_size)*(img_size[1]//patch_size))
        self.transformer = Transformer(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_dim=mlp_dim, no_last_layer_norm=no_last_layer_norm)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(obs)
        cls_tokens = self.cls_token.expand(obs.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_embed(x)
        x = self.transformer(x)
        return x


class Predictor(nn.Module):
    """Autoregressive predictor for sequential JEPA training.

    Given a history of N frame embeddings and N actions, predicts the next N
    frame embeddings using temporal causal masking:

        output[:, t]  ≈  z_{t+1},  conditioned on  z_{0..t}  and  a_{0..t}.

    A single forward pass thus produces N supervised prediction targets,
    compared to 1 in the original single-step design.

    Direct equivalent of module.py::ARPredictor:
        self.pos_embedding  ←→  ARPredictor.pos_embedding
        self.action_proj    ←→  Embedder (action conditioning)
        self.transformer    ←→  Transformer(block_class=ConditionalBlock)
        causal=True         ←→  Attention(is_causal=True)

    REF: module.py::ARPredictor
    REF: module.py::Embedder
    REF: train.py::lejepa_forward — pred_emb = model.predict(ctx_emb, ctx_act)
    Paper: "The predictor takes as input a history of N frame representations
            and predicts the next frame representation auto-regressively with
            temporal causal masking to avoid looking at future embeddings."
    """

    def __init__(
            self,
            embed_dim: int,
            hidden_dim: int,
            action_dim: int,
            depth: int,
            num_heads: int,
            mlp_dim: int,
            max_seq_len: int = 16,
            dropout: float = 0.0,
            use_adaLN: bool = True,
    ):
        super().__init__()

        # ── Temporal positional embedding ──────────────────────────────────
        # REF: module.py::ARPredictor.pos_embedding — nn.Parameter shape (1, T, D)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.dropout = nn.Dropout(dropout)

        # ── Action projection ──────────────────────────────────────────────
        # Maps (B, T, action_dim) → (B, T, hidden_dim) so conditioning has
        # the same dimension as the hidden state (fed as `c` to AdaLN blocks).
        # REF: module.py::Embedder — Conv1d + MLP action embedding.
        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # ── Causal transformer core ────────────────────────────────────────
        # cond_dim=hidden_dim because action_proj already maps to hidden_dim.
        # REF: module.py::ARPredictor — uses ConditionalBlock (AdaLN-zero).
        self.transformer = Transformer(
            input_dim=embed_dim,
            hidden_dim=hidden_dim,
            output_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            cond_dim=hidden_dim,
            dropout=dropout,
            use_adaLN=use_adaLN,
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, embed_dim)   — history of N frame embeddings
        a : (B, T, action_dim)  — corresponding N actions (e.g. one-hot)

        Returns (B, T, embed_dim) where output[:, t] predicts z_{t+1},
        using only x[:, 0..t] and a[:, 0..t] (causal masking enforced).

        REF: module.py::ARPredictor.forward — identical (x, c) interface.
        REF: train.py::lejepa_forward — pred_emb = model.predict(ctx_emb, ctx_act)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]   # add temporal positional encoding
        x = self.dropout(x)
        c = self.action_proj(a)              # (B, T, hidden_dim)
        return self.transformer(x, c, causal=True)