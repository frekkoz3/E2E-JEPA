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

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module"""

    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
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

    def forward(self, x, c = None):
        if self.do_adaLN_modulation:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(c).chunk(6, dim=-1)
            )

            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        num_heads,
        mlp_dim,
        cond_dim=1,
        dropout=0.0,
        use_adaLN=False,
        no_last_layer_norm=False
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim) if not no_last_layer_norm else nn.Identity()
        
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)

        if cond_dim != hidden_dim:
            self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        if hidden_dim != output_dim:
            self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(hidden_dim, num_heads, mlp_dim, dropout, use_adaLN)
            for _ in range(depth)])

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)
            c = c[:, None, :]

        for block in self.layers:
            x = block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
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