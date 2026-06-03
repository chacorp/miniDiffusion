import math
import torch
import torch.nn as nn


# ─── Building blocks ──────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Timestep → sinusoidal embedding (Vaswani et al.)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class ConditionedResidualBlock(nn.Module):
    """Pre-LN residual block with cross-attention conditioning.

    Step 1 — MLP update (self):   h = h + MLP(LN(h))
    Step 2 — Attention update:    stack [h, c_emb] as 2-token sequence,
                                  run self-attention, add h's output token back.
                                  h can attend to c_emb so conditioning is
                                  content-dependent, not just additive.
    """

    def __init__(self, dim: int, dropout: float, num_heads: int = 4):
        super().__init__()
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.attn_norm_h = nn.LayerNorm(dim)
        self.attn_norm_c = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, h: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        # MLP residual
        h = h + self.mlp(self.mlp_norm(h))
        # Attention: [h, c_emb] as 2-token sequence; only h's output is used
        tokens = torch.stack([self.attn_norm_h(h), self.attn_norm_c(c_emb)], dim=1)  # (B, 2, dim)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        h = h + attn_out[:, 0]  # residual on h's token only
        return h


# ─── Diffusion model ──────────────────────────────────────────────

class DiffusionMLP(nn.Module):
    """Noise-prediction network for conditioned DDPM on 2-D points.

    Inputs:
        x: (B, 2)   noisy point at timestep t
        t: (B,)     timestep indices
        c: (B,)     cluster label indices
    Output:
        (B, 2)  predicted noise ε
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_clusters: int = 5,
        emb_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Timestep embedding: sinusoidal → 2-layer MLP → hidden_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cluster condition embedding
        self.cluster_emb = nn.Embedding(num_clusters, hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(2, hidden_dim)

        # Residual hidden layers with attention-based cluster conditioning
        self.layers = nn.ModuleList(
            [ConditionedResidualBlock(hidden_dim, dropout) for _ in range(num_layers)]
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.time_mlp(t)
        c_emb = self.cluster_emb(c)          # (B, hidden_dim) — looked up once, passed to each layer
        for layer in self.layers:
            h = layer(h, c_emb)
        return self.out_proj(h)


# ─── Noise schedule ───────────────────────────────────────────────

def make_noise_schedule(
    T: int = 1000,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    device=None,
) -> tuple[torch.Tensor, ...]:
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = alphas_cumprod.sqrt()
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod).sqrt()
    return betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod


# ─── Forward process q ────────────────────────────────────────────

def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x_t ~ q(x_t | x_0) via the closed-form marginal."""
    if noise is None:
        noise = torch.randn_like(x0)
    B = x0.shape[0]
    sqrt_a  = sqrt_alphas_cumprod[t].view(B, 1)
    sqrt_1a = sqrt_one_minus_alphas_cumprod[t].view(B, 1)
    x_t = sqrt_a * x0 + sqrt_1a * noise
    return x_t, noise


# ─── Reverse process p ────────────────────────────────────────────

@torch.no_grad()
def p_sample(
    model: DiffusionMLP,
    x_t: torch.Tensor,
    t_idx: int,
    c: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """One DDPM reverse step: x_{t-1} ~ p_θ(x_{t-1} | x_t, c)."""
    B = x_t.shape[0]
    t = torch.full((B,), t_idx, device=x_t.device, dtype=torch.long)

    eps_pred  = model(x_t, t, c)
    alpha_t   = alphas[t_idx]
    alpha_bar = alphas_cumprod[t_idx]
    beta_t    = betas[t_idx]

    # DDPM posterior mean: μ_θ = 1/√αt * (x_t − (1−αt)/√(1−ᾱt) * ε_θ)
    coef = (1.0 - alpha_t) / (1.0 - alpha_bar).sqrt()
    mean = (x_t - coef * eps_pred) / alpha_t.sqrt()

    if t_idx == 0:
        return mean

    return mean + beta_t.sqrt() * torch.randn_like(x_t)


@torch.no_grad()
def p_sample_loop(
    model: DiffusionMLP,
    c: torch.Tensor,
    T: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> torch.Tensor:
    """Full reverse chain x_T → x_0 conditioned on cluster labels c."""
    x = torch.randn(c.shape[0], 2, device=c.device)
    for t_idx in reversed(range(T)):
        x = p_sample(model, x, t_idx, c, betas, alphas, alphas_cumprod)
    return x
