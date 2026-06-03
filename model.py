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
    """Pre-LN residual block with FiLM-style cluster conditioning.

    The cluster embedding predicts scale, shift, and gate vectors that modulate
    the normalized hidden state before the MLP update.
    """

    def __init__(self, dim: int, dropout: float, mlp_mult: int = 4):
        super().__init__()
        inner_dim = dim * mlp_mult

        self.norm = nn.LayerNorm(dim)
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 3),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        scale, shift, gate = self.cond(c_emb).chunk(3, dim=-1)
        x = self.norm(h) * (1.0 + scale) + shift
        return h + torch.sigmoid(gate) * self.mlp(x)


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

        # Cluster condition embedding (index num_clusters = null/unconditional token)
        self.null_class  = num_clusters
        self.cluster_emb = nn.Embedding(num_clusters + 1, hidden_dim)

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


# ─── Classifier-free guidance ─────────────────────────────────────

@torch.no_grad()
def p_sample_cfg(
    model: DiffusionMLP,
    x_t: torch.Tensor,
    t_idx: int,
    c: torch.Tensor,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    w: float = 1.0,
    cfg_trained: bool = True,
) -> torch.Tensor:
    """DDPM reverse step with classifier-free guidance.

    w=1: standard conditional (same as p_sample).
    w>1: amplifies class signal; w=0: unconditional only.

    cfg_trained=True  → null class embedding (requires retrain with p_uncond > 0).
    cfg_trained=False → class-average as unconditional proxy; works with any checkpoint
                        because ε_uncond ≈ E_c[ε_θ(x_t,t,c)] marginalises over all classes.
    """
    B = x_t.shape[0]
    t = torch.full((B,), t_idx, device=x_t.device, dtype=torch.long)

    eps_cond = model(x_t, t, c)
    if w != 1.0:
        if cfg_trained:
            null_c     = torch.full((B,), model.null_class, device=x_t.device, dtype=torch.long)
            eps_uncond = model(x_t, t, null_c)
        else:
            # ε_uncond ≈ (1/K) Σ_k ε_θ(x_t, t, k)  — single batched forward pass
            K     = model.null_class   # number of real classes
            x_rep = x_t.repeat(K, 1)  # (K*B, 2)
            t_rep = t.repeat(K)        # (K*B,)
            c_rep = torch.cat([
                torch.full((B,), k, device=x_t.device, dtype=torch.long) for k in range(K)
            ])
            eps_uncond = model(x_rep, t_rep, c_rep).view(K, B, 2).mean(0)
        eps = eps_uncond + w * (eps_cond - eps_uncond)
    else:
        eps = eps_cond

    alpha_t   = alphas[t_idx]
    alpha_bar = alphas_cumprod[t_idx]
    beta_t    = betas[t_idx]

    coef = (1.0 - alpha_t) / (1.0 - alpha_bar).sqrt()
    mean = (x_t - coef * eps) / alpha_t.sqrt()

    if t_idx == 0:
        return mean
    return mean + beta_t.sqrt() * torch.randn_like(x_t)


@torch.no_grad()
def p_sample_loop_with_traj(
    model: DiffusionMLP,
    c: torch.Tensor,
    T: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    w: float = 1.0,
    record_every: int = 50,
    x_start: torch.Tensor | None = None,
    cfg_trained: bool = True,
) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
    """Reverse chain with trajectory recording for visualization.

    Returns (x_0, traj) where traj is a list of (t_idx, positions) snapshots.
    """
    x = (torch.randn(c.shape[0], 2, device=c.device)
         if x_start is None else x_start.clone().to(c.device))

    traj: list[tuple[int, torch.Tensor]] = [(T, x.cpu().clone())]
    for t_idx in reversed(range(T)):
        x = p_sample_cfg(model, x, t_idx, c, betas, alphas, alphas_cumprod,
                         w=w, cfg_trained=cfg_trained)
        if t_idx % record_every == 0:
            traj.append((t_idx, x.cpu().clone()))

    return x, traj
