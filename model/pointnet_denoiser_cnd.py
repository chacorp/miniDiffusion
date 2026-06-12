"""PointNet denoiser with cross-attention conditioning from a reference point cloud.

Architecture per block:
    Self-Attn(h) → Cross-Attn(Q=h, K/V=cond_tokens) → FiLM(t) → FFN

Training: self-reconstruction — condition = clean sample of the same shape.
Inference: condition = any reference PC (generalizes to unseen classes).
CFG: drop condition → null_token for unconditional path.
"""
import torch
import torch.nn as nn
import numpy as np

from .mlp_denoiser import SinusoidalPosEmb, make_noise_schedule


# ─── Condition Encoder ────────────────────────────────────────────────────────

class PointNetEncoder(nn.Module):
    """Point-wise MLP encoder for the condition point cloud.

    Args:
        in_dim:     input coordinate dimension
        hidden_dim: intermediate MLP width
        out_dim:    token dimension D

    forward(x):
        x   -> [B, M, in_dim]
        out -> [B, M, out_dim]
    """

    def __init__(self, in_dim: int = 2, hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, M, in_dim] -> [B, M, out_dim]"""
        return self.net(x)


# ─── Denoiser Block ───────────────────────────────────────────────────────────

class CrossAttentionBlock(nn.Module):
    """Single denoiser layer: Self-Attn → Cross-Attn → FiLM(t) → FFN.

    All sub-layers use pre-norm + residual connection.

    Args:
        dim:       hidden dimension D
        num_heads: attention heads
        t_dim:     timestep embedding dimension
        dropout:   dropout rate

    forward(h, cond_tokens, t_emb):
        h           -> [B, N, D]   noisy point features
        cond_tokens -> [B, M, D]   encoded condition tokens
        t_emb       -> [B, t_dim]  timestep embedding
        out         -> [B, N, D]
    """

    def __init__(self, dim: int, num_heads: int, t_dim: int, dropout: float = 0.0):
        super().__init__()
        # Self-attention
        self.norm1     = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # Cross-attention
        self.norm2      = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        # FiLM from timestep
        self.film_proj = nn.Linear(t_dim, dim * 2)

        # FFN
        self.norm4 = nn.LayerNorm(dim)
        self.ffn   = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        h: torch.Tensor,           # [B, N, D]
        cond_tokens: torch.Tensor, # [B, M, D]
        t_emb: torch.Tensor,       # [B, t_dim]
    ) -> torch.Tensor:             # [B, N, D]

        # 1. Self-attention: h attends to itself
        h_n = self.norm1(h)
        h = h + self.self_attn(h_n, h_n, h_n, need_weights=False)[0]

        # 2. Cross-attention: h attends to condition tokens
        h = h + self.cross_attn(self.norm2(h), cond_tokens, cond_tokens,
                                 need_weights=False)[0]

        # 3. FiLM: timestep-conditioned scale/shift
        scale, shift = self.film_proj(t_emb).chunk(2, dim=-1)  # [B, D] each
        h = h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # broadcast over N

        # 4. FFN
        h = h + self.ffn(self.norm4(h))

        return h


# ─── Main Model ───────────────────────────────────────────────────────────────

class PointNetDenoiserCnd(nn.Module):
    """Point cloud denoiser conditioned on a reference point cloud via cross-attention.

    Replaces discrete class_emb with a PointNet encoder of a condition PC.
    Training: self-reconstruction (condition = clean GT of same shape).
    Inference: any reference PC as condition; generalizes to unseen classes.

    Args:
        input_dim:  coordinate dimension C (default 2)
        hidden_dim: transformer hidden dim D
        num_layers: number of CrossAttentionBlock layers
        emb_dim:    timestep embedding dim
        num_heads:  attention heads
        dropout:    dropout rate

    forward(x, t, cond_tokens):
        x           -> [B, N, C]   noisy input points
        t           -> [B]         timestep indices
        cond_tokens -> [B, M, D]   encoded condition (from encode_cond or get_null_tokens)
        out         -> [B, N, C]   predicted noise ε
    """

    def __init__(
        self,
        input_dim:  int   = 2,
        hidden_dim: int   = 256,
        num_layers: int   = 6,
        emb_dim:    int   = 128,
        num_heads:  int   = 4,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        # Timestep embedding: [B] -> [B, emb_dim]
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )

        # Condition encoder: [B, M, C] -> [B, M, D]
        self.encoder = PointNetEncoder(input_dim, hidden_dim // 2, hidden_dim)

        # Learned null token for CFG unconditional path: [1, 1, D]
        self.null_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Noisy input projection: [B, N, C] -> [B, N, D]
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Denoiser stack
        self.layers = nn.ModuleList([
            CrossAttentionBlock(hidden_dim, num_heads, emb_dim, dropout)
            for _ in range(num_layers)
        ])

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def get_null_tokens(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Null conditioning token for unconditional CFG path.

        Returns [B, 1, D] (single learned token broadcast over batch).
        """
        return self.null_token.expand(batch_size, -1, -1).to(device)

    def encode_cond(self, cond_pc: torch.Tensor) -> torch.Tensor:
        """Encode condition point cloud.

        cond_pc -> [B, M, C]
        out     -> [B, M, D]
        """
        return self.encoder(cond_pc)

    def forward(
        self,
        x:           torch.Tensor,  # [B, N, C]
        t:           torch.Tensor,  # [B]
        cond_tokens: torch.Tensor,  # [B, M, D]
    ) -> torch.Tensor:              # [B, N, C]

        t_emb = self.time_mlp(t)    # [B, emb_dim]
        h = self.input_proj(x)      # [B, N, D]

        for layer in self.layers:
            h = layer(h, cond_tokens, t_emb)

        return self.out_proj(self.out_norm(h))  # [B, N, C]


# ─── Training Collate ─────────────────────────────────────────────────────────

class SelfReconCollate:
    """Collate for self-reconstruction training with JsonPointShapeDataset.

    Samples x (noisy target) and cond (condition) as independent draws
    from the same shape point pool, so the model learns to reconstruct
    any sub-sample of a shape given another sub-sample as reference.

    Args:
        n_points:   number of target points N
        cond_points: number of condition points M

    __call__(batch) where batch = [(cloud [K, C], label), ...]:
        x    -> [B, N, C]  target point cloud
        cond -> [B, M, C]  condition point cloud (same shape, different sample)
        lbl  -> [B]        class label (for logging only)
    """

    def __init__(self, n_points: int = 512, cond_points: int = 256):
        self.n = n_points
        self.m = cond_points

    def __call__(self, batch):
        xs, conds, labels = [], [], []
        for cloud, label in batch:
            K = cloud.shape[0]
            xs.append(cloud[torch.randperm(K)[:self.n]])
            conds.append(cloud[torch.randperm(K)[:self.m]])
            labels.append(label)
        return (
            torch.stack(xs),    # [B, N, C]
            torch.stack(conds), # [B, M, C]
            torch.tensor(labels, dtype=torch.long),
        )


# ─── Sampling ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def ddim_sample_cnd(
    model:       PointNetDenoiserCnd,
    cond_pc:     torch.Tensor,        # [B, M, C] condition point cloud
    n_points:    int,
    alpha_bars:  torch.Tensor,
    T:           int,
    n_steps:     int   = 50,
    w:           float = 1.5,
    t_start:     int   = None,        # None = full generation; int = SDEdit start
    x_start:     torch.Tensor = None, # [B, N, C] for SDEdit; None = random noise
) -> torch.Tensor:                    # [B, N, C]
    """DDIM sampler with CFG for PointNetDenoiserCnd.

    cond_pc    -> [B, M, C]   reference shape
    n_points   -> N           number of output points
    alpha_bars -> [T]         noise schedule
    t_start    -> int | None  SDEdit noise level; None for full generation
    x_start    -> [B, N, C]   clean input for SDEdit; None for generation
    out        -> [B, N, C]   generated / reconstructed point cloud
    """
    B      = cond_pc.shape[0]
    device = cond_pc.device
    t_max  = (t_start if t_start is not None else T - 1)

    # Encode condition once; null tokens for unconditional path
    cond_tokens = model.encode_cond(cond_pc)            # [B, M, D]
    null_tokens = model.get_null_tokens(B, device)      # [B, 1, D]

    if x_start is not None:
        # SDEdit: add noise to x_start at t_max
        ab = alpha_bars[t_max]
        x  = ab.sqrt() * x_start + (1 - ab).sqrt() * torch.randn_like(x_start)
    else:
        x = torch.randn(B, n_points, model.input_dim, device=device)

    seq = np.linspace(t_max, 0, n_steps + 1).round().astype(int)

    for i in range(n_steps):
        t_cur, t_prev = int(seq[i]), int(seq[i + 1])
        t_vec = torch.full((B,), t_cur, dtype=torch.long, device=device)

        eps_u = model(x, t_vec, null_tokens)
        eps_c = model(x, t_vec, cond_tokens)
        eps   = eps_u + w * (eps_c - eps_u)

        ab_t = alpha_bars[t_cur]
        x0   = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-2.0, 2.0)

        if t_prev <= 0:
            x = x0
        else:
            ab_p = alpha_bars[t_prev]
            x    = ab_p.sqrt() * x0 + (1 - ab_p).sqrt() * eps

    return x  # [B, N, C]
