from collections.abc import Sequence

import torch
import torch.nn as nn

from .mlp_denoiser import SinusoidalPosEmb


class LearnedMeanPool(nn.Module):
    """mean(h + MLP(h)) pooling for variable-size point sets."""

    def __init__(self, dim: int, dropout: float = 0.0, mlp_mult: int = 2):
        super().__init__()
        inner_dim = dim * mlp_mult
        self.proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled = h + self.proj(h)
        if mask is None:
            return pooled.mean(dim=1)

        weights = mask.to(dtype=pooled.dtype, device=pooled.device).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (pooled * weights).sum(dim=1) / denom


class PointNetFiLMBlock(nn.Module):
    """Point-wise residual MLP modulated by a global condition vector."""

    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0, mlp_mult: int = 4):
        super().__init__()
        inner_dim = dim * mlp_mult
        self.norm = nn.LayerNorm(dim)
        self.cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 3),
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift, gate = self.cond(cond).chunk(3, dim=-1)
        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)
        gate = gate.unsqueeze(1)
        x = self.norm(h) * (1.0 + scale) + shift
        return h + torch.sigmoid(gate) * self.mlp(x)


class PointNetDenoiser(nn.Module):
    """PointNet-style DDPM denoiser for point sets.

    Inputs:
        x:    (B, N, C) noisy point cloud
        t:    (B,) timestep indices
        c:    (B,) class labels
        mask: optional (B, N) valid-point mask

    Output:
        (B, N, C) predicted noise.

    The model has no parameters tied to N, so each minibatch may use a
    different point count as long as all samples in that minibatch share shape.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_classes: int = 1,
        emb_dim: int = 128,
        dropout: float = 0.0,
        mlp_mult: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.null_class = num_classes

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(emb_dim),
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.class_emb = nn.Embedding(num_classes + 1, hidden_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.pool = LearnedMeanPool(hidden_dim, dropout=dropout)
        self.global_mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [
                PointNetFiLMBlock(
                    hidden_dim,
                    cond_dim=hidden_dim,
                    dropout=dropout,
                    mlp_mult=mlp_mult,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        c: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x with shape (B, N, C), got {tuple(x.shape)}")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {x.shape[-1]}")

        h = self.input_proj(x)
        base_cond = self.time_mlp(t) + self.class_emb(c)

        for layer in self.layers:
            global_feat = self.pool(h, mask=mask)
            cond = base_cond + self.global_mlp(global_feat)
            h = layer(h, cond)

        return self.out_proj(self.out_norm(h))


def q_sample_point_cloud(
    x0: torch.Tensor,
    t: torch.Tensor,
    sqrt_alphas_cumprod: torch.Tensor,
    sqrt_one_minus_alphas_cumprod: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample x_t for tensors shaped (B, N, C) or any (B, ...)."""
    if noise is None:
        noise = torch.randn_like(x0)

    view_shape = (x0.shape[0],) + (1,) * (x0.ndim - 1)
    sqrt_a = sqrt_alphas_cumprod[t].view(view_shape)
    sqrt_1a = sqrt_one_minus_alphas_cumprod[t].view(view_shape)
    return sqrt_a * x0 + sqrt_1a * noise, noise


def resample_points(
    points: torch.Tensor,
    n_points: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample exactly n_points from one point cloud, with replacement if needed."""
    if points.ndim != 2:
        raise ValueError(f"expected one point cloud shaped (M, C), got {tuple(points.shape)}")

    total = points.shape[0]
    if total == n_points:
        return points
    if total > n_points:
        idx = torch.randperm(total, generator=generator, device=points.device)[:n_points]
    else:
        idx = torch.randint(total, (n_points,), generator=generator, device=points.device)
    return points[idx]


class VariablePointCountCollate:
    """Collate point clouds with one randomly chosen N per minibatch.

    Each dataset item should be either a point tensor shaped (M, C), or a
    (points, label) tuple. The returned point tensor is shaped (B, N, C).
    """

    def __init__(
        self,
        n_choices: Sequence[int] = (512, 1024, 2048),
        generator: torch.Generator | None = None,
    ):
        if not n_choices:
            raise ValueError("n_choices must contain at least one point count")
        self.n_choices = tuple(int(n) for n in n_choices)
        self.generator = generator

    def __call__(self, batch):
        choice_idx = torch.randint(
            len(self.n_choices),
            (1,),
            generator=self.generator,
        ).item()
        n_points = self.n_choices[choice_idx]

        has_labels = isinstance(batch[0], tuple)
        if has_labels:
            clouds, labels = zip(*batch)
        else:
            clouds, labels = batch, None

        x = torch.stack(
            [resample_points(cloud, n_points, generator=self.generator) for cloud in clouds],
            dim=0,
        )

        if labels is None:
            return x
        return x, torch.as_tensor(labels, dtype=torch.long)
