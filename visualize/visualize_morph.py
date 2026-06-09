"""CFG morphing visualization: circle -> star.

eps_guided = eps_circle + w * (eps_star - eps_circle)

w=0 -> pure circle, w=1 -> pure star, intermediate -> morph.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

from dataset import ShapeDataset
from model import DiffusionMLP, make_noise_schedule, q_sample

# ── Load model ────────────────────────────────────────────────────
runs      = sorted(Path("outputs").glob("*/model.pt"))
MODEL_PATH = runs[-1]
cfg_data  = json.loads((MODEL_PATH.parent / "config.json").read_text())
print(f"model: {MODEL_PATH}")

T        = cfg_data["T"]
SCHEDULE = cfg_data.get("schedule", "cosine")
K        = cfg_data["num_clusters"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DiffusionMLP(
    hidden_dim   = cfg_data["hidden_dim"],
    num_layers   = cfg_data["num_layers"],
    num_clusters = K,
    emb_dim      = cfg_data["emb_dim"],
    dropout      = 0.0,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

betas, alphas, alpha_bars, _, _ = make_noise_schedule(T, schedule=SCHEDULE, device=device)

# ── Class indices ─────────────────────────────────────────────────
NAMES    = ShapeDataset.SHAPE_NAMES          # ["circle", "square", "triangle", "star"]
CLS_FROM = NAMES.index("circle")             # 0
CLS_TO   = NAMES.index("star")              # 3

W_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6]
N        = 400

# ── Morph sampler ─────────────────────────────────────────────────
@torch.no_grad()
def p_sample_morph(x_t, t_idx, c_from, c_to, w):
    B = x_t.shape[0]
    t = torch.full((B,), t_idx, device=device, dtype=torch.long)

    eps_from = model(x_t, t, c_from)
    eps_to   = model(x_t, t, c_to)
    eps      = eps_from + w * (eps_to - eps_from)

    alpha_t   = alphas[t_idx]
    alpha_bar = alpha_bars[t_idx]
    beta_t    = betas[t_idx]

    coef = (1.0 - alpha_t) / (1.0 - alpha_bar).sqrt()
    mean = (x_t - coef * eps) / alpha_t.sqrt()
    if t_idx == 0:
        return mean
    return mean + beta_t.sqrt() * torch.randn_like(x_t)


@torch.no_grad()
def sample_morph(w, x_start):
    c_from = torch.full((N,), CLS_FROM, device=device, dtype=torch.long)
    c_to   = torch.full((N,), CLS_TO,   device=device, dtype=torch.long)
    x = x_start.clone()
    for t_idx in reversed(range(T)):
        x = p_sample_morph(x, t_idx, c_from, c_to, w)
    return x.cpu().numpy()


# ── Fixed starting noise ──────────────────────────────────────────
torch.manual_seed(0)
x_start = torch.randn(N, 2, device=device)

print("Sampling morphs...")
results = {}
for w in W_VALUES:
    print(f"  w={w:.1f}")
    results[w] = sample_morph(w, x_start)

# ── Plot ──────────────────────────────────────────────────────────
n_cols = len(W_VALUES)
fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.5))
fig.suptitle(f"CFG Morph: {NAMES[CLS_FROM]} -> {NAMES[CLS_TO]}", fontsize=13)

# Color: interpolate between circle color and star color
TAB10     = plt.cm.tab10.colors
col_from  = np.array(TAB10[CLS_FROM][:3])
col_to    = np.array(TAB10[CLS_TO][:3])

for ax, w in zip(axes, W_VALUES):
    pts   = results[w]
    alpha = min(w, 1.0)
    color = tuple((1 - alpha) * col_from + alpha * col_to)

    ax.scatter(pts[:, 0], pts[:, 1], color=color, s=4, alpha=0.65, linewidths=0)
    ax.add_patch(mpatches.Rectangle((-1, -1), 2, 2,
                 fill=False, edgecolor="#cccccc", lw=0.7, ls="--"))
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_title(f"w = {w}", fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")

# Annotate endpoints
axes[0].set_title(f"w = 0.0\n(circle)", fontsize=10)
axes[-1].set_title(f"w = {W_VALUES[-1]}\n(star++)", fontsize=10)

plt.tight_layout()
out = "morph_visualization.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.show()
