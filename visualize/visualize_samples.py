"""Generated sample visualization: one subplot per class label."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from dataset import ShapeDataset
from model import DiffusionMLP, make_noise_schedule, p_sample_loop

# ── Load latest checkpoint ────────────────────────────────────────
runs = sorted(Path("outputs").glob("*/model.pt"))
if not runs:
    raise FileNotFoundError("No model.pt found in outputs/")

MODEL_PATH = runs[-1]
cfg_data   = json.loads((MODEL_PATH.parent / "config.json").read_text())
print(f"model  : {MODEL_PATH}")
print(f"config : {cfg_data}")

T            = cfg_data["T"]
NUM_CLUSTERS = cfg_data["num_clusters"]
SCHEDULE     = cfg_data.get("schedule", "cosine")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DiffusionMLP(
    hidden_dim   = cfg_data["hidden_dim"],
    num_layers   = cfg_data["num_layers"],
    num_clusters = NUM_CLUSTERS,
    emb_dim      = cfg_data["emb_dim"],
    dropout      = 0.0,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

betas, alphas, alpha_bars, _, _ = make_noise_schedule(T, schedule=SCHEDULE, device=device)

# ── Generate samples ──────────────────────────────────────────────
N_PER_CLASS = 500
SHAPE_NAMES = ShapeDataset.SHAPE_NAMES
TAB10       = plt.cm.tab10.colors

print("Generating samples...")
all_pts = []
with torch.no_grad():
    for cls in range(NUM_CLUSTERS):
        c   = torch.full((N_PER_CLASS,), cls, device=device, dtype=torch.long)
        pts = p_sample_loop(model, c, T, betas, alphas, alpha_bars).cpu().numpy()
        all_pts.append(pts)
        print(f"  {SHAPE_NAMES[cls]:10s}: x=[{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]  "
              f"y=[{pts[:,1].min():.2f}, {pts[:,1].max():.2f}]")

# ── Reference data ────────────────────────────────────────────────
ds     = ShapeDataset(num_samples=NUM_CLUSTERS * N_PER_CLASS, noise=0.01)
ref    = ds.points.numpy()
ref_lb = ds.labels.numpy()

# ── Plot: generated (top) vs reference (bottom) ───────────────────
fig, axes = plt.subplots(2, NUM_CLUSTERS, figsize=(3.5 * NUM_CLUSTERS, 7))
fig.suptitle("Generated (top)  vs  Reference (bottom)", fontsize=12)

for i in range(NUM_CLUSTERS):
    name  = SHAPE_NAMES[i]
    color = TAB10[i % 10]

    for row, pts in enumerate([all_pts[i], ref[ref_lb == i]]):
        ax = axes[row, i]
        ax.scatter(pts[:, 0], pts[:, 1],
                   color=color, s=4, alpha=0.65, linewidths=0)
        ax.add_patch(mpatches.Rectangle((-1, -1), 2, 2,
                     fill=False, edgecolor="#bbbbbb", lw=0.7, ls="--"))
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        if row == 0:
            ax.set_title(name, fontsize=11, color=color, fontweight="bold")

axes[0, 0].set_ylabel("generated", fontsize=9)
axes[1, 0].set_ylabel("reference", fontsize=9)

plt.tight_layout()
out = "sample_visualization.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.show()
