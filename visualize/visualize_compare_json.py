"""A. 생성 결과 vs 원본 비교 (12 classes)."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset import JsonShapeDataset
from model import DiffusionMLP, make_noise_schedule, p_sample_loop

MODEL_PATH = Path("outputs/20260608_222600/best.pt")
cfg_data   = json.loads((MODEL_PATH.parent / "config.json").read_text())

T = cfg_data["T"]; K = 12; N_PER = 400
device = torch.device("cpu")

model = DiffusionMLP(
    hidden_dim=cfg_data["hidden_dim"], num_layers=cfg_data["num_layers"],
    num_clusters=K, emb_dim=cfg_data["emb_dim"], dropout=0.0,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

betas, alphas, alpha_bars, _, _ = make_noise_schedule(T, schedule=cfg_data["schedule"], device=device)

ds    = JsonShapeDataset(data_dir=cfg_data["data_dir"], num_samples=60000)
names = ds.shape_names
TAB20 = plt.cm.tab20.colors

torch.manual_seed(42)

# Generate samples
gen = []
for cls in range(K):
    c   = torch.full((N_PER,), cls, dtype=torch.long)
    pts = p_sample_loop(model, c, T, betas, alphas, alpha_bars).numpy()
    gen.append(pts)

# Plot: 12 rows × 2 cols (real | generated)
fig, axes = plt.subplots(K, 2, figsize=(5, 2.6 * K))
fig.suptitle("Real (left)  vs  Generated (right)", fontsize=12, y=1.001)

for i in range(K):
    col = TAB20[i % len(TAB20)]
    real_pts = ds.points[ds.labels == i][:N_PER].numpy()

    for j, (pts, label) in enumerate([(real_pts, "real"), (gen[i], "gen")]):
        ax = axes[i, j]
        ax.scatter(pts[:, 0], pts[:, 1], s=1.5, alpha=0.6, color=col, linewidths=0)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal"); ax.axis("off")
        if j == 0:
            ax.set_ylabel(names[i], fontsize=8, labelpad=2)
        if i == 0:
            ax.set_title(label, fontsize=9)

plt.tight_layout(pad=0.3, h_pad=0.15)
out = "compare_real_vs_gen.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
