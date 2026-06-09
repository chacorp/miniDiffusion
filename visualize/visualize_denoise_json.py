"""B. 디노이징 과정 시각화 (noise → shape) for all 12 classes."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset import JsonShapeDataset
from model import DiffusionMLP, make_noise_schedule, p_sample_cfg

MODEL_PATH = Path("outputs/20260608_222600/best.pt")
cfg_data   = json.loads((MODEL_PATH.parent / "config.json").read_text())

T = cfg_data["T"]; K = 12; N_PER = 400
SHOW_T = [1000, 800, 600, 400, 200, 100, 50, 0]   # timesteps to snapshot
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

torch.manual_seed(0)

@torch.no_grad()
def run_with_snapshots(cls: int):
    c = torch.full((N_PER,), cls, dtype=torch.long)
    x = torch.randn(N_PER, 2)
    snaps = {T: x.numpy().copy()}
    for t_idx in reversed(range(T)):
        x = p_sample_cfg(model, x, t_idx, c, betas, alphas, alpha_bars, w=2.0)
        if t_idx in SHOW_T:
            snaps[t_idx] = x.numpy().copy()
    snaps[0] = x.numpy().copy()
    return snaps

print("Running reverse diffusion for all 12 classes...")
all_snaps = []
for cls in range(K):
    print(f"  class {cls} ({names[cls]})")
    all_snaps.append(run_with_snapshots(cls))

# Plot: rows=classes, cols=timestep snapshots
snap_keys = sorted([k for k in SHOW_T], reverse=True)
n_rows = K
n_cols = len(snap_keys)
cs = 1.4

fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(cs * n_cols, cs * n_rows + 0.4),
                          squeeze=False)
fig.suptitle("Denoising trajectory: noise → shape  (w=2.0)", fontsize=11, y=1.001)

for r, (snaps, name) in enumerate(zip(all_snaps, names)):
    col = TAB20[r % len(TAB20)]
    for c_idx, tk in enumerate(snap_keys):
        ax  = axes[r, c_idx]
        pts = snaps.get(tk, snaps[min(snaps.keys(), key=lambda k: abs(k - tk))])
        ax.scatter(pts[:, 0], pts[:, 1], s=1.5, alpha=0.65, color=col, linewidths=0)
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
        ax.set_aspect("equal"); ax.axis("off")
        if r == 0:
            ax.set_title(f"t={tk}", fontsize=8)
    axes[r, 0].set_ylabel(name, fontsize=8, labelpad=2)

plt.tight_layout(pad=0.15, h_pad=0.1, w_pad=0.1)
out = "denoise_trajectory.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out}")
