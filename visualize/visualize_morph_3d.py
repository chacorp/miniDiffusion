"""Interactive 3D layered visualization: diffusion denoising + CFG morph.

Z axis  = diffusion timestep  (T=1000 at bottom → t=0 at top)
Color   = w value  (circle=blue, morph=purple, star=red)
Layer   = snapshot of point cloud at that timestep
Yellow line = single tracked point trajectory through denoising steps

Shows: at t=T all w values start from the same noise,
       then diverge into different shapes as denoising progresses.

Output: morph_3d.html
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go

from dataset import ShapeDataset
from model import DiffusionMLP, make_noise_schedule

# ── Load model ────────────────────────────────────────────────────
MODEL_PATH = sorted(Path("outputs").glob("*/model.pt"))[-1]
cfg_data   = json.loads((MODEL_PATH.parent / "config.json").read_text())
print(f"model: {MODEL_PATH}")

T        = cfg_data["T"]
SCHEDULE = cfg_data.get("schedule", "cosine")
K        = cfg_data["num_clusters"]
NAMES    = ShapeDataset.SHAPE_NAMES

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DiffusionMLP(
    hidden_dim=cfg_data["hidden_dim"], num_layers=cfg_data["num_layers"],
    num_clusters=K, emb_dim=cfg_data["emb_dim"], dropout=0.0,
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

betas, alphas, alpha_bars, _, _ = make_noise_schedule(T, schedule=SCHEDULE, device=device)

# ── Settings ──────────────────────────────────────────────────────
CLS_FROM = NAMES.index("circle")
CLS_TO   = NAMES.index("star")
N        = 200   # points per w value

# w values to compare (3 curves)
W_SHOW = [0.0, 0.5, 1.0]

# Timesteps to record (bottom=noisy, top=clean)
RECORD_T = [T, 800, 600, 400, 200, 100, 50, 20, 0]

# ── Sampler with trajectory recording ────────────────────────────
@torch.no_grad()
def sample_with_traj(w, x_start):
    c_from = torch.full((N,), CLS_FROM, device=device, dtype=torch.long)
    c_to   = torch.full((N,), CLS_TO,   device=device, dtype=torch.long)
    x      = x_start.clone()
    record = {T: x.cpu().numpy().copy()}

    for t_idx in reversed(range(T)):
        B = x.shape[0]
        t = torch.full((B,), t_idx, device=device, dtype=torch.long)
        eps = model(x, t, c_from) + w * (model(x, t, c_to) - model(x, t, c_from))
        coef = (1.0 - alphas[t_idx]) / (1.0 - alpha_bars[t_idx]).sqrt()
        mean = (x - coef * eps) / alphas[t_idx].sqrt()
        x    = mean if t_idx == 0 else mean + betas[t_idx].sqrt() * torch.randn_like(x)
        if t_idx in RECORD_T:
            record[t_idx] = x.cpu().numpy().copy()

    return record   # dict: t_idx -> (N, 2)

# ── Fixed starting noise ──────────────────────────────────────────
torch.manual_seed(7)
x_start = torch.randn(N, 2, device=device)

print("Sampling denoising trajectories...")
all_records = {}
for w in W_SHOW:
    print(f"  w={w:.1f}")
    all_records[w] = sample_with_traj(w, x_start)

# ── Pick tracked point: most displaced w=0 -> w=1 at t=0 ─────────
disp    = np.linalg.norm(all_records[1.0][0] - all_records[0.0][0], axis=1)
TRACKED = int(np.argmax(disp))
print(f"Tracking point #{TRACKED}  (displacement={disp[TRACKED]:.3f})")

# ── Colors per w ──────────────────────────────────────────────────
TAB10   = plt.cm.tab10.colors
rgb_from = np.array(TAB10[CLS_FROM][:3])   # blue  (circle)
rgb_to   = np.array(TAB10[CLS_TO  ][:3])   # red   (star)

def w_color(w, alpha=1.0):
    t = np.clip(w, 0, 1)
    c = (1 - t) * rgb_from + t * rgb_to
    return f"rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{alpha})"

W_LABELS = {0.0: "circle (w=0)", 0.5: "morph (w=0.5)", 1.0: "star (w=1)"}

# ── Build figure ──────────────────────────────────────────────────
fig = go.Figure()

for wi, w in enumerate(W_SHOW):
    record   = all_records[w]
    color    = w_color(w, alpha=0.55)
    shown_lg = True

    for ti, t_snap in enumerate(RECORD_T):
        pts   = record[t_snap]
        z_val = t_snap

        # Point cloud
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=np.full(N, z_val),
            mode="markers",
            marker=dict(size=2.5, color=color),
            name=W_LABELS[w],
            legendgroup=f"w{wi}",
            showlegend=shown_lg,
            hovertemplate=f"w={w}  t={t_snap}<br>x=%{{x:.3f}}  y=%{{y:.3f}}<extra></extra>",
        ))
        shown_lg = False

        # Layer border
        bx = [-1, 1, 1, -1, -1]
        by = [-1, -1, 1, 1, -1]
        fig.add_trace(go.Scatter3d(
            x=bx, y=by, z=[z_val] * 5,
            mode="lines",
            line=dict(color="rgba(120,120,120,0.25)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

        # Tracked point on this layer
        tx, ty = pts[TRACKED]
        fig.add_trace(go.Scatter3d(
            x=[tx], y=[ty], z=[z_val],
            mode="markers",
            marker=dict(size=7, color="white", opacity=0.9,
                        line=dict(color=w_color(w, alpha=1.0), width=2)),
            showlegend=False, hoverinfo="skip",
        ))

    # Tracked-point trajectory line through time
    traj_x = [record[t][TRACKED, 0] for t in RECORD_T]
    traj_y = [record[t][TRACKED, 1] for t in RECORD_T]
    traj_z = RECORD_T
    fig.add_trace(go.Scatter3d(
        x=traj_x, y=traj_y, z=traj_z,
        mode="lines",
        line=dict(color=w_color(w, alpha=1.0), width=5),
        name=f"traj w={w}",
        legendgroup=f"w{wi}",
        showlegend=True,
        hoverinfo="skip",
    ))

# ── Layout ────────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text="CFG Morph Denoising  (Z = timestep,  color = w value)",
        font=dict(size=14),
    ),
    scene=dict(
        xaxis=dict(range=[-1.5, 1.5], title="x", showgrid=False),
        yaxis=dict(range=[-1.5, 1.5], title="y", showgrid=False),
        zaxis=dict(
            title="diffusion timestep  (T=noisy → 0=clean)",
            tickvals=RECORD_T,
            ticktext=[str(t) for t in RECORD_T],
            autorange=True,
        ),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=2.5),
        camera=dict(eye=dict(x=1.7, y=-1.7, z=0.9)),
        bgcolor="rgba(12,12,18,1)",
    ),
    paper_bgcolor="rgba(12,12,18,1)",
    font=dict(color="white"),
    legend=dict(x=0.01, y=0.95, bgcolor="rgba(0,0,0,0.45)",
                tracegroupgap=4),
    width=1050,
    height=800,
)

out = "morph_3d.html"
fig.write_html(out)
print(f"Saved -> {out}")
