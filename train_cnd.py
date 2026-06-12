"""Training script for PointNetDenoiserCnd (cross-attention, self-reconstruction)."""
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from dataset import JsonPointShapeDataset
from model import (
    PointNetDenoiserCnd,
    SelfReconCollate,
    make_noise_schedule,
    q_sample_point_cloud,
    ddim_sample_cnd,
    resample_points,
)


config = dict(
    T=1000,
    batch_size=8,
    epochs=100,
    lr=3e-4,
    # dataset
    data_dir="data",
    points_per_shape=10000,
    noise=0.005,
    val_ratio=0.1,
    clouds_per_epoch=512,
    val_clouds=64,
    # point counts
    n_points=256,       # target N during training
    cond_points=128,    # condition M during training
    sample_points=256,  # N for visualization
    # model
    point_dim=2,
    hidden_dim=128,
    num_layers=4,
    emb_dim=64,
    num_heads=4,
    dropout=0.1,
    # noise schedule
    schedule="cosine",
    # loss weighting
    min_snr_gamma=5.0,
    # timestep sampling
    t_logit_mean=-1.0,
    t_logit_std=1.5,
    # training
    sample_every=10,
    p_uncond=0.15,
    # DDIM for visualization
    ddim_steps=50,
    cfg_w=1.5,
)


wandb.init(project="mini-diffusion-cnd", config=config)
cfg = wandb.config

run_dir = Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}  |  run_dir: {run_dir}")

# ─── Dataset ──────────────────────────────────────────────────────────────────

train_ds = JsonPointShapeDataset(
    data_dir=cfg.data_dir,
    points_per_shape=cfg.points_per_shape,
    noise=cfg.noise,
    split="train",
    val_ratio=cfg.val_ratio,
    items_per_epoch=cfg.clouds_per_epoch,
    seed=42,
)
val_ds = JsonPointShapeDataset(
    data_dir=cfg.data_dir,
    points_per_shape=cfg.points_per_shape,
    noise=cfg.noise,
    split="val",
    val_ratio=cfg.val_ratio,
    items_per_epoch=cfg.val_clouds,
    seed=42,
)

num_classes = train_ds.num_clusters
shape_names = train_ds.shape_names
print(f"classes: {num_classes}  names: {shape_names}")

run_config = {
    **dict(config),
    "num_clusters": num_classes,
    "shape_names":  shape_names,
}
(run_dir / "config.json").write_text(json.dumps(run_config, indent=2))
wandb.config.update({"num_clusters": num_classes}, allow_val_change=True)

collate_train = SelfReconCollate(n_points=cfg.n_points, cond_points=cfg.cond_points)
collate_val   = SelfReconCollate(n_points=cfg.sample_points, cond_points=cfg.cond_points)

train_loader = DataLoader(
    train_ds, batch_size=cfg.batch_size, shuffle=True,
    num_workers=0, pin_memory=(device.type == "cuda"),
    collate_fn=collate_train,
)
val_loader = DataLoader(
    val_ds, batch_size=cfg.batch_size, shuffle=False,
    num_workers=0, pin_memory=(device.type == "cuda"),
    collate_fn=collate_val,
)

# ─── Model ────────────────────────────────────────────────────────────────────

model = PointNetDenoiserCnd(
    input_dim=cfg.point_dim,
    hidden_dim=cfg.hidden_dim,
    num_layers=cfg.num_layers,
    emb_dim=cfg.emb_dim,
    num_heads=cfg.num_heads,
    dropout=cfg.dropout,
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params:,}")
wandb.summary["parameters"] = n_params

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

# ─── Noise schedule & SNR weights ─────────────────────────────────────────────

_, _, alpha_bars, sqrt_ab, sqrt_one_minus_ab = make_noise_schedule(
    cfg.T, schedule=cfg.schedule, device=device
)

data_var  = train_ds.points.var(unbiased=False).item()
snr       = alpha_bars / (1.0 - alpha_bars + 1e-8) * data_var
min_snr_w = snr.clamp(max=cfg.min_snr_gamma) / snr.clamp(min=1e-8)
print(f"data_var={data_var:.4f}  SNR=1 at t~{(snr > 1).sum().item()}")


# ─── Visualization ────────────────────────────────────────────────────────────

@torch.no_grad()
def log_samples(epoch: int):
    model.eval()
    tab10 = plt.cm.tab10.colors

    vis_classes = list(range(min(num_classes, 4)))  # max 4 classes to keep it fast
    fig, axes = plt.subplots(1, len(vis_classes), figsize=(3.5 * len(vis_classes), 3.5))
    if len(vis_classes) == 1:
        axes = [axes]

    for cls in vis_classes:
        # Use GT cloud of this class as condition
        cond_pc = resample_points(train_ds.clouds[cls], cfg.cond_points)
        cond_pc = cond_pc.unsqueeze(0).to(device)  # [1, M, 2]

        pts = ddim_sample_cnd(
            model, cond_pc, cfg.sample_points, alpha_bars,
            cfg.T, n_steps=cfg.ddim_steps, w=cfg.cfg_w,
        )[0].cpu().numpy()  # [N, 2]

        ax = axes[vis_classes.index(cls)]
        ax.scatter(pts[:, 0], pts[:, 1], color=tab10[cls % len(tab10)],
                   s=5, alpha=0.7, linewidths=0)
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(shape_names[cls], fontsize=10, fontweight="bold",
                     color=tab10[cls % len(tab10)])

    fig.suptitle(f"epoch {epoch} — cond-generated", fontsize=11)
    plt.tight_layout()
    wandb.log({"generated": wandb.Image(fig)}, step=epoch)
    plt.close(fig)


# ─── Training loop ────────────────────────────────────────────────────────────

best_val = float("inf")

for epoch in range(1, cfg.epochs + 1):
    # ── train
    model.train()
    train_loss = 0.0

    for x0, cond_pc, _ in train_loader:
        # x0:      [B, N, 2]  target (noisy)
        # cond_pc: [B, M, 2]  condition (clean, same shape)
        x0      = x0.to(device)
        cond_pc = cond_pc.to(device)
        B       = x0.shape[0]

        # Logit-normal timestep sampling (biased toward mid-t)
        u = torch.randn(B, device=device) * cfg.t_logit_std + cfg.t_logit_mean
        t = (torch.sigmoid(u) * cfg.T).long().clamp(0, cfg.T - 1)

        x_t, noise = q_sample_point_cloud(x0, t, sqrt_ab, sqrt_one_minus_ab)

        # Encode condition; CFG dropout → null_token
        cond_tokens = model.encode_cond(cond_pc)                          # [B, M, D]
        drop_mask   = torch.rand(B, device=device) < cfg.p_uncond         # [B]
        null_tokens = model.get_null_tokens(B, device).expand_as(cond_tokens)
        cond_tokens = torch.where(drop_mask[:, None, None], null_tokens, cond_tokens)

        pred        = model(x_t, t, cond_tokens)
        per_sample  = F.mse_loss(pred, noise, reduction="none").mean(dim=(1, 2))
        loss        = (min_snr_w[t] * per_sample).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    # ── val
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x0, cond_pc, _ in val_loader:
            x0      = x0.to(device)
            cond_pc = cond_pc.to(device)
            B       = x0.shape[0]

            t = torch.randint(0, cfg.T, (B,), device=device)
            x_t, noise  = q_sample_point_cloud(x0, t, sqrt_ab, sqrt_one_minus_ab)
            cond_tokens = model.encode_cond(cond_pc)
            val_loss   += F.mse_loss(model(x_t, t, cond_tokens), noise).item()

    scheduler.step()

    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    lr        = scheduler.get_last_lr()[0]

    wandb.log({"train/loss": avg_train, "val/loss": avg_val, "lr": lr}, step=epoch)

    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), run_dir / "best.pt")

    if epoch % cfg.sample_every == 0 or epoch == 1:
        print(f"epoch {epoch:4d}/{cfg.epochs}  "
              f"train={avg_train:.6f}  val={avg_val:.6f}  lr={lr:.2e}")
        log_samples(epoch)

torch.save(model.state_dict(), run_dir / "model.pt")
print(f"done — best val={best_val:.6f}  saved to {run_dir}")
wandb.finish()
