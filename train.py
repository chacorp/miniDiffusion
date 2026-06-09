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
    PointNetDenoiser,
    VariablePointCountCollate,
    make_noise_schedule,
    p_sample_loop_point_cloud,
    q_sample_point_cloud,
)


config = dict(
    T=1000,
    batch_size=8,
    epochs=300,
    lr=3e-4,
    # dataset
    data_dir="data",
    points_per_shape=10000,
    noise=0.005,
    val_ratio=0.1,
    point_counts=[512, 1024, 2048],
    sample_points=2048,
    clouds_per_epoch=2048,
    val_clouds=128,
    # model
    point_dim=2,
    hidden_dim=256,
    num_layers=6,
    emb_dim=128,
    dropout=0.1,
    # noise schedule
    schedule="cosine",
    # loss weighting
    min_snr_gamma=5.0,
    # timestep sampling
    t_logit_mean=-1.0,
    t_logit_std=1.5,
    # training
    sample_every=20,
    p_uncond=0.1,
)


wandb.init(project="mini-diffusion", config=config)
cfg = wandb.config

run_dir = Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
print(f"run dir: {run_dir}")

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
wandb.config.update({"num_clusters": num_classes}, allow_val_change=True)
run_config = {**config, "num_clusters": num_classes, "shape_names": shape_names}
(run_dir / "config.json").write_text(json.dumps(run_config, indent=2))

train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=device.type == "cuda",
    collate_fn=VariablePointCountCollate(
        cfg.point_counts,
        generator=torch.Generator().manual_seed(1),
    ),
)
val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=device.type == "cuda",
    collate_fn=VariablePointCountCollate(
        (cfg.sample_points,),
        generator=torch.Generator().manual_seed(2),
    ),
)

model = PointNetDenoiser(
    input_dim=cfg.point_dim,
    hidden_dim=cfg.hidden_dim,
    num_layers=cfg.num_layers,
    num_classes=num_classes,
    emb_dim=cfg.emb_dim,
    dropout=cfg.dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = (
    make_noise_schedule(cfg.T, schedule=cfg.schedule, device=device)
)

data_var = train_ds.points.var(unbiased=False).item()
snr = alphas_cumprod / (1.0 - alphas_cumprod + 1e-8) * data_var
min_snr_w = snr.clamp(max=cfg.min_snr_gamma) / snr.clamp(min=1e-8)
print(f"data_var={data_var:.4f}  SNR=1 at t~{(snr > 1).sum().item()}")

n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params:,}")
wandb.summary["parameters"] = n_params


@torch.no_grad()
def log_samples(epoch: int):
    model.eval()
    tab10 = plt.cm.tab10.colors

    all_pts = []
    for cls in range(num_classes):
        c = torch.full((1,), cls, device=device, dtype=torch.long)
        pts = p_sample_loop_point_cloud(
            model,
            c,
            cfg.sample_points,
            cfg.T,
            betas,
            alphas,
            alphas_cumprod,
        )[0].cpu().numpy()
        all_pts.append(pts)

    fig, axes = plt.subplots(1, num_classes, figsize=(3.5 * num_classes, 3.5))
    if num_classes == 1:
        axes = [axes]

    for i, (ax, pts) in enumerate(zip(axes, all_pts)):
        color = tab10[i % len(tab10)]
        ax.scatter(pts[:, 0], pts[:, 1], color=color, s=5, alpha=0.7, linewidths=0)
        ax.add_patch(
            plt.Rectangle((-1, -1), 2, 2, fill=False, edgecolor="#aaaaaa", lw=0.7, ls="--")
        )
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_title(shape_names[i], fontsize=11, color=color, fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(f"epoch {epoch}", fontsize=11)
    plt.tight_layout()
    wandb.log({"generated": wandb.Image(fig)}, step=epoch)
    plt.close(fig)


best_val = float("inf")

for epoch in range(1, cfg.epochs + 1):
    model.train()
    train_loss = 0.0

    for x0, c in train_loader:
        x0 = x0.to(device)
        c = c.to(device)

        u = torch.randn(x0.shape[0], device=device) * cfg.t_logit_std + cfg.t_logit_mean
        t = (torch.sigmoid(u) * cfg.T).long().clamp(0, cfg.T - 1)
        x_t, noise = q_sample_point_cloud(
            x0,
            t,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
        )

        c_in = c.clone()
        c_in[torch.rand(c_in.shape[0], device=device) < cfg.p_uncond] = num_classes

        pred = model(x_t, t, c_in)
        per_sample = F.mse_loss(pred, noise, reduction="none").mean(dim=(1, 2))
        loss = (min_snr_w[t] * per_sample).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x0, c in val_loader:
            x0 = x0.to(device)
            c = c.to(device)
            t = torch.randint(0, cfg.T, (x0.shape[0],), device=device)
            x_t, noise = q_sample_point_cloud(
                x0,
                t,
                sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod,
            )
            val_loss += F.mse_loss(model(x_t, t, c), noise).item()

    scheduler.step()

    avg_train = train_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)
    lr = scheduler.get_last_lr()[0]

    wandb.log({"train/loss": avg_train, "val/loss": avg_val, "lr": lr}, step=epoch)

    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), run_dir / "best.pt")

    if epoch % cfg.sample_every == 0 or epoch == 1:
        print(
            f"epoch {epoch:4d}/{cfg.epochs}  "
            f"train={avg_train:.6f}  val={avg_val:.6f}  lr={lr:.2e}"
        )
        log_samples(epoch)

final_path = run_dir / "model.pt"
torch.save(model.state_dict(), final_path)
wandb.save(str(final_path))
print(f"saved {final_path}")

wandb.finish()
