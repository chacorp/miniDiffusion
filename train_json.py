import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import wandb
from dataset import JsonShapeDataset
from model import DiffusionMLP, make_noise_schedule, q_sample, p_sample_loop

# ─── Hyperparameters ──────────────────────────────────────────────
config = dict(
    T            = 1000,
    batch_size   = 512,
    epochs       = 500,
    lr           = 3e-4,
    # dataset
    data_dir     = "data",
    num_samples  = 60000,
    noise        = 0.005,
    val_ratio    = 0.1,
    # model
    hidden_dim   = 128,
    num_layers   = 4,
    emb_dim      = 64,
    dropout      = 0.1,
    # noise schedule
    schedule      = "cosine",
    # loss weighting
    min_snr_gamma = 5.0,
    # timestep sampling
    t_logit_mean  = -1.0,
    t_logit_std   = 1.5,
    # training
    sample_every  = 25,
    p_uncond      = 0.1,
)

# ─── Setup ────────────────────────────────────────────────────────
wandb.init(project="mini-diffusion", name="json-shapes", config=config)
cfg = wandb.config

run_dir = Path("outputs") / datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir.mkdir(parents=True)
(run_dir / "config.json").write_text(json.dumps(config, indent=2))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
print(f"run dir: {run_dir}")

dataset = JsonShapeDataset(
    data_dir    = cfg.data_dir,
    num_samples = cfg.num_samples,
    noise       = cfg.noise,
)
K     = dataset.num_clusters
names = dataset.shape_names
print(f"classes: {K}  shapes: {names}")

n_val   = int(len(dataset) * cfg.val_ratio)
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(
    dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
)

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=0, pin_memory=device.type == "cuda")
val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                          num_workers=0, pin_memory=device.type == "cuda")

model = DiffusionMLP(
    hidden_dim   = cfg.hidden_dim,
    num_layers   = cfg.num_layers,
    num_clusters = K,
    emb_dim      = cfg.emb_dim,
    dropout      = cfg.dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = (
    make_noise_schedule(cfg.T, schedule=cfg.schedule, device=device)
)

_data_var = dataset.points.var(unbiased=False).item()
snr       = alphas_cumprod / (1.0 - alphas_cumprod + 1e-8) * _data_var
min_snr_w = (snr.clamp(max=cfg.min_snr_gamma) / snr.clamp(min=1e-8))
print(f"data_var={_data_var:.4f}  SNR=1 at t~{(snr > 1).sum().item()}")

n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params:,}")
wandb.summary["parameters"] = n_params


# ─── Helper: generate & plot samples ──────────────────────────────
@torch.no_grad()
def log_samples(epoch: int):
    model.eval()
    n_per = 400
    tab20 = plt.cm.tab20.colors

    all_pts = []
    for cls in range(K):
        c   = torch.full((n_per,), cls, device=device, dtype=torch.long)
        pts = p_sample_loop(model, c, cfg.T, betas, alphas, alphas_cumprod).cpu().numpy()
        all_pts.append(pts)

    ncols = 4
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
    axes = axes.flat

    for i in range(K):
        ax  = axes[i]
        pts = all_pts[i]
        ax.scatter(pts[:, 0], pts[:, 1],
                   color=tab20[i % len(tab20)], s=4, alpha=0.7, linewidths=0)
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2,
                     fill=False, edgecolor="#aaaaaa", lw=0.7, ls="--"))
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_title(names[i], fontsize=9, color=tab20[i % len(tab20)], fontweight="bold")
        ax.set_aspect("equal")
        ax.axis("off")

    for ax in list(axes)[K:]:
        ax.axis("off")

    fig.suptitle(f"epoch {epoch}", fontsize=11)
    plt.tight_layout()
    wandb.log({"generated": wandb.Image(fig)}, step=epoch)
    plt.close(fig)


# ─── Training loop ────────────────────────────────────────────────
best_val = float("inf")

for epoch in range(1, cfg.epochs + 1):
    model.train()
    train_loss = 0.0
    for x0, c in train_loader:
        x0, c = x0.to(device), c.to(device)

        u = torch.randn(x0.shape[0], device=device) * cfg.t_logit_std + cfg.t_logit_mean
        t = (torch.sigmoid(u) * cfg.T).long().clamp(0, cfg.T - 1)

        x_t, noise = q_sample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

        c_in = c.clone()
        c_in[torch.rand(c_in.shape[0], device=device) < cfg.p_uncond] = K

        per_sample = F.mse_loss(model(x_t, t, c_in), noise, reduction="none").mean(-1)
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
            x0, c = x0.to(device), c.to(device)
            t     = torch.randint(0, cfg.T, (x0.shape[0],), device=device)
            x_t, noise = q_sample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            val_loss  += F.mse_loss(model(x_t, t, c), noise).item()

    scheduler.step()

    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    lr        = scheduler.get_last_lr()[0]

    wandb.log({"train/loss": avg_train, "val/loss": avg_val, "lr": lr}, step=epoch)

    if avg_val < best_val:
        best_val = avg_val
        torch.save(model.state_dict(), run_dir / "best.pt")

    if epoch % cfg.sample_every == 0 or epoch == 1:
        print(f"epoch {epoch:4d}/{cfg.epochs}  train={avg_train:.6f}  val={avg_val:.6f}  lr={lr:.2e}")
        log_samples(epoch)

# ─── Save ─────────────────────────────────────────────────────────
final_path = run_dir / "model.pt"
torch.save(model.state_dict(), final_path)
wandb.save(str(final_path))
print(f"saved {final_path}")

wandb.finish()
