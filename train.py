import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

import wandb
from dataset import GaussianClusters
from model import DiffusionMLP, make_noise_schedule, q_sample, p_sample_loop

# ─── Hyperparameters ──────────────────────────────────────────────
config = dict(
    T            = 1000,
    batch_size   = 512,
    epochs       = 200,
    lr           = 3e-4,
    num_clusters = 5,
    overlap      = 0.3,
    hidden_dim   = 128,
    num_layers   = 4,
    emb_dim      = 64,
    dropout      = 0.1,
    num_samples  = 50000,
    val_ratio    = 0.1,
    sample_every = 20,   # log generated scatter every N epochs
    p_uncond     = 0.1,  # prob of dropping class label during training (CFG)
)

# ─── Setup ────────────────────────────────────────────────────────
wandb.init(project="mini-diffusion", config=config)
cfg = wandb.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

dataset = GaussianClusters(
    num_samples=cfg.num_samples,
    num_clusters=cfg.num_clusters,
    overlap=cfg.overlap,
)
n_val   = int(len(dataset) * cfg.val_ratio)
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0, pin_memory=device.type == "cuda")
val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

model = DiffusionMLP(
    hidden_dim   = cfg.hidden_dim,
    num_layers   = cfg.num_layers,
    num_clusters = cfg.num_clusters,
    emb_dim      = cfg.emb_dim,
    dropout      = cfg.dropout,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod = (
    make_noise_schedule(cfg.T, device=device)
)

n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params:,}")
wandb.summary["parameters"] = n_params


# ─── Helper: generate & plot samples ──────────────────────────────
@torch.no_grad()
def log_samples(epoch: int):
    model.eval()
    n_per = 200
    c_all = torch.arange(cfg.num_clusters, device=device).repeat_interleave(n_per)
    pts   = p_sample_loop(model, c_all, cfg.T, betas, alphas, alphas_cumprod).cpu().numpy()
    labels = c_all.cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="tab10", s=6, alpha=0.7, linewidths=0)
    ax.set_title(f"epoch {epoch}")
    ax.set_aspect("equal")
    ax.axis("off")
    wandb.log({"generated": wandb.Image(fig)}, step=epoch)
    plt.close(fig)


# ─── Training loop ────────────────────────────────────────────────
for epoch in range(1, cfg.epochs + 1):
    # Train
    model.train()
    train_loss = 0.0
    for x0, c in train_loader:
        x0, c = x0.to(device), c.to(device)
        t = torch.randint(0, cfg.T, (x0.shape[0],), device=device)
        x_t, noise = q_sample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

        # CFG: randomly replace class label with null token
        c_in = c.clone()
        c_in[torch.rand(c_in.shape[0], device=device) < cfg.p_uncond] = cfg.num_clusters

        loss = F.mse_loss(model(x_t, t, c_in), noise)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x0, c in val_loader:
            x0, c = x0.to(device), c.to(device)
            t = torch.randint(0, cfg.T, (x0.shape[0],), device=device)
            x_t, noise = q_sample(x0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
            val_loss += F.mse_loss(model(x_t, t, c), noise).item()

    scheduler.step()

    avg_train = train_loss / len(train_loader)
    avg_val   = val_loss   / len(val_loader)
    lr        = scheduler.get_last_lr()[0]

    wandb.log({"train/loss": avg_train, "val/loss": avg_val, "lr": lr}, step=epoch)

    if epoch % cfg.sample_every == 0 or epoch == 1:
        print(f"epoch {epoch:4d}/{cfg.epochs}  train={avg_train:.6f}  val={avg_val:.6f}  lr={lr:.2e}")
        log_samples(epoch)

# ─── Save ─────────────────────────────────────────────────────────
torch.save(model.state_dict(), "model.pt")
wandb.save("model.pt")
print("saved model.pt")

wandb.finish()
