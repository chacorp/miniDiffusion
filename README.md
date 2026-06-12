# miniDiffusion

2D clothing shape point cloud diffusion model.

---

## Structure

```
miniDiffusion/
├── data/               # JSON shape files (0000–0016)
├── model/
│   ├── mlp_denoiser.py          # Point-wise MLP denoiser (class-conditioned)
│   ├── pointnet_denoiser.py     # PointNet denoiser (class-conditioned)
│   └── pointnet_denoiser_cnd.py # PointNet denoiser (shape-conditioned, cross-attn)
├── visualize/          # Visualization scripts
├── data_create.py      # Interactive shape editor (PyQt6)
├── dataset.py          # JsonShapeDataset, JsonPointShapeDataset
├── train.py            # Train PointNetDenoiser (class-conditioned)
├── train_cnd.py        # Train PointNetDenoiserCnd (shape-conditioned)
└── pyproject.toml
```

### Model: PointNetDenoiserCnd

```
cond_pc [B,M,2]
  └─ PointNetEncoder (point-wise MLP)
       └─ pre_pool [B,M,D]
            ├─ LearnedPooling (FC + sum) -> global_token [B,1,D]
            └─ PointNetDecoder           -> recon [B,M,2]  (aux loss)

noisy_x [B,N,2]
  └─ input_proj -> h [B,N,D]
       └─ x n CrossAttentionBlock
            ├─ Self-Attn (h)
            ├─ Cross-Attn (Q=h, K/V=global_token)
            ├─ FiLM (timestep)
            └─ FFN
                └─ eps_pred [B,N,2]

L_total = L_diffusion + lambda * L_recon
```

---

## Setup

```bash
# install uv
pip install uv

# create venv and install dependencies (CUDA)
uv sync

# CPU only: remove [[tool.uv.index]] and [tool.uv.sources] in pyproject.toml, then:
uv sync
```

---

## Data Creation

```bash
python data_create.py
```

Draw clothing shapes interactively and save as `data/<name>.json`.

---

## Training

```bash
# Shape-conditioned model (PointNetDenoiserCnd)
python train_cnd.py

# Class-conditioned PointNet model
python train.py

# Class-conditioned MLP model
python train_json.py
```

---

## Visualization

```bash
# Real vs Generated
python visualize/pn_compare.py

# Denoising trajectory
python visualize/pn_denoise.py

# SDEdit morphing overlay
python visualize/pn_overlay.py

# Unseen class reconstruction (SDEdit)
python visualize/pn_new_classes.py
```
