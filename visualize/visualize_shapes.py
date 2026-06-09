"""Visualise ShapeDataset: 4 shapes in [-1, 1]² space."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from dataset import ShapeDataset

ds     = ShapeDataset(num_samples=12000, noise=0.01, outline=True)
pts    = ds.points.numpy()
labels = ds.labels.numpy()

TAB10 = plt.cm.tab10.colors

fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
fig.suptitle("ShapeDataset  —  [-1, 1]² space, condition → shape", fontsize=13)

for i, (ax, name) in enumerate(zip(axes, ShapeDataset.SHAPE_NAMES)):
    mask = labels == i
    ax.scatter(pts[mask, 0], pts[mask, 1],
               color=TAB10[i], s=3, alpha=0.6, linewidths=0)

    # [-1, 1] boundary
    ax.add_patch(mpatches.Rectangle((-1, -1), 2, 2,
                 fill=False, edgecolor="#aaaaaa", linewidth=0.8, linestyle="--"))
    ax.axhline(0, color="#dddddd", linewidth=0.4, zorder=0)
    ax.axvline(0, color="#dddddd", linewidth=0.4, zorder=0)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_title(f"{i}  {name}", fontsize=11, color=TAB10[i], fontweight="bold")
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout()
plt.savefig("shape_visualization.png", dpi=150, bbox_inches="tight")
print("Saved -> shape_visualization.png")
plt.show()
