import matplotlib.pyplot as plt
import numpy as np

from dataset import GaussianClusters

OVERLAP_VALUES = [0.1, 0.3, 0.6, 1.0]
NUM_CLUSTERS = 5
NUM_SAMPLES = 3000

fig, axes = plt.subplots(1, len(OVERLAP_VALUES), figsize=(4 * len(OVERLAP_VALUES), 4))

for ax, overlap in zip(axes, OVERLAP_VALUES):
    ds = GaussianClusters(num_samples=NUM_SAMPLES, num_clusters=NUM_CLUSTERS, overlap=overlap)
    points = ds.points.numpy()
    labels = ds.labels.numpy()

    ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10", s=6, alpha=0.6, linewidths=0)
    ax.scatter(ds.centers[:, 0], ds.centers[:, 1], c="black", s=60, marker="x", linewidths=1.5)
    ax.set_title(f"overlap={overlap}\nstd={ds.std:.3f}")
    ax.set_aspect("equal")
    ax.axis("off")

fig.suptitle(f"Gaussian Clusters  (num_clusters={NUM_CLUSTERS})", fontsize=13, y=1.02)
plt.tight_layout()

out_path = "cluster_visualization.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved → {out_path}")
plt.show()
