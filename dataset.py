import numpy as np
import torch
from torch.utils.data import Dataset


class GaussianClusters(Dataset):
    """2D points sampled from a mixture of isotropic Gaussian clusters.

    Args:
        num_samples:  Total number of points to sample.
        num_clusters: Number of Gaussian components.
        overlap:      Controls how much clusters overlap.
                      Std of each cluster = overlap * (distance to nearest neighbor center).
                      overlap~0.1 → well separated, overlap~0.5 → moderate, overlap~1.0+ → heavy.
        seed:         RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_samples: int = 10000,
        num_clusters: int = 5,
        overlap: float = 0.3,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed)

        # Place cluster centers uniformly on a unit circle
        angles = np.linspace(0, 2 * np.pi, num_clusters, endpoint=False)
        centers = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (K, 2)

        # Std scales with inter-cluster distance so overlap is meaningful
        # regardless of num_clusters
        inter_dist = 2 * np.sin(np.pi / num_clusters) if num_clusters > 1 else 1.0
        std = overlap * inter_dist

        cluster_ids = rng.integers(0, num_clusters, size=num_samples)
        noise = rng.normal(0, std, size=(num_samples, 2))
        points = centers[cluster_ids] + noise

        self.points = torch.tensor(points, dtype=torch.float32)
        self.labels = torch.tensor(cluster_ids, dtype=torch.long)
        self.centers = centers
        self.std = float(std)
        self.num_clusters = num_clusters

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.points[idx], self.labels[idx]
