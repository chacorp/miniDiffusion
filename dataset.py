import json
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicSpline
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


# ─── ShapeDataset ──────────────────────────────────────────────────

_SQRT3 = float(np.sqrt(3))


def _in_polygon(
    px: np.ndarray, py: np.ndarray,
    vx: np.ndarray, vy: np.ndarray,
) -> np.ndarray:
    """Even-odd ray-casting point-in-polygon test (vectorised over points)."""
    inside = np.zeros(len(px), dtype=bool)
    j = len(vx) - 1
    for i in range(len(vx)):
        xi, yi, xj, yj = vx[i], vy[i], vx[j], vy[j]
        straddle = (yi > py) != (yj > py)
        x_cross  = (xj - xi) * (py - yi) / ((yj - yi) + 1e-12) + xi
        inside  ^= straddle & (px < x_cross)
        j = i
    return inside


def _rejection_sample(n: int, rng, half: float, accept_fn) -> np.ndarray:
    """Rejection-sample n 2-D points from [−half, half]² using accept_fn."""
    chunks: list[np.ndarray] = []
    total = 0
    while total < n:
        batch    = rng.uniform(-half, half, (max(n * 6, 512), 2))
        accepted = batch[accept_fn(batch[:, 0], batch[:, 1])]
        chunks.append(accepted)
        total   += len(accepted)
    return np.concatenate(chunks)[:n]


def _sample_perimeter(vertices: np.ndarray, n: int, rng) -> np.ndarray:
    """Sample n points uniformly from the perimeter of a closed polygon."""
    V       = np.asarray(vertices, dtype=np.float64)
    edges   = np.roll(V, -1, axis=0) - V          # edge vectors, wraps around
    lengths = np.linalg.norm(edges, axis=1)
    probs   = lengths / lengths.sum()
    idx     = rng.choice(len(V), size=n, p=probs)  # edge index ~ length
    t       = rng.uniform(0.0, 1.0, n)             # position along edge
    return (V[idx] + t[:, None] * edges[idx]).astype(np.float32)


class ShapeDataset(Dataset):
    """2D points sampled from geometric shapes, all within [-1, 1]².

    Each class label maps to a distinct shape centered at the origin:
      0 → circle   (radius 0.80)
      1 → square   (half-side 0.75)
      2 → triangle (circumradius 0.90)
      3 → star     (outer 0.90, inner 0.38)

    Args:
        num_samples: Total number of 2-D points.
        noise:       Std of isotropic Gaussian noise added to each point.
        outline:     True  → sample from perimeter only (contour).
                     False → sample from filled interior (default).
        seed:        RNG seed.
    """

    SHAPE_NAMES: list[str] = ["circle", "square", "triangle", "star"]

    def __init__(
        self,
        num_samples: int   = 10000,
        noise:       float = 0.01,
        outline:     bool  = False,
        seed:        int   = 42,
    ):
        rng = np.random.default_rng(seed)
        K   = len(self.SHAPE_NAMES)

        n_each    = num_samples // K
        remainder = num_samples - n_each * K

        all_pts: list[np.ndarray] = []
        all_lbl: list[np.ndarray] = []

        for idx, name in enumerate(self.SHAPE_NAMES):
            n   = n_each + (1 if idx < remainder else 0)
            pts = self._outline(name, n, rng) if outline else self._fill(name, n, rng)
            pts = pts + rng.standard_normal(pts.shape) * noise
            all_pts.append(pts.astype(np.float32))
            all_lbl.append(np.full(n, idx, dtype=np.int64))

        pts_arr = np.concatenate(all_pts)
        lbl_arr = np.concatenate(all_lbl)
        perm    = rng.permutation(len(pts_arr))

        self.points       = torch.from_numpy(pts_arr[perm])
        self.labels       = torch.from_numpy(lbl_arr[perm])
        self.centers      = np.zeros((K, 2))
        self.num_clusters = K

    # ── Outline (perimeter) samplers ──────────────────────────────

    def _outline(self, name: str, n: int, rng) -> np.ndarray:
        if name == "circle":
            theta = rng.uniform(0, 2 * np.pi, n)
            return np.stack([0.80 * np.cos(theta), 0.80 * np.sin(theta)], axis=1)

        if name == "square":
            s = 0.75
            verts = np.array([[-s,-s],[s,-s],[s,s],[-s,s]])
            return _sample_perimeter(verts, n, rng)

        if name == "triangle":
            R  = 0.90
            v0 = np.array([0.0,             R       ])
            v1 = np.array([-R * _SQRT3 / 2, -R / 2.0])
            v2 = np.array([ R * _SQRT3 / 2, -R / 2.0])
            return _sample_perimeter(np.array([v0, v1, v2]), n, rng)

        if name == "star":
            outer, inner = 0.90, 0.38
            a     = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 10, endpoint=False)
            r     = np.where(np.arange(10) % 2 == 0, outer, inner)
            verts = np.stack([r * np.cos(a), r * np.sin(a)], axis=1)
            return _sample_perimeter(verts, n, rng)

        raise ValueError(f"unknown shape: {name!r}")

    # ── Fill (interior) samplers ───────────────────────────────────

    def _fill(self, name: str, n: int, rng) -> np.ndarray:
        if name == "circle":
            theta = rng.uniform(0, 2 * np.pi, n)
            r     = np.sqrt(rng.uniform(0.0, 1.0, n)) * 0.80
            return np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

        if name == "square":
            return rng.uniform(-0.75, 0.75, (n, 2))

        if name == "triangle":
            R  = 0.90
            v0 = np.array([0.0,             R       ])
            v1 = np.array([-R * _SQRT3 / 2, -R / 2.0])
            v2 = np.array([ R * _SQRT3 / 2, -R / 2.0])
            u, v = rng.uniform(0.0, 1.0, n), rng.uniform(0.0, 1.0, n)
            fold = u + v > 1
            u[fold], v[fold] = 1 - u[fold], 1 - v[fold]
            return u[:, None] * v0 + v[:, None] * v1 + (1 - u - v)[:, None] * v2

        if name == "star":
            outer, inner = 0.90, 0.38
            a  = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, 10, endpoint=False)
            r  = np.where(np.arange(10) % 2 == 0, outer, inner)
            vx, vy = r * np.cos(a), r * np.sin(a)
            return _rejection_sample(n, rng, outer,
                                     lambda px, py: _in_polygon(px, py, vx, vy))

        raise ValueError(f"unknown shape: {name!r}")

    # ── kept for backwards compat ──────────────────────────────────
    def _sample(self, name: str, n: int, rng) -> np.ndarray:
        return self._fill(name, n, rng)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.points[idx], self.labels[idx]


# ─── JsonShapeDataset ──────────────────────────────────────────────

class JsonShapeDataset(Dataset):
    """2D outline points sampled from JSON shape files.

    Each JSON file in data_dir is one class (label = file index, sorted).
    Points are sampled uniformly by arc length from line segments and
    cubic-spline curves defined in each file.

    Args:
        data_dir:    Directory containing *.json shape files.
        num_samples: Total number of 2-D points across all classes.
        noise:       Std of isotropic Gaussian noise added to each point.
        seed:        RNG seed.
    """

    def __init__(
        self,
        data_dir:    str   = "data",
        num_samples: int   = 60000,
        noise:       float = 0.005,
        seed:        int   = 42,
    ):
        rng        = np.random.default_rng(seed)
        json_files = sorted(Path(data_dir).glob("*.json"))
        K          = len(json_files)
        assert K > 0, f"No JSON files found in {data_dir}"

        n_each    = num_samples // K
        remainder = num_samples - n_each * K

        all_pts: list[np.ndarray] = []
        all_lbl: list[np.ndarray] = []

        for idx, filepath in enumerate(json_files):
            with open(filepath, encoding="utf-8") as f:
                shape = json.load(f)
            n   = n_each + (1 if idx < remainder else 0)
            pts = self._sample_shape(shape, n, rng)
            pts = pts + rng.standard_normal(pts.shape).astype(np.float32) * noise
            all_pts.append(pts)
            all_lbl.append(np.full(n, idx, dtype=np.int64))

        pts_arr = np.concatenate(all_pts)
        lbl_arr = np.concatenate(all_lbl)
        perm    = rng.permutation(len(pts_arr))

        self.points       = torch.from_numpy(pts_arr[perm])
        self.labels       = torch.from_numpy(lbl_arr[perm])
        self.num_clusters = K
        self.shape_names  = [f.stem for f in json_files]

    # ── Sampling helpers ──────────────────────────────────────────

    def _sample_shape(self, shape: dict, n: int, rng) -> np.ndarray:
        """Sample n points uniformly by arc length from a shape's outlines."""
        segments: list[tuple] = []

        for seg in shape.get("lines", []):
            p1, p2 = np.array(seg[0], dtype=np.float64), np.array(seg[1], dtype=np.float64)
            length = float(np.linalg.norm(p2 - p1))
            if length > 1e-8:
                segments.append(("line", p1, p2, length))

        for ctrl in shape.get("curves", []):
            pts = np.array(ctrl, dtype=np.float64)
            if len(pts) < 2:
                continue
            arc = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
            if arc > 1e-8:
                segments.append(("curve", pts, None, arc))

        if not segments:
            raw = np.array(shape.get("points", [[0.0, 0.0]]), dtype=np.float32)
            return raw[rng.integers(0, len(raw), size=n)]

        lengths = np.array([s[3] for s in segments])
        counts  = rng.multinomial(n, lengths / lengths.sum())

        sampled: list[np.ndarray] = []
        for seg, cnt in zip(segments, counts):
            if cnt == 0:
                continue
            if seg[0] == "line":
                _, p1, p2, _ = seg
                t   = rng.uniform(0.0, 1.0, cnt)
                pts = (p1 + t[:, None] * (p2 - p1)).astype(np.float32)
            else:
                _, ctrl, _, _ = seg
                pts = self._sample_spline(ctrl, cnt, rng)
            sampled.append(pts)

        return np.concatenate(sampled)

    @staticmethod
    def _sample_spline(ctrl: np.ndarray, n: int, rng) -> np.ndarray:
        """Sample n points uniformly by arc length along a cubic spline."""
        t     = np.zeros(len(ctrl))
        t[1:] = np.cumsum(np.linalg.norm(np.diff(ctrl, axis=0), axis=1))

        if len(ctrl) == 2:
            alpha = rng.uniform(0.0, 1.0, n)
            return (ctrl[0] + alpha[:, None] * (ctrl[1] - ctrl[0])).astype(np.float32)

        cs_x = CubicSpline(t, ctrl[:, 0], bc_type="natural")
        cs_y = CubicSpline(t, ctrl[:, 1], bc_type="natural")

        # Dense evaluation for arc-length reparameterisation
        t_dense = np.linspace(t[0], t[-1], 500)
        arc     = np.zeros(500)
        arc[1:] = np.cumsum(
            np.hypot(np.diff(cs_x(t_dense)), np.diff(cs_y(t_dense)))
        )

        s_vals  = rng.uniform(0.0, arc[-1], n)
        t_query = np.interp(s_vals, arc, t_dense)
        return np.stack([cs_x(t_query), cs_y(t_query)], axis=1).astype(np.float32)

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.points[idx], self.labels[idx]


class JsonPointShapeDataset(JsonShapeDataset):
    """Point-cloud dataset for PointNet-style denoising from JSON shapes.

    Each JSON file is one class. Each item returns a class-level point pool
    shaped (M, 2), plus its class label. Use a collate function such as
    VariablePointCountCollate to sample a minibatch-wide N from that pool.
    """

    def __init__(
        self,
        data_dir: str = "data",
        points_per_shape: int = 10000,
        noise: float = 0.005,
        split: str = "train",
        val_ratio: float = 0.1,
        items_per_epoch: int = 2048,
        seed: int = 42,
    ):
        if split not in {"train", "val", "all"}:
            raise ValueError(f"split must be 'train', 'val', or 'all', got {split!r}")

        rng = np.random.default_rng(seed)
        json_files = sorted(Path(data_dir).glob("*.json"))
        K = len(json_files)
        assert K > 0, f"No JSON files found in {data_dir}"

        self.clouds: list[torch.Tensor] = []
        self.shape_names = [f.stem for f in json_files]
        self.num_clusters = K
        self.items_per_epoch = int(items_per_epoch)

        all_pts: list[torch.Tensor] = []
        all_lbl: list[torch.Tensor] = []
        for idx, filepath in enumerate(json_files):
            with open(filepath, encoding="utf-8") as f:
                shape = json.load(f)

            pts = self._sample_shape(shape, points_per_shape, rng)
            pts = pts + rng.standard_normal(pts.shape).astype(np.float32) * noise
            perm = rng.permutation(len(pts))

            n_val = int(len(pts) * val_ratio)
            if split == "train":
                pts = pts[perm[n_val:]]
            elif split == "val":
                pts = pts[perm[:n_val]]
            else:
                pts = pts[perm]

            cloud = torch.from_numpy(pts.astype(np.float32))
            self.clouds.append(cloud)
            all_pts.append(cloud)
            all_lbl.append(torch.full((len(cloud),), idx, dtype=torch.long))

        self.points = torch.cat(all_pts, dim=0)
        self.labels = torch.cat(all_lbl, dim=0)

    def __len__(self) -> int:
        return self.items_per_epoch

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        cls = idx % self.num_clusters
        return self.clouds[cls], cls
