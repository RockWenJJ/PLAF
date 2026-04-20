"""
3D feature pool storage for PLAF.

Implements the efficient 3D storage scheme from the paper (Eqs. 7-10).

Key idea: Instead of storing per-point features, store:
    1. A feature pool: (M, C) float32 - unique features after fusion
    2. Point references: (N,) uint32 - index into pool for each 3D point

This achieves >99% storage reduction for large-scale 3D scenes.
"""

from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from dataclasses import dataclass
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PointObservation:
    """
    A single 3D point observation from a 2D view.

    Represents the back-projection of a 2D pixel to 3D space.

    Attributes:
        point_3d: 3D coordinate (x, y, z) in meters
        mask_id: The mask ID this point belongs to (from 2D)
        view_index: Which view this observation came from
        pixel_coord: Original (u, v) pixel coordinate
        confidence: Observation confidence (e.g., based on depth)
    """
    point_3d: np.ndarray  # (3,)
    mask_id: int
    view_index: int
    pixel_coord: Tuple[int, int]  # (u, v)
    confidence: float = 1.0

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array for storage."""
        return np.array([
            self.point_3d[0], self.point_3d[1], self.point_3d[2],
            self.mask_id, self.view_index,
            self.pixel_coord[0], self.pixel_coord[1],
            self.confidence
        ], dtype=np.float32)

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> "PointObservation":
        """Create from numpy array."""
        return cls(
            point_3d=arr[:3],
            mask_id=int(arr[3]),
            view_index=int(arr[4]),
            pixel_coord=(int(arr[5]), int(arr[6])),
            confidence=float(arr[7])
        )


class FeaturePool3D:
    """
    3D feature pool with point-wise indexing.

    Implements the storage scheme from Eqs. 7-10 of the paper.

    Storage components:
        points: (N, 3) float32 - xyz coordinates
        feature_refs: (N,) uint32 - indices into feature pool
        feature_pool: (M, C) float32 - unique feature vectors

    Attributes:
        num_points: Number of 3D points N
        feature_dim: Feature dimension C
        pool_size: Number of unique features M
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        fusion_distance: float = 0.05
    ):
        """
        Initialize 3D feature pool.

        Args:
            feature_dim: Feature dimension (C)
            fusion_distance: Distance threshold for feature fusion (meters)
        """
        self.feature_dim = feature_dim
        self.fusion_distance = fusion_distance

        # Storage arrays
        self.points = []  # List of (3,) arrays
        self.feature_refs = []  # List of int indices
        self.feature_pool = None  # Will be (M, C) tensor

        # Mask ID tracking (for each point)
        self.mask_ids = []

    @property
    def num_points(self) -> int:
        """Get number of points."""
        return len(self.points)

    @property
    def pool_size(self) -> int:
        """Get feature pool size."""
        if self.feature_pool is None:
            return 0
        return self.feature_pool.shape[0]

    def add_observation(
        self,
        obs: PointObservation,
        mask_feature: torch.Tensor
    ):
        """
        Add a single observation from a 2D view.

        Args:
            obs: Point observation with 3D coordinate and mask reference
            mask_feature: Feature vector for the mask (C,)
        """
        self.points.append(obs.point_3d)
        self.mask_ids.append(obs.mask_id)
        self.feature_refs.append(-1)  # Temporary ref

        if self.feature_pool is None:
            # Initialize pool with first feature
            device = mask_feature.device
            self.feature_pool = mask_feature.unsqueeze(0).clone()
        else:
            # Check if we should fuse with existing feature
            self._fuse_feature_to_pool(obs.mask_id, mask_feature, obs.view_index)

    def add_observations(
        self,
        observations: List[PointObservation],
        mask_features: Dict[int, torch.Tensor]
    ):
        """
        Add multiple observations from a view.

        Args:
            observations: List of point observations
            mask_features: Map from mask_id to feature vector
        """
        for obs in observations:
            mask_feature = mask_features.get(obs.mask_id)
            if mask_feature is not None:
                self.add_observation(obs, mask_feature)

    def _fuse_feature_to_pool(
        self,
        mask_id: int,
        new_feature: torch.Tensor,
        view_index: int
    ):
        """
        Fuse a new feature into the pool.

        Strategy:
            1. If mask_id already exists in pool, use that
            2. Otherwise, find similar feature in pool
            3. If similar enough, merge; otherwise add new

        Args:
            mask_id: Mask identifier
            new_feature: New feature vector (C,)
            view_index: Source view index
        """
        # L2 normalize for comparison
        new_feature_norm = torch.nn.functional.normalize(new_feature.unsqueeze(0), p=2, dim=-1)

        # Find similar features in pool
        similarity = torch.matmul(
            self.feature_pool,
            new_feature_norm.T
        ).squeeze(-1)  # (M,)

        max_sim_idx = similarity.argmax().item()
        max_sim = similarity[max_sim_idx].item()

        # Threshold for fusion (can be tuned)
        fusion_threshold = 0.95

        if max_sim > fusion_threshold:
            # Fuse with existing feature
            # Use existing pool entry
            ref_idx = max_sim_idx
        else:
            # Add new feature to pool
            ref_idx = self.pool_size
            self.feature_pool = torch.cat([
                self.feature_pool,
                new_feature_norm
            ], dim=0)

        # Store reference
        # Find the point we just added
        if len(self.feature_refs) < len(self.points):
            self.feature_refs.append(ref_idx)
        else:
            self.feature_refs[-1] = ref_idx

    def finalize(self):
        """
        Finalize the storage after all observations added.

        Converts lists to tensors and computes final stats.
        """
        if not self.points:
            logger.warning("No points added to feature pool")
            return

        # Convert to tensors
        self.points = np.stack(self.points, axis=0)  # (N, 3)
        self.feature_refs = np.array(self.feature_refs, dtype=np.int64)  # Use int64 for PyTorch indexing
        self.mask_ids = np.array(self.mask_ids, dtype=np.uint16)

        logger.info(
                    f"Finalized: {self.num_points} points, "
                    f"{self.pool_size} unique features"
                )

    def get_point_features(self) -> torch.Tensor:
        """
        Get features for all points by dereferencing pool.

        Returns:
            point_features: Features (N, C)
        """
        if self.feature_pool is None:
            raise RuntimeError("No features in pool")

        # Dereference: each point uses its indexed feature
        device = self.feature_pool.device
        point_features = self.feature_pool[self.feature_refs]  # (N, C)
        return point_features

    def query_by_mask_id(self, mask_id: int) -> torch.Tensor:
        """
        Query features by mask ID.

        Args:
            mask_id: Mask identifier

        Returns:
            features: Features for points with this mask (M, C)
        """
        mask_indices = np.where(self.mask_ids == mask_id)[0]
        if len(mask_indices) == 0:
            return torch.zeros((0, self.feature_dim))

        refs = self.feature_refs[mask_indices]
        return self.feature_pool[refs]

    def query_by_text(
        self,
        text_feature: torch.Tensor,
        top_k: int = 100
    ) -> Tuple[np.ndarray, torch.Tensor]:
        """
        Query points by text similarity.

        Args:
            text_feature: Text-encoded feature vector (C,)
            top_k: Return top-k most similar points

        Returns:
            points: Top-k point coordinates (K, 3)
            similarities: Similarity scores (K,)
        """
        if self.feature_pool is None:
            return np.zeros((0, 3)), torch.zeros(0)

        # Compute similarity to all pooled features
        pool_sim = torch.matmul(
            self.feature_pool,
            text_feature.T
        ).squeeze(-1)  # (M,)

        # Get similarity for each point via dereferencing
        point_sim = pool_sim[self.feature_refs]  # (N,)

        # Get top-k
        top_indices = torch.from_numpy(
            np.argpartition(-point_sim.cpu().numpy(), top_k)[:top_k]
        )
        top_points = self.points[top_indices.numpy()]
        top_sims = point_sim[top_indices]

        return top_points, top_sims

    def compute_storage_sizes(
        self,
        bytes_per_float: int = 4,
        bytes_per_ref: int = 4
    ) -> Dict[str, float]:
        """
        Compute storage sizes for per-point vs feature pool storage.

        Args:
            bytes_per_float: Bytes per float (4 for FP32)
            bytes_per_ref: Bytes per reference (4 for uint32)

        Returns:
            sizes: Dictionary with storage sizes in MB
        """
        n = self.num_points
        m = self.pool_size
        c = self.feature_dim

        # Per-point storage: S_dense^3D = N * C * b_f (Eq. 8)
        s_dense = n * c * bytes_per_float

        # Feature pool: S_index^3D = N * b_r + M * C * b_f (Eq. 9)
        s_index = n * bytes_per_ref + m * c * bytes_per_float

        # Compression ratio (Eq. 10)
        ratio = s_index / s_dense if s_dense > 0 else 0

        return {
            "num_points": n,
            "pool_size": m,
            "per_point_mb": s_dense / (1024 * 1024),
            "feature_pool_mb": s_index / (1024 * 1024),
            "compression_ratio": ratio,
            "reduction_percent": (1 - ratio) * 100
        }

    def save(self, path: str):
        """Save to disk."""
        np.save(f"{path}_points.npy", self.points)
        np.save(f"{path}_refs.npy", self.feature_refs)
        np.save(f"{path}_mask_ids.npy", self.mask_ids)
        torch.save(self.feature_pool, f"{path}_pool.pt")
        logger.info(f"Saved 3D storage to {path}")

    @classmethod
    def load(cls, path: str, feature_dim: int = 1024):
        """Load from disk."""
        data = cls(feature_dim=feature_dim)
        data.points = np.load(f"{path}_points.npy")
        data.feature_refs = np.load(f"{path}_refs.npy")
        data.mask_ids = np.load(f"{path}_mask_ids.npy")
        data.feature_pool = torch.load(f"{path}_pool.pt")
        logger.info(f"Loaded 3D storage: {data.num_points} points")
        return data


def backproject_pixels_to_3d(
    rgb: np.ndarray,
    depth: np.ndarray,
    camera_intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    mask_ids: np.ndarray,
    max_depth: float = 5.0
) -> List[PointObservation]:
    """
    Back-project 2D pixels to 3D coordinates.

    Args:
        rgb: RGB image (H, W, 3)
        depth: Depth map (H, W) in meters
        camera_intrinsics: Camera intrinsics (3, 3)
        camera_pose: Camera pose (4, 4) world-to-camera
        mask_ids: Mask ID map (H, W)
        max_depth: Maximum depth for projection

    Returns:
        observations: List of PointObservation
    """
    h, w = rgb.shape[:2]
    observations = []

    # Camera intrinsics
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # ScanNet pose format is camera-to-world (4x4 matrix)
    # p_world = R_cw @ p_cam + t_cw
    R_cw = camera_pose[:3, :3]
    t_cw = camera_pose[:3, 3]

    for v in range(h):
        for u in range(w):
            d = depth[v, u]

            # Skip invalid depths
            if d <= 0 or d > max_depth:
                continue

            # Back-project to camera coordinates
            x_c = (u - cx) * d / fx
            y_c = (v - cy) * d / fy
            z_c = d

            # Transform to world coordinates using camera-to-world pose
            p_c = np.array([x_c, y_c, z_c])
            p_w = R_cw @ p_c + t_cw  # p_world = R_cw @ p_cam + t_cw

            # Create observation
            obs = PointObservation(
                point_3d=p_w,
                mask_id=int(mask_ids[v, u]),
                view_index=0,  # Will be updated by caller
                pixel_coord=(u, v),
                confidence=max(0, 1 - d / max_depth)  # Closer = higher confidence
            )
            observations.append(obs)

    return observations


if __name__ == "__main__":
    # Test 3D storage
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-points", type=int, default=1000000)
    parser.add_argument("--pool-size", type=int, default=10000)
    args = parser.parse_args()

    storage = FeaturePool3D()

    # Simulate data
    storage.points = np.randn(args.num_points, 3).astype(np.float32)
    storage.mask_ids = np.random.randint(0, 200, args.num_points, dtype=np.uint16)
    storage.feature_refs = np.random.randint(0, args.pool_size, args.num_points, dtype=np.uint32)
    storage.feature_pool = torch.randn(args.pool_size, 1024)

    # Compute storage
    sizes = storage.compute_storage_sizes()
    print(f"Points: {sizes['num_points']}")
    print(f"Unique features: {sizes['pool_size']}")
    print(f"Per-point storage: {sizes['per_point_mb']:.2f} MB")
    print(f"Feature pool storage: {sizes['feature_pool_mb']:.2f} MB")
    print(f"Compression ratio: {sizes['compression_ratio']:.4%}")
    print(f"Storage reduction: {sizes['reduction_percent']:.1f}%")
