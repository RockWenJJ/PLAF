"""
Data type definitions for PLAF (Pixel-level Language-aligned Features).

This module defines the core data structures used throughout the PLAF framework,
including mask annotations, pixel features, and image annotations.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch


@dataclass
class MaskAnnotation:
    """Single mask annotation for a region in an image.

    Attributes:
        segmentation: Binary mask (H, W) where True indicates the region
        area: Number of pixels in the mask
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates
        score: Confidence score from the mask generator (if available)
        stability_score: Stability score from SAM (if available)
    """
    segmentation: np.ndarray  # (H, W) bool
    area: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    score: Optional[float] = None
    stability_score: Optional[float] = None


@dataclass
class MaskFeature:
    """Feature representation for a single mask region.

    Attributes:
        mask_id: Unique identifier for this mask
        feature: Aggregated feature vector (C,)
        area: Number of pixels in mask
        bbox: Bounding box coordinates
    """
    mask_id: int
    feature: torch.Tensor  # shape: (C,)
    area: int
    bbox: Tuple[int, int, int, int]


@dataclass
class PixelFeatures:
    """Pixel-level language-aligned features for an image.

    This class stores the dense pixel features along with mask-guided
    region features for efficient downstream processing.

    Attributes:
        features: Dense pixel features (H, W, C) where C is feature dimension
        mask_ids: Region ID map (H, W) where -1 is background, 0-N are regions
        mask_features: Pooled region features (K, C) where K is number of masks
        feature_dim: Dimension C of the feature vectors
    """
    features: torch.Tensor  # (H, W, C)
    mask_ids: np.ndarray   # (H, W) uint16, -1 for background
    mask_features: torch.Tensor  # (K, C)

    @property
    def feature_dim(self) -> int:
        """Feature dimension C."""
        return self.features.shape[2]

    @property
    def num_regions(self) -> int:
        """Number of regions K."""
        return self.mask_features.shape[0]

    @property
    def height(self) -> int:
        """Image height H."""
        return self.features.shape[0]

    @property
    def width(self) -> int:
        """Image width W."""
        return self.features.shape[1]

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Feature shape (H, W, C)."""
        return self.features.shape

    def get_pixel_feature(self, u: int, v: int) -> torch.Tensor:
        """Get mask-aggregated feature for pixel (u, v)."""
        mask_id = self.mask_ids[u, v]
        if mask_id == 255:  # background (using 255 to represent -1 in uint8)
            return torch.zeros(self.mask_features.shape[1])
        return self.mask_features[mask_id]


@dataclass
class ImageAnnotation:
    """Complete annotation data for an image.

    Attributes:
        rgb: RGB image (H, W, 3) uint8
        depth: Depth map (H, W) float32, in meters
        mask_ids: Region ID map (H, W) uint16, -1 for background
        semantic: Optional semantic segmentation (H, W) int32
        image_path: Optional path to the original image file
    """
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32
    mask_ids: np.ndarray  # (H, W) uint16
    semantic: Optional[np.ndarray] = None  # (H, W) int32
    image_path: Optional[str] = None

    @property
    def height(self) -> int:
        """Image height H."""
        return self.rgb.shape[0]

    @property
    def width(self) -> int:
        """Image width W."""
        return self.rgb.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        """Image shape (H, W)."""
        return (self.height, self.width)

    @property
    def has_semantic(self) -> bool:
        """Whether semantic segmentation is available."""
        return self.semantic is not None


@dataclass
class TextQuery:
    """Text query for feature retrieval.

    Attributes:
        text: Query text string
        features: Encoded text features (1, D) or (D,)
    """
    text: str
    features: torch.Tensor  # (1, D) or (D,)

    @property
    def feature_dim(self) -> int:
        """Feature dimension D."""
        if self.features.dim() == 2:
            return self.features.shape[1]
        return self.features.shape[0]


@dataclass
class SimilarityResult:
    """Result of computing similarity between features and a query.

    Attributes:
        heatmap: Similarity heatmap (H, W) in range [0, 1]
        max_similarity: Maximum similarity score
        max_location: (y, x) location of maximum similarity
        region_similarities: Per-region similarity scores (K,)
    """
    heatmap: np.ndarray  # (H, W)
    max_similarity: float
    max_location: Tuple[int, int]  # (y, x)
    region_similarities: np.ndarray  # (K,)

    @property
    def shape(self) -> Tuple[int, int]:
        """Heatmap shape (H, W)."""
        return self.heatmap.shape

    def get_top_k_regions(self, k: int) -> np.ndarray:
        """Get indices of top K regions by similarity."""
        if k > len(self.region_similarities):
            k = len(self.region_similarities)
        return np.argsort(self.region_similarities)[-k:][::-1]


@dataclass
class SceneAnnotation:
    """Complete annotation for a scene with multiple views.

    Attributes:
        scene_name: Name of the scene
        views: List of ImageAnnotation for each view
    """
    scene_name: str
    views: List[ImageAnnotation]

    @property
    def num_views(self) -> int:
        """Number of views."""
        return len(self.views)

    def add_view(self, view: "ImageAnnotation"):
        """Add a view to the scene."""
        self.views.append(view)
