"""
Feature fusion module for PLAF.

Implements mask-guided feature aggregation (Paper Eq. 1-2).
This is the core innovation that creates spatially consistent, pixel-wise features.

Key idea: Instead of storing per-pixel features, we aggregate features within SAM masks
and store only mask-level features. This reduces storage while improving spatial consistency.
"""

from typing import List, Dict, Optional
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from plaf.core.types import MaskAnnotation, PixelFeatures

logger = logging.getLogger(__name__)


@dataclass
class FusionConfig:
    """
    Configuration for mask-feature fusion.

    Attributes:
        alpha_mode: How to compute pixel weights within masks
            - "constant": All pixels get equal weight (w=1)
            - "gaussian": Weight by Gaussian distance to mask center
            - "linear": Weight by linear distance to mask center
        sigma_ratio: For gaussian mode, sigma as ratio of mask size
        normalize_features: Whether to L2 normalize aggregated features
    """
    alpha_mode: str = "constant"  # "constant", "gaussian", "linear"
    sigma_ratio: float = 0.35
    normalize_features: bool = True

    def __post_init__(self):
        """Validate fusion configuration."""
        valid_modes = ["constant", "gaussian", "linear"]
        if self.alpha_mode not in valid_modes:
            raise ValueError(f"alpha_mode must be one of {valid_modes}")


class FeatureFusion:
    """
    Mask-guided feature fusion for PLAF.

    Implements the pixel-wise feature extraction scheme:
    1. Extract dense RADIO features (H, W, C)
    2. Generate SAM masks
    3. Aggregate features within each mask (Eq. 1-2)
    4. Create mask-indexed storage (mask_ids + mask_features)

    This creates the efficient storage representation from the paper.

    Attributes:
        config: Fusion configuration
        device: torch device
    """

    def __init__(
        self,
        config: Optional[FusionConfig] = None,
        device: str = "cuda"
    ):
        """
        Initialize feature fusion module.

        Args:
            config: Fusion configuration
            device: torch device
        """
        self.config = config or FusionConfig()
        self.device = torch.device(device)
        logger.info(f"Initialized FeatureFusion with mode={self.config.alpha_mode}")

    @torch.no_grad()
    def fuse_mask_features(
        self,
        features: torch.Tensor,
        masks: List[Dict]
    ) -> PixelFeatures:
        """
        Fuse dense features with SAM masks to create mask-indexed features.

        This is the main method that creates the PLAF representation.
        Uses boolean-indexed extraction so only masked pixels are touched
        per mask, instead of operating on the full (H, W, C) tensor.

        The feature aggregation follows SamRadio's approach:
        1. Normalize each pixel's feature individually
        2. Compute the average of normalized features per mask
        This ensures mask_features are in the same space as text features.

        Args:
            features: Dense RADIO features (H, W, C) - should be L2 normalized
            masks: List of SAM mask annotations

        Returns:
            pixel_features: PixelFeatures with mask_ids and mask_features
        """
        h, w, c = features.shape
        device = features.device
        feat_dtype = features.dtype

        masks = sorted(masks, key=lambda m: m["area"], reverse=True)

        mask_ids = np.zeros((h, w), dtype=np.uint16)
        mask_features_list = []

        need_coords = self.config.alpha_mode != "constant"
        if need_coords:
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij",
            )

        for mask_idx, mask_dict in enumerate(masks):
            seg = mask_dict["segmentation"]
            seg_t = torch.as_tensor(seg, device=device, dtype=torch.bool)

            pix = features[seg_t]  # (N, C) — only masked pixels
            n = pix.shape[0]
            if n == 0:
                mask_features_list.append(torch.zeros(c, device=device, dtype=feat_dtype))
                continue

            # SamRadio-style aggregation:
            # 1. Each pixel feature is already L2 normalized (from extract_features)
            # 2. For non-constant modes, we blend between mean and each pixel
            # 3. Then average the normalized/blended features per region

            if need_coords:
                my = yy[seg_t]
                mx = xx[seg_t]
                cy, cx = my.mean(), mx.mean()
                bbox = mask_dict["bbox"]
                x1, y1, x2, y2 = bbox
                scale = max(y2 - y1, x2 - x1, 1.0)
                dist = torch.sqrt(((my - cy) / scale) ** 2 + ((mx - cx) / scale) ** 2)

                # Compute alpha for spatial weighting
                if self.config.alpha_mode == "gaussian":
                    sigma = max(self.config.sigma_ratio, 1e-3)
                    alpha = 1.0 - torch.exp(-(dist ** 2) / (2.0 * sigma * sigma))
                else:  # linear
                    t = max(self.config.sigma_ratio, 1e-3)
                    alpha = (dist / t).clamp(0.0, 1.0)

                # Clamp alpha to avoid extreme values
                alpha = alpha.clamp(0.0, 1.0)

                # Compute region mean (already normalized)
                pix_mean = pix.mean(dim=0, keepdim=True)  # (1, C)

                # SamRadio blend: alpha * mean + (1-alpha) * pixel
                # Then re-normalize each blended pixel feature
                pix_blended = alpha.unsqueeze(1) * pix_mean + (1.0 - alpha.unsqueeze(1)) * pix
                pix_normalized = F.normalize(pix_blended, p=2, dim=-1)

                # Average the normalized blended features
                mask_feature = pix_normalized.mean(dim=0)
            else:
                # Constant mode: just average the (already normalized) pixel features
                mask_feature = pix.mean(dim=0)

            # Final normalization (mask_feature should already be ~unit norm, but ensure it)
            if self.config.normalize_features:
                mask_feature = F.normalize(mask_feature.unsqueeze(0), p=2, dim=-1).squeeze(0)

            mask_features_list.append(mask_feature.to(feat_dtype))
            mask_ids[seg] = mask_idx + 1

        mask_features = torch.stack(mask_features_list, dim=0)  # (K, C)
        # Use mean feature for background instead of zeros
        if len(mask_features_list) > 0:
            bg_feature = mask_features.mean(dim=0, keepdim=True) * 0.0  # Keep zeros for background
        else:
            bg_feature = torch.zeros((1, c), device=device, dtype=feat_dtype)
        mask_features = torch.cat([bg_feature, mask_features], dim=0)  # (K+1, C)

        return PixelFeatures(
            features=features.cpu(),
            mask_ids=mask_ids,
            mask_features=mask_features.cpu()
        )

    @torch.no_grad()
    def fuse_with_multi_mask_overlap(
        self,
        features: torch.Tensor,
        masks: List[Dict]
    ) -> PixelFeatures:
        """
        Fuse features allowing multiple mask assignments per pixel.

        Instead of assigning each pixel to a single mask, pixels can belong to
        multiple masks and their features are averaged.

        Args:
            features: Dense RADIO features (H, W, C)
            masks: List of SAM mask annotations

        Returns:
            pixel_features: PixelFeatures with mask_ids and mask_features
        """
        h, w, c = features.shape
        device = features.device
        feat_dtype = features.dtype

        mask_ids = np.zeros((h, w), dtype=np.uint16)
        aggregated_features = torch.zeros((h, w, c), device=device)
        pixel_mask_count = torch.zeros((h, w), device=device)

        masks = sorted(masks, key=lambda m: m["area"], reverse=True)

        need_coords = self.config.alpha_mode != "constant"
        if need_coords:
            yy, xx = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij",
            )

        mask_features_list = []

        for mask_idx, mask_dict in enumerate(masks):
            seg = mask_dict["segmentation"]
            seg_t = torch.as_tensor(seg, device=device, dtype=torch.bool)

            pix = features[seg_t]
            n = pix.shape[0]
            if n == 0:
                mask_features_list.append(torch.zeros(c, device=device, dtype=feat_dtype))
                continue

            pix_f = pix.float()
            avg = pix_f.mean(dim=0)

            if need_coords:
                my = yy[seg_t]
                mx = xx[seg_t]
                cy, cx = my.mean(), mx.mean()
                bbox = mask_dict["bbox"]
                x1, y1, x2, y2 = bbox
                scale = max(y2 - y1, x2 - x1, 1.0)
                dist = torch.sqrt(((my - cy) / scale) ** 2 + ((mx - cx) / scale) ** 2)

                if self.config.alpha_mode == "gaussian":
                    sigma = max(self.config.sigma_ratio, 1e-3)
                    weights = torch.exp(-(dist ** 2) / (2.0 * sigma * sigma))
                else:
                    weights = (1.0 - dist / max(self.config.sigma_ratio, 1e-3)).clamp(0.0, 1.0)

                w_sum = weights.sum().clamp(min=1e-8)
                mask_feature = (pix_f * weights.unsqueeze(1)).sum(0) / w_sum
            else:
                mask_feature = avg

            if self.config.normalize_features:
                mask_feature = F.normalize(mask_feature.unsqueeze(0), p=2, dim=-1).squeeze(0)

            mask_feature = mask_feature.to(feat_dtype)
            mask_features_list.append(mask_feature)

            aggregated_features[seg_t] += mask_feature.unsqueeze(0).expand(n, -1)
            pixel_mask_count[seg_t] += 1.0
            mask_ids[seg] = max(mask_ids[seg], mask_idx + 1)

        pixel_mask_count = torch.clamp(pixel_mask_count, min=1.0)
        final_features = aggregated_features / pixel_mask_count.unsqueeze(-1)

        mask_features = torch.stack(mask_features_list, dim=0)
        bg_feature = torch.zeros((1, c), device=device, dtype=feat_dtype)
        mask_features = torch.cat([bg_feature, mask_features], dim=0)

        return PixelFeatures(
            features=final_features.cpu(),
            mask_ids=mask_ids,
            mask_features=mask_features.cpu()
        )

    @staticmethod
    def compute_similarity_map(
        features_hw_c: torch.Tensor,
        query_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between features and query.

        Args:
            features_hw_c: Feature map (H, W, C)
            query_features: Query features (C,) or (N, C)

        Returns:
            similarity: Cosine similarity (H, W) or (H, W, N)
        """
        # Normalize if not already
        if query_features.ndim == 1:
            query_features = F.normalize(query_features.unsqueeze(0), p=2, dim=-1)
            # features_hw_c: (H, W, C)
            similarity = (features_hw_c * query_features).sum(dim=-1)  # (H, W)
        else:
            query_features = F.normalize(query_features, p=2, dim=-1)
            # (H, W, C) @ (N, C) -> (H, W, N)
            similarity = torch.matmul(features_hw_c, query_features.T)

        return similarity


def create_fusion_module(
    alpha_mode: str = "constant",
    sigma_ratio: float = 0.35,
    normalize_features: bool = True,
    device: str = "cuda"
) -> FeatureFusion:
    """
    Factory function to create fusion module.

    Args:
        alpha_mode: Mask weighting mode
        sigma_ratio: Gaussian sigma ratio
        normalize_features: Whether to L2 normalize
        device: torch device

    Returns:
        fusion: FeatureFusion module
    """
    config = FusionConfig(
        alpha_mode=alpha_mode,
        sigma_ratio=sigma_ratio,
        normalize_features=normalize_features
    )
    return FeatureFusion(config, device=device)


if __name__ == "__main__":
    # Test feature fusion
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="test_output")
    args = parser.parse_args()

    # Load image
    image = np.array(Image.open(args.image))
    print(f"Loaded image: {image.shape}")

    # Create dummy features (H, W, C=1024)
    h, w = image.shape[:2]
    features = torch.randn(h, w, 1024)

    # Create dummy masks
    masks = [{
        "segmentation": np.zeros((h, w), dtype=bool),
        "area": h * w // 4,
        "bbox": (w//4, h//4, 3*w//4, 3*h//4),
        "predicted_iou": 0.9
    }]
    masks[0]["segmentation"][h//4:3*h//4, w//4:3*w//4] = True

    # Test fusion
    fusion = FeatureFusion()
    pixel_features = fusion.fuse_mask_features(features, masks)

    print(f"Mask IDs shape: {pixel_features.mask_ids.shape}")
    print(f"Mask features shape: {pixel_features.mask_features.shape}")
    print(f"Number of masks: {pixel_features.num_masks}")

    # Test compression ratio
    ratio = pixel_features.compute_compression_ratio()
    print(f"Compression ratio: {ratio:.4%} (lower is better)")
