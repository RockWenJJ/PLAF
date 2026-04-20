"""
SAM mask generator for PLAF.

This module implements class-agnostic instance mask generation using the
Segment Anything Model (SAM).
"""

from typing import List, Optional, Dict, Any, Tuple
import warnings

import cv2
import numpy as np
import torch

from ..utils.model_loader import load_sam_mask_generator, ensure_device, get_global_cache
from ..utils.config import SamConfig
from .types import MaskAnnotation


class SamMaskGenerator:
    """SAM (Segment Anything Model) mask generator.

    This class generates class-agnostic instance masks for images using
    the Segment Anything Model.

    Attributes:
        mask_generator: SAM automatic mask generator instance
        config: Configuration for mask generation
        device: Device to run inference on

    Examples:
        >>> generator = SamMaskGenerator(model_type="vit_h", device="cuda")
        >>> masks = generator.generate_masks(image)
        >>> print(f"Generated {len(masks)} masks")
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.85,
        crop_n_layers: int = 0,
        crop_n_points_downscale_factor: int = 2,
        min_mask_region_area: int = 200,
        stability_score_offset: float = 1.0,
        stability_score_thresh: float = 0.92,
        box_nms_thresh: float = 0.7,
        device: str = "cuda",
        use_cache: bool = True,
    ):
        """Initialize SAM mask generator.

        Args:
            model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            checkpoint_path: Explicit checkpoint path
            checkpoint_dir: Directory for checkpoints
            points_per_side: Number of grid points for mask generation
            pred_iou_thresh: IoU threshold for keeping masks
            crop_n_layers: Number of crop layers
            crop_n_points_downscale_factor: Downscale factor for crop points
            min_mask_region_area: Minimum area for mask regions
            stability_score_offset: Stability score offset
            stability_score_thresh: Stability score threshold
            box_nms_thresh: NMS threshold for box predictions
            device: Device to run on ("cuda" or "cpu")
            use_cache: Whether to use global model cache
        """
        self.config = SamConfig(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            crop_n_layers=crop_n_layers,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
            stability_score_offset=stability_score_offset,
            stability_score_thresh=stability_score_thresh,
            box_nms_thresh=box_nms_thresh,
            device=device,
        )

        self.device = ensure_device(device)
        self._checkpoint_dir = checkpoint_dir
        self._use_cache = use_cache

        # Load mask generator
        self.mask_generator = self._load_mask_generator()

    def _load_mask_generator(self):
        """Load SAM mask generator."""
        if self._use_cache:
            cache = get_global_cache()
            return cache.get_sam_mask_generator(
                model_type=self.config.model_type,
                checkpoint_path=self.config.checkpoint_path,
                checkpoint_dir=self._checkpoint_dir,
                points_per_side=self.config.points_per_side,
                pred_iou_thresh=self.config.pred_iou_thresh,
                crop_n_layers=self.config.crop_n_layers,
                crop_n_points_downscale_factor=self.config.crop_n_points_downscale_factor,
                min_mask_region_area=self.config.min_mask_region_area,
                stability_score_offset=self.config.stability_score_offset,
                stability_score_thresh=self.config.stability_score_thresh,
                box_nms_thresh=self.config.box_nms_thresh,
                device=self.device,
            )
        else:
            return load_sam_mask_generator(
                model_type=self.config.model_type,
                checkpoint_path=self.config.checkpoint_path,
                checkpoint_dir=self._checkpoint_dir,
                points_per_side=self.config.points_per_side,
                pred_iou_thresh=self.config.pred_iou_thresh,
                crop_n_layers=self.config.crop_n_layers,
                crop_n_points_downscale_factor=self.config.crop_n_points_downscale_factor,
                min_mask_region_area=self.config.min_mask_region_area,
                stability_score_offset=self.config.stability_score_offset,
                stability_score_thresh=self.config.stability_score_thresh,
                box_nms_thresh=self.config.box_nms_thresh,
                device=self.device,
            )

    def generate_masks(
        self,
        image: np.ndarray,
        return_type: str = "dict",
    ) -> List[Any]:
        """Generate class-agnostic instance masks for an image.

        Args:
            image: RGB image as numpy array (H, W, 3) uint8
            return_type: Type of masks to return ("dict", "annotation", or "both")

        Returns:
            List of masks (dict, MaskAnnotation, or tuple based on return_type)

        Raises:
            ValueError: If image is invalid or return_type is unknown
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be (H, W, 3), got shape {image.shape}")

        if image.dtype != np.uint8:
            warnings.warn(f"Image dtype is {image.dtype}, converting to uint8")
            image = image.astype(np.uint8)

        # Generate masks using SAM
        masks = self.mask_generator.generate(image)

        if return_type == "dict":
            return masks
        elif return_type == "annotation":
            return [MaskAnnotation.from_sam_dict(m) for m in masks]
        elif return_type == "both":
            annotations = [MaskAnnotation.from_sam_dict(m) for m in masks]
            return list(zip(masks, annotations))
        else:
            raise ValueError(
                f"Unknown return_type: {return_type}. "
                f"Must be 'dict', 'annotation', or 'both'"
            )

    def filter_by_area(
        self,
        masks: List[Any],
        min_area: int = 100,
        max_area: int = 100000,
        return_indices: bool = False,
    ) -> List[Any]:
        """Filter masks by area.

        Args:
            masks: List of masks (dict or MaskAnnotation)
            min_area: Minimum area threshold
            max_area: Maximum area threshold
            return_indices: Whether to return indices instead of filtered masks

        Returns:
            Filtered list of masks or indices
        """
        filtered = []
        indices = []

        for i, mask in enumerate(masks):
            area = mask.area if hasattr(mask, "area") else mask.get("area", 0)
            if min_area <= area <= max_area:
                filtered.append(mask)
                indices.append(i)

        return indices if return_indices else filtered

    def filter_by_score(
        self,
        masks: List[Any],
        min_score: float = 0.5,
        return_indices: bool = False,
    ) -> List[Any]:
        """Filter masks by confidence score.

        Args:
            masks: List of masks (dict or MaskAnnotation)
            min_score: Minimum confidence score threshold
            return_indices: Whether to return indices instead of filtered masks

        Returns:
            Filtered list of masks or indices
        """
        filtered = []
        indices = []

        for i, mask in enumerate(masks):
            score = None
            if hasattr(mask, "score"):
                score = mask.score
            elif hasattr(mask, "predicted_iou"):
                score = mask.predicted_iou
            else:
                score = mask.get("predicted_iou", mask.get("score", 0))

            if score is not None and score >= min_score:
                filtered.append(mask)
                indices.append(i)

        return indices if return_indices else filtered

    def filter_by_stability(
        self,
        masks: List[Any],
        min_stability: float = 0.8,
        return_indices: bool = False,
    ) -> List[Any]:
        """Filter masks by stability score.

        Args:
            masks: List of masks (dict or MaskAnnotation)
            min_stability: Minimum stability score threshold
            return_indices: Whether to return indices instead of filtered masks

        Returns:
            Filtered list of masks or indices
        """
        filtered = []
        indices = []

        for i, mask in enumerate(masks):
            stability = None
            if hasattr(mask, "stability_score"):
                stability = mask.stability_score
            else:
                stability = mask.get("stability_score", 0)

            if stability is not None and stability >= min_stability:
                filtered.append(mask)
                indices.append(i)

        return indices if return_indices else filtered

    def masks_to_index_map(
        self,
        masks: List[Any],
        height: int,
        width: int,
        background_id: int = -1,
        overlap_strategy: str = "first",
    ) -> np.ndarray:
        """Convert list of masks to a single index map.

        Args:
            masks: List of masks (dict or MaskAnnotation)
            height: Height of output map
            width: Width of output map
            background_id: ID value for background pixels
            overlap_strategy: How to handle overlapping masks
                - "first": Keep first mask (original order)
                - "largest": Keep largest mask
                - "smallest": Keep smallest mask

        Returns:
            Index map of shape (H, W) with dtype int32
        """
        # Initialize with background ID
        index_map = np.full((height, width), background_id, dtype=np.int32)

        if not masks:
            return index_map

        # Sort masks based on overlap strategy
        if overlap_strategy == "largest":
            masks_sorted = sorted(masks, key=lambda m: m.area if hasattr(m, "area") else m.get("area", 0), reverse=True)
        elif overlap_strategy == "smallest":
            masks_sorted = sorted(masks, key=lambda m: m.area if hasattr(m, "area") else m.get("area", 0))
        else:  # "first"
            masks_sorted = masks

        # Assign indices
        for mask_idx, mask in enumerate(masks_sorted):
            # Get segmentation
            if hasattr(mask, "segmentation"):
                seg = mask.segmentation
            else:
                seg = mask.get("segmentation", None)

            if seg is None:
                continue

            # Ensure boolean
            if seg.dtype != bool:
                seg = seg.astype(bool)

            # Resize if needed
            if seg.shape != (height, width):
                seg = cv2.resize(
                    seg.astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            # Assign pixels (only where still background for non-overlapping)
            if overlap_strategy in ["first", "largest", "smallest"]:
                index_map[seg & (index_map == background_id)] = mask_idx
            else:
                index_map[seg] = mask_idx

        return index_map

    def merge_overlapping_masks(
        self,
        masks: List[Any],
        iou_threshold: float = 0.8,
    ) -> List[Any]:
        """Merge overlapping masks based on IoU threshold.

        Args:
            masks: List of masks
            iou_threshold: IoU threshold for merging

        Returns:
            List of merged masks
        """
        if len(masks) <= 1:
            return masks

        # Sort by area (descending)
        masks_sorted = sorted(
            masks,
            key=lambda m: m.area if hasattr(m, "area") else m.get("area", 0),
            reverse=True
        )

        merged = []
        merged_masks = []

        for mask in masks_sorted:
            seg = mask.segmentation if hasattr(mask, "segmentation") else mask.get("segmentation")

            if seg is None:
                continue

            # Ensure boolean
            if seg.dtype != bool:
                seg = seg.astype(bool)

            # Check overlap with already merged masks
            should_add = True
            for merged_seg in merged_masks:
                # Compute IoU
                intersection = np.logical_and(seg, merged_seg).sum()
                union = np.logical_or(seg, merged_seg).sum()

                if union > 0:
                    iou = intersection / union
                    if iou > iou_threshold:
                        should_add = False
                        break

            if should_add:
                merged.append(mask)
                merged_masks.append(seg)

        return merged

    @property
    def model_type(self) -> str:
        """Get SAM model type."""
        return self.config.model_type

    def __call__(self, image: np.ndarray) -> List[Any]:
        """Convenience method for mask generation.

        Args:
            image: RGB image (H, W, 3) uint8

        Returns:
            List of masks
        """
        return self.generate_masks(image)

    def to(self, device: str):
        """Move model to different device.

        Note:
            SAM mask generator doesn't support direct device transfer.
            Recreate the generator with new device instead.
        """
        warnings.warn(
            "SAM mask generator doesn't support device transfer. "
            "Create a new instance with desired device."
        )
        return self


def generate_masks_for_batch(
    images: List[np.ndarray],
    generator: SamMaskGenerator,
) -> List[List[Any]]:
    """Generate masks for a batch of images.

    Args:
        images: List of RGB images (H, W, 3) uint8
        generator: SamMaskGenerator instance

    Returns:
        List of mask lists, one per image
    """
    return [generator.generate_masks(img) for img in images]
