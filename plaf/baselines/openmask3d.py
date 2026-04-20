"""
OpenMask3D baseline implementation.

Based on: "OpenMask3D: Open-Vocabulary 3D Instance Segmentation"
Paper: https://arxiv.org/abs/2306.13631
GitHub: https://github.com/OpenMask3D/openmask3d

Key idea: Multi-level CLIP features with SAM mask refinement for open-vocabulary queries.
"""

from typing import List, Dict, Optional, Tuple
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def mask2box(mask: torch.Tensor) -> Tuple[int, int, int, int]:
    """Convert binary mask to bounding box.

    Args:
        mask: Binary mask (H, W)

    Returns:
        x1, y1, x2, y2: Bounding box coordinates (inclusive)
    """
    rows = torch.any(mask, dim=1)
    cols = torch.any(mask, dim=0)
    if not rows.any() or not cols.any():
        return 0, 0, 0, 0

    rmin, rmax = torch.where(rows)[0][[0, -1]]
    cmin, cmax = torch.where(cols)[0][[0, -1]]

    return rmin.item(), cmin.item(), rmax.item(), cmax.item()


def mask2box_multi_level(mask: torch.Tensor, level: int, expansion_ratio: float) -> Tuple[int, int, int, int]:
    """Convert binary mask to expanded bounding box at multiple levels.

    Args:
        mask: Binary mask (H, W)
        level: Expansion level (0 = no expansion)
        expansion_ratio: Expansion ratio per level

    Returns:
        x1, y1, x2, y2: Bounding box coordinates
    """
    y1, x1, y2, x2 = mask2box(mask)
    h, w = mask.shape

    if level == 0:
        return x1, y1, x2, y2

    x_exp = int(abs(x2 - x1) * expansion_ratio) * level
    y_exp = int(abs(y2 - y1) * expansion_ratio) * level

    return (
        max(0, x1 - x_exp),
        max(0, y1 - y_exp),
        min(w - 1, x2 + x_exp),
        min(h - 1, y2 + y_exp)
    )


class OpenMask3DExtractor:
    """
    OpenMask3D feature extractor for 2D images.

    Implements multi-level CLIP feature extraction with SAM mask refinement,
    following the original OpenMask3D approach adapted for single 2D images.

    Attributes:
        feature_dim: CLIP feature dimension (512 for ViT-B/32, 768 for ViT-L/14)
    """

    def __init__(
        self,
        device: str = "cuda",
        clip_model: str = "ViT-B/32",
        sam_model_type: str = "vit_h",
        multi_level_expansion_ratio: float = 0.1,
        num_levels: int = 3,
        num_random_rounds: int = 10,
        num_selected_points: int = 5,
    ):
        """
        Initialize OpenMask3D extractor.

        Args:
            device: torch device
            clip_model: CLIP model name (ViT-B/32 or ViT-L/14)
            sam_model_type: SAM model type (vit_h, vit_l, vit_b)
            multi_level_expansion_ratio: Crop expansion ratio per level
            num_levels: Number of multi-level crops to extract
            num_random_rounds: Number of SAM sampling rounds
            num_selected_points: Number of points per SAM round
        """
        self.device = torch.device(device)
        self.clip_model_name = clip_model
        self.sam_model_type = sam_model_type
        self.multi_level_expansion_ratio = multi_level_expansion_ratio
        self.num_levels = num_levels
        self.num_random_rounds = num_random_rounds
        self.num_selected_points = num_selected_points

        # Initialize models (lazy loading)
        self.clip_model = None
        self.clip_preprocess = None
        self.sam_predictor = None

        logger.info("OpenMask3D extractor initialized (models loaded on first use)")

    def _load_clip(self):
        """Load CLIP model."""
        if self.clip_model is None:
            try:
                import clip
                self.clip_model, self.clip_preprocess = clip.load(
                    self.clip_model_name, device=self.device
                )
                self.clip_model.eval()
                logger.info(f"CLIP {self.clip_model_name} loaded")
            except ImportError:
                raise RuntimeError(
                    "CLIP not installed. Install with: pip install openai-clip"
                )

    def _load_sam(self):
        """Load SAM model."""
        if self.sam_predictor is None:
            from segment_anything import sam_model_registry
            from plaf.utils.model_loader import resolve_sam_checkpoint

            sam_checkpoint = resolve_sam_checkpoint(
                self.sam_model_type, checkpoint_dir="./checkpoints"
            )
            sam = sam_model_registry[self.sam_model_type](checkpoint=sam_checkpoint)
            sam.to(self.device)
            sam.eval()

            from segment_anything import SamPredictor
            self.sam_predictor = SamPredictor(sam)
            logger.info(f"SAM {self.sam_model_type} loaded")

    @torch.no_grad()
    def refine_mask_with_sam(
        self,
        image: np.ndarray,
        initial_mask: np.ndarray
    ) -> np.ndarray:
        """Refine a mask using SAM with point sampling.

        Args:
            image: RGB image (H, W, 3)
            initial_mask: Initial binary mask (H, W)

        Returns:
            refined_mask: Refined binary mask (H, W)
        """
        self._load_sam()

        # Set image for SAM predictor
        self.sam_predictor.set_image(image)

        # Get point coordinates from the initial mask
        point_coords = np.transpose(np.where(initial_mask))

        if point_coords.shape[0] == 0:
            return initial_mask

        # SAM point sampling: try multiple random subsets
        best_score = 0
        best_mask = initial_mask.astype(bool)

        point_coords_new = point_coords.copy()
        # SAM expects (x, y) format, so swap columns
        point_coords_new[:, [0, 1]] = point_coords_new[:, [1, 0]]

        for _ in range(self.num_random_rounds):
            np.random.shuffle(point_coords_new)
            n_points = min(self.num_selected_points, point_coords_new.shape[0])

            masks, scores, _ = self.sam_predictor.predict(
                point_coords=point_coords_new[:n_points],
                point_labels=np.ones(n_points),
                multimask_output=False,
            )

            if scores[0] > best_score:
                best_score = scores[0]
                best_mask = masks[0]

        return best_mask

    @torch.no_grad()
    def extract_mask_features(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        use_sam_refinement: bool = True
    ) -> torch.Tensor:
        """Extract CLIP features for a single mask with multi-level cropping.

        Args:
            image: RGB image (H, W, 3)
            mask: Binary mask (H, W)
            use_sam_refinement: Whether to refine mask with SAM

        Returns:
            mask_features: Aggregated CLIP features (C,)
        """
        self._load_clip()

        h, w = image.shape[:2]
        pil_image = Image.fromarray(image)

        # Optionally refine mask with SAM
        if use_sam_refinement:
            refined_mask = self.refine_mask_with_sam(image, mask)
        else:
            refined_mask = mask.astype(bool)

        mask_t = torch.from_numpy(refined_mask)

        # Check if mask has any pixels
        if not mask_t.any():
            return torch.zeros(512, device=self.device)  # Return zeros for empty mask

        # Extract multi-level crops
        images_crops = []

        for level in range(self.num_levels):
            x1, y1, x2, y2 = mask2box_multi_level(mask_t, level, self.multi_level_expansion_ratio)

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Crop image
            cropped_img = pil_image.crop((x1, y1, x2 + 1, y2 + 1))

            # Resize small crops to avoid CLIP issues
            if cropped_img.size[0] < 224 or cropped_img.size[1] < 224:
                cropped_img = cropped_img.resize((224, 224), Image.LANCZOS)

            # Preprocess for CLIP
            cropped_img_processed = self.clip_preprocess(cropped_img)
            images_crops.append(cropped_img_processed)

        if len(images_crops) == 0:
            return torch.zeros(512, device=self.device)

        # Batch encode all crops
        image_input = torch.stack(images_crops).to(self.device)

        image_features = self.clip_model.encode_image(image_input).float()
        # L2 normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Average across all crops
        mask_features = image_features.mean(dim=0)

        return mask_features

    @torch.no_grad()
    def extract_pixel_features(
        self,
        image: np.ndarray,
        masks: Optional[List[Dict]] = None,
        max_size: int = 2048,
    ) -> torch.Tensor:
        """
        Extract dense pixel-aligned features from CLIP.

        For OpenMask3D, this extracts features at patch resolution and
        interpolates to full resolution for compatibility with the pipeline.

        Args:
            image: RGB image (H, W, 3)
            masks: Optional list of SAM masks (ignored, uses patch-based features)
            max_size: Maximum dimension for output features (to avoid memory issues)

        Returns:
            pixel_features: Dense features (H, W, C)
        """
        self._load_clip()

        h, w = image.shape[:2]

        # Resize image if too large to avoid interpolation issues
        if max(h, w) > max_size:
            from PIL import Image as PILImage
            scale = max_size / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            img_pil = PILImage.fromarray(image)
            img_resized = img_pil.resize((new_w, new_h), PILImage.LANCZOS)
            image = np.array(img_resized)
            h, w = new_h, new_w

        pil_image = Image.fromarray(image).convert("RGB")

        # CLIP processes at 224x224
        inputs = self.clip_preprocess(pil_image).to(self.device)

        # Get patch features
        patch_features = self.clip_model.encode_image(inputs)

        # Get feature dimension
        C = patch_features.shape[-1]

        # CLIP ViT-B/32 produces 7x7 patches for 224x224 input
        patch_h, patch_w = 7, 7

        # Reshape to spatial and upsample
        patch_features = patch_features.reshape(1, patch_h, patch_w, C)
        patch_features = patch_features.permute(0, 3, 1, 2)  # (1, C, H, W)

        # Upsample to original resolution
        pixel_features = F.interpolate(
            patch_features,
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
        pixel_features = pixel_features.permute(0, 2, 3, 1).squeeze(0)  # (H, W, C)

        # L2 normalize
        pixel_features = F.normalize(pixel_features, p=2, dim=-1)

        return pixel_features.cpu()

    def encode_text_query(self, text: str) -> torch.Tensor:
        """Encode text query using CLIP.

        Args:
            text: Query text string

        Returns:
            text_features: Normalized text features (1, C)
        """
        self._load_clip()

        import clip
        text_tokens = clip.tokenize(text).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)

        # L2 normalize
        text_features = F.normalize(text_features, p=2, dim=-1)

        return text_features


if __name__ == "__main__":
    # Test OpenMask3D
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    image = np.array(Image.open(args.image))
    print(f"Loaded image: {image.shape}")

    extractor = OpenMask3DExtractor()

    # Test pixel features
    features = extractor.extract_pixel_features(image)
    print(f"Feature shape: {features.shape}")

    # Test text encoding
    text_feat = extractor.encode_text_query("chair")
    print(f"Text feature shape: {text_feat.shape}")

    # Test mask features with dummy mask
    h, w = image.shape[:2]
    dummy_mask = np.zeros((h, w), dtype=bool)
    dummy_mask[h//4:3*h//4, w//4:3*w//4] = True

    mask_feat = extractor.extract_mask_features(image, dummy_mask)
    print(f"Mask feature shape: {mask_feat.shape}")
