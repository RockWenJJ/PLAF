"""
ConceptFusion baseline implementation.

Based on: "ConceptFusion: Open-set Multimodal 3D Mapping"
Paper: https://arxiv.org/abs/2304.01572
GitHub: https://github.com/concept-fusion/concept-fusion

Key idea: Combine global CLIP features with local CLIP features
using similarity-based weighting.
"""

from typing import List, Dict, Optional
import logging

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConceptFusionExtractor:
    """
    ConceptFusion feature extractor.

    Implements the pixel-aligned feature extraction from ConceptFusion:
    1. Global CLIP features: One vector per image
    2. Local CLIP features: One vector per mask region
    3. Fusion weight: w_i = (sim_global + avg_sim_local_i) / sum

    Attributes:
        global_dim: Global feature dimension
        local_dim: Local feature dimension
    """

    def __init__(
        self,
        device: str = "cuda",
        mask_model_type: str = "vit_h"
    ):
        """
        Initialize ConceptFusion extractor.

        Args:
            device: torch device
            mask_model_type: SAM model type for region proposals
        """
        self.device = torch.device(device)
        self.mask_model_type = mask_model_type

        # Initialize models (lazy loading)
        self.clip_model = None
        self.sam_model = None

        logger.info("ConceptFusion extractor initialized (models loaded on first use)")

    def _load_models(self):
        """Load CLIP and SAM models."""
        if self.clip_model is None:
            try:
                import clip
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                self.clip_model.eval()
                logger.info("CLIP model loaded")
            except ImportError:
                raise RuntimeError(
                            "CLIP not installed. Install with: pip install openai-clip"
                        )

        if self.sam_model is None:
            try:
                from segment_anything import sam_model_registry, SamPredictor
                sam = sam_model_registry[self.mask_model_type](None)
                sam.to(self.device)
                sam.eval()
                self.sam_model = SamPredictor(sam)
                logger.info(f"SAM {self.mask_model_type} loaded")
            except ImportError:
                raise RuntimeError(
                            "SAM not installed. Install with: pip install segment-anything"
                        )

    @torch.no_grad()
    def extract_global_features(
        self,
        image: np.ndarray
    ) -> torch.Tensor:
        """
        Extract global (image-level) CLIP features.

        Args:
            image: RGB image (H, W, 3)

        Returns:
            global_features: Global feature vector (C,)
        """
        self._load_models()

        # Preprocess for CLIP
        from PIL import Image
        pil_image = Image.fromarray(image).convert("RGB")

        # CLIP preprocess returns a tensor that needs to be unsqueezed to add batch dimension
        image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        image_features = self.clip_model.encode_image(image_input)

        # L2 normalize
        return F.normalize(image_features, p=2, dim=-1).squeeze(0)

    @torch.no_grad()
    def extract_local_features(
        self,
        image: np.ndarray,
        masks: List[Dict]
    ) -> torch.Tensor:
        """
        Extract local (region-level) CLIP features.

        Args:
            image: RGB image (H, W, 3)
            masks: List of SAM mask annotations

        Returns:
            local_features: Local feature vectors (K, C)
        """
        self._load_models()

        from PIL import Image

        local_features_list = []

        for mask_dict in masks:
            mask = mask_dict["segmentation"]
            bbox = mask_dict["bbox"]

            # Crop to bounding box
            x1, y1, x2, y2 = bbox
            cropped = image[y1:y2, x1:x2]

            # Encode cropped region
            pil_image = Image.fromarray(cropped).convert("RGB")

            # Get minimal size to avoid CLIP issues
            if pil_image.size[0] < 224 or pil_image.size[1] < 224:
                pil_image = pil_image.resize((224, 224), Image.LANCZOS)

            # CLIP preprocess returns a tensor that needs to be unsqueezed to add batch dimension
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            features = self.clip_model.encode_image(image_input)

            # L2 normalize
            features = F.normalize(features, p=2, dim=-1)
            local_features_list.append(features.squeeze(0))

        return torch.stack(local_features_list, dim=0)

    @torch.no_grad()
    def extract_pixel_features(
        self,
        image: np.ndarray,
        masks: List[Dict]
    ) -> torch.Tensor:
        """
        Extract pixel-aligned features using ConceptFusion fusion.

        Matches the original ConceptFusion implementation:
        1. Compute cosine similarity between global and each local feature
        2. Apply softmax to get per-mask weights
        3. For each mask: weighted_feat = w_i * global + (1 - w_i) * local
        4. Additively accumulate and normalize after each mask

        Args:
            image: RGB image (H, W, 3)
            masks: List of SAM mask annotations

        Returns:
            pixel_features: Dense pixel features (H, W, C) on CPU, float32
        """
        h, w = image.shape[:2]

        global_feature = self.extract_global_features(image)  # (C,) on device
        local_features = self.extract_local_features(image, masks)  # (K, C) on device

        # Ensure both are on the same device
        global_feature = global_feature.to(self.device)
        local_features = local_features.to(self.device)

        # Cosine similarity between global feature and each local feature
        similarity_scores = F.cosine_similarity(
            global_feature.unsqueeze(0),  # (1, C)
            local_features,               # (K, C)
            dim=-1,
        )  # (K,)

        softmax_scores = F.softmax(similarity_scores, dim=0)  # (K,)

        feat_dim = global_feature.shape[0]
        pixel_features = torch.zeros((h, w, feat_dim), dtype=torch.float32)

        for i, mask_dict in enumerate(masks):
            mask = mask_dict["segmentation"]
            if mask.sum() == 0:
                continue

            nonzero_inds = torch.argwhere(torch.from_numpy(mask))  # (N, 2): row, col
            if nonzero_inds.shape[0] == 0:
                continue

            w_i = softmax_scores[i]
            weighted_feat = w_i * global_feature + (1 - w_i) * local_features[i]
            weighted_feat = F.normalize(weighted_feat.unsqueeze(0), dim=-1).squeeze(0)

            rows, cols = nonzero_inds[:, 0], nonzero_inds[:, 1]
            pixel_features[rows, cols] += weighted_feat.cpu()

            # Normalize to avoid accumulation issues
            current_feats = pixel_features[rows, cols]
            norm = torch.norm(current_feats, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            pixel_features[rows, cols] = current_feats / norm

        # Handle any remaining NaN or Inf values
        pixel_features = torch.nan_to_num(pixel_features, nan=0.0, posinf=0.0, neginf=0.0)

        return pixel_features

    def encode_text_query(self, text: str) -> torch.Tensor:
        """Encode text query using CLIP."""
        self._load_models()

        import clip
        text_tokens = clip.tokenize(text).to(self.device)
        text_features = self.clip_model.encode_text(text_tokens)

        return F.normalize(text_features, p=2, dim=-1)


if __name__ == "__main__":
    # Test ConceptFusion
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()

    # Load image
    image = np.array(Image.open(args.image))
    print(f"Loaded image: {image.shape}")

    # Create dummy masks
    h, w = image.shape[:2]
    masks = [{
        "segmentation": np.zeros((h, w), dtype=bool),
        "area": h * w // 4,
        "bbox": (w//4, h//4, w//2, h//2),  # XYWH format
    }]
    masks[0]["segmentation"][h//4:3*h//4, w//4:3*w//4] = True

    # Extract features
    extractor = ConceptFusionExtractor()
    features = extractor.extract_pixel_features(image, masks)

    print(f"Feature shape: {features.shape}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
