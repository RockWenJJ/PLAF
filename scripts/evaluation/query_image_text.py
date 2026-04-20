#!/usr/bin/env python3
"""
Interactive text-image query using PLAF and baseline methods.

This script loads an image and allows interactive text queries to find
regions matching the text description.

Usage:
    # PLAF method (default)
    python scripts/evaluation/query_image_text.py --input_image ./image.jpg --method plaf

    # With pre-computed features
    python scripts/evaluation/query_image_text.py --input_image ./image.jpg --input_feature ./features/image --method plaf

    # ConceptFusion baseline
    python scripts/evaluation/query_image_text.py --input_image ./image.jpg --method concept_fusion

    # OpenMask3D baseline
    python scripts/evaluation/query_image_text.py --input_image ./image.jpg --method openmask3d
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plaf.core import RadioFeatureExtractor, SamMaskGenerator, FeatureFusion
from plaf.baselines.concept_fusion import ConceptFusionExtractor
from plaf.baselines.openmask3d import OpenMask3DExtractor


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image.

    Args:
        image: Original image (H, W, 3) BGR or RGB
        heatmap: Heatmap (H, W) with values in [0, 1]
        alpha: Blending factor
        colormap: OpenCV colormap

    Returns:
        Overlaid image (H, W, 3) RGB
    """
    image_rgb = image

    # Normalize heatmap to 0-255
    heatmap_norm = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8))
    heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0

    # Blend images
    overlaid = (alpha * heatmap_colored + (1 - alpha) * image_rgb.astype(np.float32) / 255.0)
    overlaid = np.clip(overlaid, 0, 1)

    return (overlaid * 255).astype(np.uint8)


def display_image(image: np.ndarray, title: str = "Image", size: Tuple[int, int] = (12, 8)):
    """Display image using matplotlib."""
    plt.figure(figsize=size)
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)


class PLAFQueryProcessor:
    """Text query processor for PLAF method."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        print("Initializing PLAF models...")
        self.radio_extractor = RadioFeatureExtractor(device=device)
        self.sam_generator = SamMaskGenerator(device=device)
        self.fusion = FeatureFusion(device=device)
        print("PLAF models loaded.")

    def extract_features(self, image: np.ndarray):
        """Extract mask-indexed features from image."""
        # Extract RADIO features
        radio_features = self.radio_extractor.extract_features(image)

        # Generate SAM masks
        masks = self.sam_generator.generate_masks(image)

        # Fuse features
        pixel_features = self.fusion.fuse_mask_features(
            radio_features.cpu(),
            masks
        )

        return pixel_features

    def process_query(
        self,
        image: np.ndarray,
        pixel_features,
        query: str,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[float]]:
        """
        Process a text query.

        Returns:
            Tuple of (heatmap, overlaid_image, top_k_mask_ids, top_k_scores)
        """
        # Encode text query using the RadioFeatureExtractor's encode_text method
        # This properly handles the T5Tokenizer used by RADIO's language adaptor
        text_features = self.radio_extractor.encode_text(query)  # (1, D) on CPU, already normalized

        # Get mask features
        mask_ids = pixel_features.mask_ids
        mask_features = pixel_features.mask_features

        # Move to device for computation
        mask_features = mask_features.to(self.radio_extractor.device).half()
        text_features = text_features.to(self.radio_extractor.device).half()

        # mask_ids values: 0 (background), 1, 2, ..., N (regions)
        # So num_regions is the max value in mask_ids
        num_regions = mask_ids.max().item() if torch.is_tensor(mask_ids) else mask_ids.max()

        # Compute similarity using the same method as SamRadio
        # text_feat shape: (1, D) -> (D,)
        text_feat = text_features[0]  # (D,)

        # mask_features are already L2 normalized from FeatureFusion
        # mask_features shape: (N+1, D) where:
        #   - index 0 is background (zeros)
        #   - index 1 corresponds to mask_id 1
        #   - index N corresponds to mask_id N
        # We want to compare with text features

        # Compute similarity for all (N+1) masks including background
        # (N+1, D) @ (D,) -> (N+1,)
        similarities = (mask_features * text_feat.unsqueeze(0)).sum(dim=-1).cpu().numpy()

        # Keep raw similarities for ranking
        raw_similarities = similarities.copy()

        # Debug output - exclude background from statistics
        valid_similarities = raw_similarities[1:]  # Skip background (index 0)
        if len(valid_similarities) > 0:
            print(f"  Raw cosine similarities - min: {valid_similarities.min():.6f}, max: {valid_similarities.max():.6f}, mean: {valid_similarities.mean():.6f}")

        # Normalize similarities to [0, 1] for visualization only
        # Use only non-background similarities for normalization range
        if len(valid_similarities) > 0:
            min_val, max_val = valid_similarities.min(), valid_similarities.max()
            if max_val > min_val:
                similarities = (similarities - min_val) / (max_val - min_val)
            else:
                similarities = np.zeros_like(similarities)

        # Create pixel-wise similarity heatmap
        h, w = mask_ids.shape
        heatmap = np.zeros((h, w), dtype=np.float32)

        # Note: mask_ids uses 0 for background, 1-N for regions
        # similarities[0] is for background, similarities[i] is for mask_id=i
        for mask_id in range(1, num_regions + 1):
            if mask_id < len(similarities):
                heatmap[mask_ids == mask_id] = similarities[mask_id]

        # Get top-k masks using RAW similarities for correct ranking (skip background)
        if num_regions > 0:
            valid_similarities = raw_similarities[1:]  # Skip background (index 0)
            top_k_local = min(top_k, len(valid_similarities))
            top_k_indices = np.argsort(valid_similarities)[-top_k_local:][::-1] + 1  # +1 to convert from similarities index to mask_id
            top_k_scores = raw_similarities[top_k_indices].tolist()
        else:
            top_k_indices = []
            top_k_scores = []

        # Create heatmap overlay
        overlaid = overlay_heatmap(image, heatmap)

        return heatmap, overlaid, top_k_indices, top_k_scores


class ConceptFusionQueryProcessor:
    """Text query processor for ConceptFusion baseline."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        print("Initializing ConceptFusion model...")
        self.extractor = ConceptFusionExtractor(device=device)
        print("ConceptFusion model loaded.")

    def extract_features(self, image: np.ndarray):
        """Extract mask-indexed features from image."""
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        from plaf.utils.model_loader import resolve_sam_checkpoint

        # Generate SAM masks using SamAutomaticMaskGenerator
        sam_checkpoint = resolve_sam_checkpoint("vit_h", checkpoint_dir="./checkpoints")
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(self.device)
        sam.eval()

        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.85,
            stability_score_thresh=0.92,
            min_mask_region_area=200,
        )
        masks = mask_generator.generate(image)

        mask_list = []
        for m in masks:
            mask_list.append({
                "segmentation": m["segmentation"].astype(bool),
                "area": int(m["segmentation"].sum()),
                "bbox": m["bbox"]
            })

        # Extract features with ConceptFusion
        features = self.extractor.extract_pixel_features(image, mask_list)

        # Create mask_ids array
        h, w = image.shape[:2]
        mask_ids = np.zeros((h, w), dtype=np.int32)

        mask_features_list = []
        for i, mask_dict in enumerate(mask_list):
            mask = mask_dict["segmentation"]
            mask_ids[mask] = i + 1
            mask_feat = features[mask].mean(dim=0)
            mask_features_list.append(mask_feat)

        # Add background
        mask_union = np.zeros((h, w), dtype=bool)
        for mask_dict in mask_list:
            mask_union = mask_union | mask_dict["segmentation"]
        if (~mask_union).sum() > 0:
            background_feat = features[~mask_union].mean(dim=0)
            mask_features_list.insert(0, background_feat)

        mask_features_tensor = torch.stack(mask_features_list)

        return type('obj', (object,), {
            'mask_ids': mask_ids,
            'mask_features': mask_features_tensor
        })()

    def process_query(
        self,
        image: np.ndarray,
        pixel_features,
        query: str,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[float]]:
        """Process a text query."""
        # Encode text query
        text_feat = self.extractor.encode_text_query(query)

        # Get mask features
        mask_ids = pixel_features.mask_ids
        mask_features = pixel_features.mask_features

        # mask_features has shape (N+1, C) where index 0 is background
        num_regions = mask_ids.max().item() if torch.is_tensor(mask_ids) else mask_ids.max()

        # Normalize and compute similarity
        # text_feat shape is (1, C), squeeze to (C,) for dot product
        mask_feats_norm = F.normalize(mask_features.float(), dim=-1)
        text_feat_norm = F.normalize(text_feat.squeeze(0).float().cpu(), dim=-1)  # (C,)
        similarities = (mask_feats_norm @ text_feat_norm).detach().cpu().numpy()  # (N+1,)

        # Save raw similarities for ranking
        raw_similarities = similarities.copy()

        # Debug output - exclude background from statistics
        valid_similarities = similarities[1:]  # Skip background (index 0)
        if len(valid_similarities) > 0:
            print(f"  Raw cosine similarities - min: {valid_similarities.min():.6f}, max: {valid_similarities.max():.6f}, mean: {valid_similarities.mean():.6f}")

        # Normalize similarities to [0, 1] for visualization only
        if len(valid_similarities) > 0:
            min_val, max_val = valid_similarities.min(), valid_similarities.max()
            if max_val > min_val:
                similarities = (similarities - min_val) / (max_val - min_val)
            else:
                similarities = np.zeros_like(similarities)

        # Create pixel-wise similarity heatmap
        h, w = mask_ids.shape
        heatmap = np.zeros((h, w), dtype=np.float32)

        # similarities[0] is for background, similarities[i] is for mask_id=i
        for mask_id in range(1, num_regions + 1):
            if mask_id < len(similarities):
                heatmap[mask_ids == mask_id] = similarities[mask_id]

        # Get top-k masks using RAW similarities for correct ranking (skip background)
        if num_regions > 0:
            valid_similarities = raw_similarities[1:]  # Skip background (index 0)
            top_k_local = min(top_k, len(valid_similarities))
            top_k_indices = np.argsort(valid_similarities)[-top_k_local:][::-1] + 1
            top_k_scores = raw_similarities[top_k_indices].tolist()
        else:
            top_k_indices = []
            top_k_scores = []

        # Create heatmap overlay
        overlaid = overlay_heatmap(image, heatmap)

        return heatmap, overlaid, top_k_indices, top_k_scores


class OpenMask3DQueryProcessor:
    """Text query processor for OpenMask3D baseline."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        print("Initializing OpenMask3D model...")
        self.extractor = OpenMask3DExtractor(device=device)
        print("OpenMask3D model loaded.")

    def extract_features(self, image: np.ndarray):
        """Extract mask-indexed features from image."""
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        from plaf.utils.model_loader import resolve_sam_checkpoint

        # Generate SAM masks using SamAutomaticMaskGenerator
        sam_checkpoint = resolve_sam_checkpoint("vit_h", checkpoint_dir="./checkpoints")
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(self.device)
        sam.eval()

        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=16,
            pred_iou_thresh=0.85,
            stability_score_thresh=0.92,
            min_mask_region_area=200,
        )
        masks = mask_generator.generate(image)

        mask_list = []
        for m in masks:
            mask_list.append({
                "segmentation": m["segmentation"].astype(bool),
                "area": int(m["segmentation"].sum()),
                "bbox": m["bbox"]
            })

        # Extract mask-level features using OpenMask3D's multi-level cropping
        # This follows the original OpenMask3D approach: SAM refinement + multi-level CLIP crops
        h, w = image.shape[:2]
        mask_ids = np.zeros((h, w), dtype=np.int32)

        mask_features_list = []
        print(f"Extracting OpenMask3D features for {len(mask_list)} masks...")

        for i, mask_dict in enumerate(mask_list):
            mask = mask_dict["segmentation"]
            mask_ids[mask] = i + 1

            # Extract features using OpenMask3D's multi-level cropping approach
            # This includes SAM refinement and multi-scale CLIP encoding
            mask_feat = self.extractor.extract_mask_features(
                image, mask, use_sam_refinement=False
            )

            # Handle case where SAM is already used in mask generation
            # (we skip refinement to avoid redundant SAM calls)
            mask_features_list.append(mask_feat.cpu())

        # Add background feature (use mean of CLIP features from a dummy center crop)
        # For background, use a small center crop of the image
        center_h, center_w = h // 2, w // 2
        crop_size = min(224, min(h, w) // 4)
        center_crop = image[
            max(0, center_h - crop_size):min(h, center_h + crop_size),
            max(0, center_w - crop_size):min(w, center_w + crop_size)
        ]

        # Get background feature from center crop
        from PIL import Image as PILImage
        from plaf.baselines.openmask3d import OpenMask3DExtractor

        pil_crop = PILImage.fromarray(center_crop)
        if pil_crop.size[0] < 224 or pil_crop.size[1] < 224:
            pil_crop = pil_crop.resize((224, 224), PILImage.LANCZOS)

        # Use CLIP directly for background
        import clip
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=self.device)
        with torch.no_grad():
            crop_input = clip_preprocess(pil_crop).unsqueeze(0).to(self.device)
            bg_feat = clip_model.encode_image(crop_input).float()
            bg_feat = bg_feat / bg_feat.norm(dim=-1, keepdim=True)

        mask_features_list.insert(0, bg_feat.squeeze(0).cpu())

        mask_features_tensor = torch.stack(mask_features_list)

        return type('obj', (object,), {
            'mask_ids': mask_ids,
            'mask_features': mask_features_tensor
        })()

    def process_query(
        self,
        image: np.ndarray,
        pixel_features,
        query: str,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, List[int], List[float]]:
        """Process a text query."""
        # Encode text query
        text_feat = self.extractor.encode_text_query(query)

        # Get mask features
        mask_ids = pixel_features.mask_ids
        mask_features = pixel_features.mask_features

        # mask_features has shape (N+1, C) where index 0 is background
        num_regions = mask_ids.max().item() if torch.is_tensor(mask_ids) else mask_ids.max()

        # Normalize and compute similarity
        # text_feat shape is (1, C), squeeze to (C,) for dot product
        mask_feats_norm = F.normalize(mask_features.float(), dim=-1)
        text_feat_norm = F.normalize(text_feat.squeeze(0).float().cpu(), dim=-1)  # (C,)
        similarities = (mask_feats_norm @ text_feat_norm).detach().cpu().numpy()  # (N+1,)

        # Save raw similarities for ranking
        raw_similarities = similarities.copy()

        # Debug output - exclude background from statistics
        valid_similarities = similarities[1:]  # Skip background (index 0)
        if len(valid_similarities) > 0:
            print(f"  Raw cosine similarities - min: {valid_similarities.min():.6f}, max: {valid_similarities.max():.6f}, mean: {valid_similarities.mean():.6f}")

        # Normalize similarities to [0, 1] for visualization only
        if len(valid_similarities) > 0:
            min_val, max_val = valid_similarities.min(), valid_similarities.max()
            if max_val > min_val:
                similarities = (similarities - min_val) / (max_val - min_val)
            else:
                similarities = np.zeros_like(similarities)

        # Create pixel-wise similarity heatmap
        h, w = mask_ids.shape
        heatmap = np.zeros((h, w), dtype=np.float32)

        # similarities[0] is for background, similarities[i] is for mask_id=i
        for mask_id in range(1, num_regions + 1):
            if mask_id < len(similarities):
                heatmap[mask_ids == mask_id] = similarities[mask_id]

        # Get top-k masks using RAW similarities for correct ranking (skip background)
        if num_regions > 0:
            valid_similarities = raw_similarities[1:]  # Skip background (index 0)
            top_k_local = min(top_k, len(valid_similarities))
            top_k_indices = np.argsort(valid_similarities)[-top_k_local:][::-1] + 1
            top_k_scores = raw_similarities[top_k_indices].tolist()
        else:
            top_k_indices = []
            top_k_scores = []

        # Create heatmap overlay
        overlaid = overlay_heatmap(image, heatmap)

        return heatmap, overlaid, top_k_indices, top_k_scores


def save_features(image_path: str, pixel_features, output_base: str):
    """Save extracted features for faster subsequent queries."""
    import pickle

    output_path = Path(output_base)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save mask_ids and mask_features
    data = {
        'mask_ids': pixel_features.mask_ids,
        'mask_features': pixel_features.mask_features.cpu() if torch.is_tensor(pixel_features.mask_features) else pixel_features.mask_features
    }

    with open(f'{output_base}.pkl', 'wb') as f:
        pickle.dump(data, f)

    print(f"Features saved to {output_base}.pkl")


def load_features(input_feature: str):
    """Load pre-extracted features."""
    import pickle

    with open(f'{input_feature}.pkl', 'rb') as f:
        data = pickle.load(f)

    return type('obj', (object,), {
        'mask_ids': data['mask_ids'],
        'mask_features': data['mask_features']
    })()


def main():
    # Disable HF_HUB_OFFLINE mode if set
    if 'HF_HUB_OFFLINE' in os.environ:
        del os.environ['HF_HUB_OFFLINE']

    parser = argparse.ArgumentParser(
        description="Interactive text-image query using PLAF and baseline methods"
    )

    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--input_feature",
        type=str,
        default=None,
        help="Path to pre-extracted features (without .pkl extension)"
    )
    parser.add_argument(
        "--save_feature",
        type=str,
        default=None,
        help="Save extracted features to this path (for faster subsequent queries)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="plaf",
        choices=["plaf", "concept_fusion", "openmask3d"],
        help="Feature extraction method"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu, default: cuda)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top matching regions to display"
    )
    parser.add_argument(
        "--heatmap_alpha",
        type=float,
        default=0.5,
        help="Blending factor for heatmap overlay (0.0-1.0, default: 0.5)"
    )
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        default=None,
        help="Pre-defined queries to process (non-interactive mode)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for saving result images"
    )

    args = parser.parse_args()

    # Validate image path
    if not os.path.exists(args.input_image):
        raise ValueError(f"Input image does not exist: {args.input_image}")

    # Load image
    print(f"Loading image: {args.input_image}")
    original_image = np.array(Image.open(args.input_image))
    if original_image.ndim == 2:
        original_image = np.stack([original_image] * 3, axis=-1)
    elif original_image.shape[2] == 4:
        original_image = original_image[:, :, :3]

    # Initialize processor
    if args.method == "plaf":
        processor = PLAFQueryProcessor(device=args.device)
    elif args.method == "concept_fusion":
        processor = ConceptFusionQueryProcessor(device=args.device)
    elif args.method == "openmask3d":
        processor = OpenMask3DQueryProcessor(device=args.device)

    # Extract or load features
    if args.input_feature is not None:
        print(f"Loading pre-extracted features from {args.input_feature}.pkl...")
        pixel_features = load_features(args.input_feature)
    else:
        print("Extracting features from image...")
        pixel_features = processor.extract_features(original_image)
        num_masks = pixel_features.mask_ids.max().item() + 1 if torch.is_tensor(pixel_features.mask_ids) else pixel_features.mask_ids.max() + 1
        print(f"Extracted {num_masks} masks")

        # Save features if requested
        if args.save_feature is not None:
            save_features(args.input_image, pixel_features, args.save_feature)

    # Process queries
    if args.queries is not None:
        # Non-interactive mode: process pre-defined queries
        print(f"\nProcessing {len(args.queries)} pre-defined queries...")
        print("=" * 60)

        output_dir = None
        if args.output_dir is not None:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving results to {output_dir}")

        for i, query in enumerate(args.queries):
            print(f"\n[{i+1}/{len(args.queries)}] Processing query: '{query}'...")

            # Process query
            heatmap, overlaid, top_k_ids, top_k_scores = processor.process_query(
                original_image,
                pixel_features,
                query,
                top_k=args.top_k
            )

            # Save result if output directory specified
            if output_dir is not None:
                # Save heatmap overlay
                safe_name = query.replace(" ", "_").replace("/", "_")[:50]
                result_path = output_dir / f"query_{i:02d}_{safe_name}.jpg"
                Image.fromarray(overlaid).save(result_path)
                print(f"  Saved to {result_path}")

            # Print results
            print(f"  Similarity range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
            print(f"  Top {len(top_k_ids)} matching regions:")
            for j, (mask_id, score) in enumerate(zip(top_k_ids, top_k_scores)):
                print(f"    {j+1}. Mask {mask_id}: {score:.3f}")

        print("\n" + "=" * 60)
        print("Done!")

    else:
        # Interactive mode
        # Display original image
        print("\nDisplaying original image...")
        display_image(original_image, "Original Image")

        # Interactive query loop
        print("\n" + "=" * 60)
        print(f"Interactive text query mode ({args.method})")
        print("Enter text queries to find matching regions in the image")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60 + "\n")

        while True:
            try:
                # Get text query from user
                query = input("Enter text query: ").strip()

                if query.lower() in ['quit', 'exit', 'q']:
                    print("Exiting...")
                    break

                if not query:
                    print("Please enter a non-empty query.")
                    continue

                print(f"Processing query: '{query}'...")

                # Process query
                heatmap, overlaid, top_k_ids, top_k_scores = processor.process_query(
                    original_image,
                    pixel_features,
                    query,
                    top_k=args.top_k
                )

                # Display result
                display_image(overlaid, f"Query: '{query}'")

                print(f"Query processed. Similarity range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")
                print(f"Top {len(top_k_ids)} matching regions:")
                for i, (mask_id, score) in enumerate(zip(top_k_ids, top_k_scores)):
                    print(f"  {i+1}. Mask {mask_id}: {score:.3f}")
                print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error processing query: {e}")
                import traceback
                traceback.print_exc()
                continue

    plt.close('all')


if __name__ == "__main__":
    main()
