"""
Core modules for PLAF feature extraction and fusion.

Components:
    - types: Data type definitions
    - radio_feature_extractor: RADIO v2.5 feature extractor
    - sam_mask_generator: SAM mask generator
    - feature_fusion: Mask-guided feature aggregation
"""

from plaf.core.types import (
    MaskAnnotation,
    MaskFeature,
    PixelFeatures,
    ImageAnnotation,
    SceneAnnotation,
    TextQuery,
    SimilarityResult,
)
from plaf.core.radio_feature_extractor import RadioFeatureExtractor
from plaf.core.sam_mask_generator import SamMaskGenerator, generate_masks_for_batch

# Try to import feature_fusion, handle if it has issues
try:
    from plaf.core.feature_fusion import FeatureFusion
except ImportError:
    FeatureFusion = None

__all__ = [
    # Types
    "MaskAnnotation",
    "MaskFeature",
    "PixelFeatures",
    "ImageAnnotation",
    "SceneAnnotation",
    "TextQuery",
    "SimilarityResult",
    # Feature extraction
    "RadioFeatureExtractor",
    "SamMaskGenerator",
    "generate_masks_for_batch",
    "FeatureFusion",
]
