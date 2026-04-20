"""
PLAF: Pixel-level Language-aligned Features.

This package implements the PLAF framework for extracting and storing
language-aligned features at pixel level using RADIO v2.5 and SAM.

Main components:
    - core: Feature extraction and fusion modules
    - storage: 3D feature pool storage
    - utils: Configuration and model loading utilities
    - baselines: ConceptFusion and OpenMask3D baseline methods

Reference: "PLAF: Pixel-wise Language-Aligned Feature Extraction for Efficient 3D Scene Understanding"
"""

__version__ = "0.1.0"

from plaf.core import (
    # Types
    MaskAnnotation,
    MaskFeature,
    PixelFeatures,
    ImageAnnotation,
    SceneAnnotation,
    TextQuery,
    SimilarityResult,
    # Core modules
    RadioFeatureExtractor,
    SamMaskGenerator,
    FeatureFusion,
    generate_masks_for_batch,
)
from plaf.storage.feature_pool_3d import FeaturePool3D, PointObservation
from plaf.utils import (
    # Configuration
    RadioConfig,
    SamConfig,
    FusionConfig,
    ProcessingConfig,
    PlafConfig,
    create_default_config,
    load_config_or_default,
    # Model loading
    ensure_device,
    resolve_sam_checkpoint,
    load_radio_model,
    load_sam_model,
    load_sam_mask_generator,
    ModelCache,
    get_global_cache,
)
from plaf.baselines import (
    ConceptFusionExtractor,
    OpenMask3DExtractor,
)

__all__ = [
    # Version info
    "__version__",
    # Types
    "MaskAnnotation",
    "MaskFeature",
    "PixelFeatures",
    "ImageAnnotation",
    "SceneAnnotation",
    "TextQuery",
    "SimilarityResult",
    # Core modules
    "RadioFeatureExtractor",
    "SamMaskGenerator",
    "FeatureFusion",
    "generate_masks_for_batch",
    # Storage
    "FeaturePool3D",
    "PointObservation",
    # Baselines
    "ConceptFusionExtractor",
    "OpenMask3DExtractor",
    # Configuration
    "RadioConfig",
    "SamConfig",
    "FusionConfig",
    "ProcessingConfig",
    "PlafConfig",
    "create_default_config",
    "load_config_or_default",
    # Model loading
    "ensure_device",
    "resolve_sam_checkpoint",
    "load_radio_model",
    "load_sam_model",
    "load_sam_mask_generator",
    "ModelCache",
    "get_global_cache",
]
