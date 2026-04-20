"""
Utility modules for PLAF.

Components:
    - config: Configuration management
    - model_loader: Model loading utilities
"""

from plaf.utils.config import (
    RadioConfig,
    SamConfig,
    FusionConfig,
    ProcessingConfig,
    PlafConfig,
    create_default_config,
    load_config_or_default,
)
from plaf.utils.model_loader import (
    ensure_device,
    resolve_sam_checkpoint,
    load_radio_model,
    load_sam_model,
    load_sam_mask_generator,
    ModelCache,
    get_global_cache,
)

__all__ = [
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

# Alias for backward compatibility
PLAFConfig = PlafConfig
