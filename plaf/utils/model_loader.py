"""
Model loading utilities for PLAF (Pixel-level Language-aligned Features).

This module provides utilities for loading RADIO and SAM models, including
checkpoint downloading and device management.
"""

import os
from typing import Optional, Tuple
from pathlib import Path

import torch
import torch.hub


def ensure_device(device_str: str) -> torch.device:
    """Ensure device string is valid and available.

    Args:
        device_str: Device string ("cuda", "cpu", "mps")

    Returns:
        torch.device object

    Examples:
        >>> ensure_device("cuda")
        device(type='cuda')
        >>> ensure_device("cpu")
        device(type='cpu')
    """
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def resolve_sam_checkpoint(
    model_type: str,
    checkpoint_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None
) -> str:
    """Resolve SAM checkpoint path, downloading if necessary.

    Args:
        model_type: SAM model type ("vit_h", "vit_l", "vit_b")
        checkpoint_path: Explicit checkpoint path (overrides auto-resolution)
        checkpoint_dir: Directory to check/store checkpoints

    Returns:
        Path to SAM checkpoint

    Raises:
        ValueError: If model_type is invalid
        FileNotFoundError: If checkpoint cannot be found and download fails

    Examples:
        >>> resolve_sam_checkpoint("vit_h", checkpoint_dir="./checkpoints")
        './checkpoints/sam_vit_h_4b8939.pth'
    """
    checkpoint_names = {
        "vit_h": "sam_vit_h_4b8939.pth",
        "vit_l": "sam_vit_l_0b3195.pth",
        "vit_b": "sam_vit_b_01ec64.pth",
    }

    download_urls = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }

    if model_type not in checkpoint_names:
        raise ValueError(
            f"Invalid model_type: {model_type}. "
            f"Must be one of {list(checkpoint_names.keys())}"
        )

    if checkpoint_path is not None:
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        raise FileNotFoundError(f"Specified checkpoint not found: {checkpoint_path}")

    if checkpoint_dir is None:
        checkpoint_dir = "./checkpoints"

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, checkpoint_names[model_type])

    if not os.path.exists(ckpt_path):
        print(f"SAM checkpoint not found at {ckpt_path}. Downloading...")
        url = download_urls[model_type]
        try:
            torch.hub.download_url_to_file(url, ckpt_path)
            print(f"Downloaded SAM checkpoint to {ckpt_path}")
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to download SAM checkpoint from {url}: {e}"
            )

    return ckpt_path


def load_radio_model(
    model_version: str = "radio_v2.5-b",
    lang_model: str = "siglip",
    device: Optional[torch.device] = None,
    skip_validation: bool = True,
) -> torch.nn.Module:
    """Load RADIO v2.5 model from torch hub.

    Args:
        model_version: RADIO model version
        lang_model: Language model adaptor name
        device: Device to load model on (None for auto-detect)
        skip_validation: Whether to skip model validation

    Returns:
        RADIO model

    Examples:
        >>> model = load_radio_model(device=torch.device("cuda"))
        >>> model.eval()
    """
    if device is None:
        device = ensure_device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading RADIO model: {model_version} with {lang_model} adaptor...")

    # Monkey patch for torch compatibility (weights_only parameter)
    import sys

    # Clear cached RADIO hubconf to force reload with patch
    modules_to_clear = [m for m in sys.modules if m.startswith('NVlabs_RADIO') or m == 'radio']
    for m in modules_to_clear:
        del sys.modules[m]

    # Save original functions
    original_torch_load = torch.load
    original_hub_load_state_dict = getattr(torch.hub, 'load_state_dict_from_url', None)

    # Patch torch.load to remove weights_only
    def patched_torch_load(*args, **kwargs):
        if 'weights_only' in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load

    # Patch torch.hub.load_state_dict_from_url to remove weights_only
    # This needs to be done before hubconf is imported
    if original_hub_load_state_dict is not None:
        import inspect
        sig = inspect.signature(original_hub_load_state_dict)

        # Check if weights_only parameter exists (PyTorch 2.0+)
        if 'weights_only' in sig.parameters:
            # Wrap to remove weights_only for older hubconf.py compatibility
            def patched_hub_load_from_url(*args, **kwargs):
                if 'weights_only' in kwargs:
                    kwargs['weights_only'] = False
                return original_hub_load_state_dict(*args, **kwargs)
            torch.hub.load_state_dict_from_url = patched_hub_load_from_url
    else:
        # PyTorch < 2.0: create a wrapper that accepts and ignores weights_only
        # This won't be called by old PyTorch, but prevents errors if hubconf tries to use it
        def dummy_load_from_url(*args, **kwargs):
            if 'weights_only' in kwargs:
                kwargs['weights_only'] = False
            # For old PyTorch, fall back to torch.load after downloading
            from torch.hub import download_url_to_file
            import tempfile
            import os
            if args and isinstance(args[0], str):
                url = args[0]
                model_dir = args[1] if len(args) > 1 else kwargs.get('model_dir', None)
                # Download to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp:
                    tmp_path = tmp.name
                download_url_to_file(url, tmp_path, progress=kwargs.get('progress', True))
                # Load with patched torch.load
                result = torch.load(tmp_path, map_location=kwargs.get('map_location', None))
                os.unlink(tmp_path)
                return result
        torch.hub.load_state_dict_from_url = dummy_load_from_url

    try:
        # Use local cache path directly to avoid SSL errors
        radio_cache_path = os.path.expanduser("~/.cache/torch/hub/NVlabs_RADIO_main")
        radio_model = torch.hub.load(
            radio_cache_path,
            "radio_model",
            version=model_version,
            progress=True,
            skip_validation=skip_validation,
            trust_repo=True,
            source="local",
            adaptor_names=[lang_model]
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load RADIO model: {e}")
    finally:
        # Restore original functions
        torch.load = original_torch_load
        if original_hub_load_state_dict is not None:
            torch.hub.load_state_dict_from_url = original_hub_load_state_dict

    radio_model.to(device)
    radio_model.eval()

    print(f"RADIO model loaded on {device}")

    return radio_model


def load_sam_model(
    model_type: str = "vit_h",
    checkpoint_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, str]:
    """Load SAM model.

    Args:
        model_type: SAM model type ("vit_h", "vit_l", "vit_b")
        checkpoint_path: Explicit checkpoint path
        checkpoint_dir: Directory for checkpoints
        device: Device to load model on (None for auto-detect)

    Returns:
        Tuple of (SAM model, checkpoint path)

    Examples:
        >>> sam, ckpt = load_sam_model()
        >>> sam.eval()
    """
    if device is None:
        device = ensure_device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_sam_checkpoint(model_type, checkpoint_path, checkpoint_dir)

    print(f"Loading SAM model: {model_type} from {ckpt_path}...")

    try:
        from segment_anything import sam_model_registry
        sam = sam_model_registry[model_type](checkpoint=ckpt_path)
    except ImportError:
        raise ImportError(
            "segment_anything package not found. "
            "Install it with: pip install git+https://github.com/facebookresearch/segment-anything.git"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load SAM model: {e}")

    sam.to(device)
    sam.eval()

    print(f"SAM model loaded on {device}")

    return sam, ckpt_path


def load_sam_mask_generator(
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
    device: Optional[torch.device] = None,
):
    """Load SAM automatic mask generator.

    Args:
        model_type: SAM model type
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
        device: Device to load model on

    Returns:
        SamAutomaticMaskGenerator instance

    Examples:
        >>> mask_gen = load_sam_mask_generator(points_per_side=32)
        >>> masks = mask_gen.generate(image)
    """
    sam, _ = load_sam_model(model_type, checkpoint_path, checkpoint_dir, device)

    from segment_anything import SamAutomaticMaskGenerator

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area,
        stability_score_offset=stability_score_offset,
        stability_score_thresh=stability_score_thresh,
        box_nms_thresh=box_nms_thresh,
    )

    return mask_generator


class ModelCache:
    """Cache for loaded models to avoid redundant loading.

    This class maintains a cache of loaded models and provides methods
    to retrieve or create models as needed.

    Attributes:
        _cache: Dictionary mapping cache keys to loaded models

    Examples:
        >>> cache = ModelCache()
        >>> radio_model = cache.get_radio("radio_v2.5-b", "cuda")
        >>> sam_model = cache.get_sam("vit_h", "cuda")
    """

    def __init__(self):
        """Initialize empty model cache."""
        self._cache = {}

    def _make_key(self, *args) -> str:
        """Create cache key from arguments."""
        return "_".join(str(arg) for arg in args)

    def get_radio(
        self,
        model_version: str = "radio_v2.5-b",
        lang_model: str = "siglip",
        device: Optional[torch.device] = None,
    ) -> torch.nn.Module:
        """Get RADIO model from cache or load it."""
        key = self._make_key("radio", model_version, lang_model, device)

        if key not in self._cache:
            self._cache[key] = load_radio_model(model_version, lang_model, device)

        return self._cache[key]

    def get_sam(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        device: Optional[torch.device] = None,
    ) -> torch.nn.Module:
        """Get SAM model from cache or load it."""
        key = self._make_key("sam", model_type, checkpoint_path, device)

        if key not in self._cache:
            sam, _ = load_sam_model(model_type, checkpoint_path, checkpoint_dir, device)
            self._cache[key] = sam

        return self._cache[key]

    def get_sam_mask_generator(
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
        device: Optional[torch.device] = None,
    ):
        """Get SAM mask generator from cache or load it."""
        key = self._make_key(
            "sam_mask_gen", model_type, checkpoint_path, device,
            points_per_side, pred_iou_thresh
        )

        if key not in self._cache:
            self._cache[key] = load_sam_mask_generator(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                checkpoint_dir=checkpoint_dir,
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

        return self._cache[key]

    def clear(self):
        """Clear the cache."""
        self._cache.clear()

    def remove(self, key: str):
        """Remove specific entry from cache."""
        if key in self._cache:
            del self._cache[key]


# Global model cache instance
_global_cache = ModelCache()


def get_global_cache() -> ModelCache:
    """Get the global model cache instance."""
    return _global_cache
