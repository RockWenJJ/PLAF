"""
RADIO v2.5 feature extractor for PLAF.

This module implements dense pixel-level language-aligned feature extraction
using the RADIO v2.5 model with language adaptor.
"""

from typing import List, Optional, Tuple, Union
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from ..utils.model_loader import load_radio_model, ensure_device, get_global_cache
from ..utils.config import RadioConfig


class RadioFeatureExtractor:
    """RADIO v2.5 feature extractor for language-aligned pixel features.

    This class extracts dense pixel-level features that are aligned with language
    through the SIGLIP adaptor, enabling open-vocabulary querying.

    Attributes:
        model: RADIO model instance
        lang_adaptor: Language adaptor (e.g., SIGLIP)
        config: Configuration for feature extraction
        device: Device to run inference on

    Examples:
        >>> extractor = RadioFeatureExtractor(device="cuda")
        >>> features = extractor.extract_features(image)
        >>> print(features.shape)  # (H, W, 768)
    """

    def __init__(
        self,
        model_version: str = "radio_v2.5-b",
        lang_model: str = "siglip",
        input_resolution: int = 512,
        language_aligned: bool = True,
        device: str = "cuda",
        compile_model: bool = False,
        amp: bool = True,
        use_cache: bool = True,
    ):
        """Initialize RADIO feature extractor.

        Args:
            model_version: RADIO model version
            lang_model: Language model adaptor name
            input_resolution: Input resolution for RADIO encoder
            language_aligned: Whether to use language-aligned features
            device: Device to run on ("cuda" or "cpu")
            compile_model: Whether to compile model with torch.compile
            amp: Whether to use automatic mixed precision
            use_cache: Whether to use global model cache
        """
        self.config = RadioConfig(
            model_version=model_version,
            lang_model=lang_model,
            input_resolution=input_resolution,
            language_aligned=language_aligned,
            compile_model=compile_model,
            amp=amp,
            device=device,
        )

        self.device = ensure_device(device)
        self._amp = amp
        self._use_cache = use_cache

        # Load model
        self.model = self._load_model()

        # Get language adaptor
        self.lang_adaptor = self.model.adaptors[lang_model]
        self.lang_adaptor.eval()

        # Get patch size for feature extraction
        self.patch_size = self.model.patch_size

        # Compile model if requested
        if compile_model:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            warnings.warn("Model compiled with torch.compile. First call will be slow.")

    def _load_model(self) -> torch.nn.Module:
        """Load RADIO model."""
        if self._use_cache:
            cache = get_global_cache()
            model = cache.get_radio(
                model_version=self.config.model_version,
                lang_model=self.config.lang_model,
                device=self.device,
            )
        else:
            model = load_radio_model(
                model_version=self.config.model_version,
                lang_model=self.config.lang_model,
                device=self.device,
            )
        return model

    @torch.no_grad()
    def extract_features(
        self,
        image: np.ndarray,
        normalize: bool = True,
        max_size: int = 2048,
    ) -> torch.Tensor:
        """Extract dense pixel-level features from a single image.

        Args:
            image: RGB image as numpy array (H, W, 3) uint8
            normalize: Whether to L2 normalize features
            max_size: Maximum dimension for output features (to avoid memory issues)

        Returns:
            Features tensor of shape (H, W, C) float32 on CPU

        Raises:
            ValueError: If image is invalid
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be (H, W, 3), got shape {image.shape}")

        if image.dtype != np.uint8:
            warnings.warn(f"Image dtype is {image.dtype}, converting to uint8")
            image = image.astype(np.uint8)

        H, W = image.shape[:2]

        # Resize image if too large to avoid interpolation issues
        if max(H, W) > max_size:
            scale = max_size / max(H, W)
            new_h = int(H * scale)
            new_w = int(W * scale)
            # Use PIL for high-quality resize
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray(image)
            img_resized_pil = img_pil.resize((new_w, new_h), PILImage.LANCZOS)
            image = np.array(img_resized_pil)
            H, W = H, W = new_h, new_w

        # Convert to tensor and normalize to [0, 1]
        img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        img_t = img_t.unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # Resize to input resolution
        input_res = self.config.input_resolution
        img_resized = F.interpolate(
            img_t,
            size=(input_res, input_res),
            mode="bilinear",
            align_corners=False,
        )

        # Extract features
        with torch.cuda.amp.autocast(enabled=self._amp):
            outputs = self.model(img_resized)
            summary, patch_feat = outputs["backbone"].summary, outputs["backbone"].features

            # Apply language adaptor if configured
            if self.config.language_aligned:
                patch_feat = self.lang_adaptor.head_mlp(patch_feat)

        # Rearrange from (B, N, C) to (B, C, H, W)
        spatial_h = input_res // self.patch_size
        spatial_w = input_res // self.patch_size
        patch_feat = rearrange(
            patch_feat,
            "b (h w) c -> b c h w",
            h=spatial_h,
            w=spatial_w,
        )

        # Upsample to original resolution
        pixel_feat = F.interpolate(
            patch_feat,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (1, C, H, W)

        # Convert to (H, W, C) and move to CPU
        pixel_feat = pixel_feat[0].permute(1, 2, 0).contiguous()  # (H, W, C)

        if normalize:
            pixel_feat = F.normalize(pixel_feat, dim=-1)

        return pixel_feat.float().cpu()

    @torch.no_grad()
    def extract_batch(
        self,
        images: List[np.ndarray],
        normalize: bool = True,
    ) -> List[torch.Tensor]:
        """Extract features from a batch of images.

        Args:
            images: List of RGB images (H, W, 3) uint8
            normalize: Whether to L2 normalize features

        Returns:
            List of feature tensors, each (H, W, C) float32 on CPU

        Note:
            Images are processed individually to handle different sizes.
            For true batch processing, resize all images to the same size
            and use extract_features_batched instead.
        """
        return [self.extract_features(img, normalize) for img in images]

    @torch.no_grad()
    def extract_features_batched(
        self,
        images: np.ndarray,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Extract features from a batch of same-sized images.

        Args:
            images: Batch of RGB images (B, H, W, 3) uint8
            normalize: Whether to L2 normalize features

        Returns:
            Features tensor of shape (B, H, W, C) float32 on CPU

        Raises:
            ValueError: If images are not the same size
        """
        if images.ndim != 4:
            raise ValueError(f"Images must be (B, H, W, 3), got shape {images.shape}")

        B, H, W, C = images.shape

        # Convert to tensor and normalize
        img_t = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0  # (B, 3, H, W)
        img_t = img_t.to(self.device)

        # Resize to input resolution
        input_res = self.config.input_resolution
        img_resized = F.interpolate(
            img_t,
            size=(input_res, input_res),
            mode="bilinear",
            align_corners=False,
        )

        # Extract features
        with torch.cuda.amp.autocast(enabled=self._amp):
            outputs = self.model(img_resized)
            patch_feat = outputs["backbone"].features

            if self.config.language_aligned:
                patch_feat = self.lang_adaptor.head_mlp(patch_feat)

        # Rearrange and upsample
        spatial_h = input_res // self.patch_size
        spatial_w = input_res // self.patch_size
        patch_feat = rearrange(
            patch_feat,
            "b (h w) c -> b c h w",
            h=spatial_h,
            w=spatial_w,
        )

        pixel_feat = F.interpolate(
            patch_feat,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (B, C, H, W)

        pixel_feat = pixel_feat.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)

        if normalize:
            pixel_feat = F.normalize(pixel_feat, dim=-1)

        return pixel_feat.float().cpu()

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text query using the language adaptor.

        Args:
            text: Query text string

        Returns:
            Text features of shape (1, D) float32 on CPU, L2 normalized
        """
        with torch.no_grad():
            text_tokens = self.lang_adaptor.tokenizer([text]).to(self.device)
            text_features = self.lang_adaptor.encode_text(text_tokens)  # (1, D)
            text_features = F.normalize(text_features, dim=-1)

        return text_features.float().cpu()

    def encode_text_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode multiple text queries.

        Args:
            texts: List of query text strings

        Returns:
            Text features of shape (B, D) float32 on CPU, L2 normalized
        """
        with torch.no_grad():
            text_tokens = self.lang_adaptor.tokenizer(texts).to(self.device)
            text_features = self.lang_adaptor.encode_text(text_tokens)  # (B, D)
            text_features = F.normalize(text_features, dim=-1)

        return text_features.float().cpu()

    @property
    def feature_dim(self) -> int:
        """Get the feature dimension C."""
        with torch.no_grad():
            # Get feature dimension from a dummy input
            dummy = torch.zeros(1, 3, self.config.input_resolution, self.config.input_resolution)
            dummy = dummy.to(self.device)
            outputs = self.model(dummy)
            feat = outputs["backbone"].features

            if self.config.language_aligned:
                feat = self.lang_adaptor.head_mlp(feat)

            return feat.shape[-1]

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Convenience method for feature extraction.

        Args:
            image: RGB image (H, W, 3) uint8

        Returns:
            Features tensor (H, W, C)
        """
        return self.extract_features(image)

    def to(self, device: str):
        """Move model to different device.

        Args:
            device: Target device ("cuda" or "cpu")
        """
        self.device = ensure_device(device)
        self.model = self.model.to(self.device)
        self.lang_adaptor = self.lang_adaptor.to(self.device)
        return self
