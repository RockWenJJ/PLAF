"""
Configuration management for PLAF (Pixel-level Language-aligned Features).

This module provides configuration classes for managing PLAF model parameters,
including RADIO feature extractor, SAM mask generator, and feature fusion settings.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import yaml


@dataclass
class RadioConfig:
    """Configuration for RADIO v2.5 feature extractor.

    Attributes:
        model_version: RADIO model version to use
        lang_model: Language model for alignment ("siglip")
        input_resolution: Input resolution for RADIO encoder (square)
        language_aligned: Whether to use language-aligned features
        compile_model: Whether to compile model with torch.compile
        amp: Whether to use automatic mixed precision
        device: Device to run on ("cuda" or "cpu")
    """
    model_version: str = "radio_v2.5-b"
    lang_model: str = "siglip"
    input_resolution: int = 512
    language_aligned: bool = True
    compile_model: bool = False
    amp: bool = True
    device: str = "cuda"

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_devices = ["cuda", "cpu", "mps"]
        if self.device not in valid_devices:
            raise ValueError(f"Invalid device: {self.device}. Must be one of {valid_devices}")

        if self.input_resolution <= 0:
            raise ValueError(f"input_resolution must be positive, got {self.input_resolution}")

        if self.input_resolution % 16 != 0:
            import warnings
            warnings.warn(
                f"input_resolution ({self.input_resolution}) should be divisible by 16 "
                "for optimal performance with RADIO's patch-based architecture"
            )


@dataclass
class SamConfig:
    """Configuration for SAM (Segment Anything Model) mask generator.

    Attributes:
        model_type: SAM model type ("vit_h", "vit_l", "vit_b")
        checkpoint_path: Path to SAM checkpoint (None for auto-resolution)
        points_per_side: Number of grid points for mask generation
        pred_iou_thresh: IoU threshold for keeping masks
        crop_n_layers: Number of crop layers
        crop_n_points_downscale_factor: Downscale factor for crop points
        min_mask_region_area: Minimum area for mask regions
        stability_score_offset: Stability score offset
        stability_score_thresh: Stability score threshold
        box_nms_thresh: NMS threshold for box predictions
    """
    model_type: str = "vit_h"
    checkpoint_path: Optional[str] = None
    points_per_side: int = 16
    pred_iou_thresh: float = 0.85
    crop_n_layers: int = 0
    crop_n_points_downscale_factor: int = 2
    min_mask_region_area: int = 200
    stability_score_offset: float = 1.0
    stability_score_thresh: float = 0.92
    box_nms_thresh: float = 0.7
    device: str = "cuda"

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_models = ["vit_h", "vit_l", "vit_b"]
        if self.model_type not in valid_models:
            raise ValueError(f"Invalid model_type: {self.model_type}. Must be one of {valid_models}")

        if not (0.0 <= self.pred_iou_thresh <= 1.0):
            raise ValueError(f"pred_iou_thresh must be in [0, 1], got {self.pred_iou_thresh}")

        if self.points_per_side <= 0:
            raise ValueError(f"points_per_side must be positive, got {self.points_per_side}")

        if self.checkpoint_path is not None:
            if not os.path.exists(self.checkpoint_path):
                raise FileNotFoundError(f"SAM checkpoint not found: {self.checkpoint_path}")

    def resolve_checkpoint_path(self, checkpoint_dir: Optional[str] = None) -> str:
        """Resolve SAM checkpoint path.

        Args:
            checkpoint_dir: Directory to check for checkpoints (default: ./checkpoints)

        Returns:
            Resolved checkpoint path

        Raises:
            FileNotFoundError: If checkpoint cannot be found
        """
        if self.checkpoint_path is not None:
            return self.checkpoint_path

        checkpoint_names = {
            "vit_h": "sam_vit_h_4b8933.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth",
        }

        if checkpoint_dir is None:
            checkpoint_dir = "./checkpoints"

        ckpt_path = os.path.join(checkpoint_dir, checkpoint_names[self.model_type])

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"SAM checkpoint not found at {ckpt_path}. "
                f"Download it from https://dl.fbaipublicfiles.com/segment_anything/"
            )

        return ckpt_path


@dataclass
class FusionConfig:
    """Configuration for mask-guided feature fusion.

    Attributes:
        alpha_mode: Fusion mode ("constant", "gaussian", "linear")
        sigma_ratio: Sigma parameter for gaussian/linear modes
        alpha_min: Minimum alpha value for fusion
        alpha_max: Maximum alpha value for fusion
        normalize_features: Whether to L2 normalize features after fusion
    """
    alpha_mode: str = "constant"
    sigma_ratio: float = 0.35
    alpha_min: float = 0.0
    alpha_max: float = 1.0
    normalize_features: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_modes = ["constant", "gaussian", "linear"]
        if self.alpha_mode not in valid_modes:
            raise ValueError(f"Invalid alpha_mode: {self.alpha_mode}. Must be one of {valid_modes}")

        if self.sigma_ratio <= 0:
            raise ValueError(f"sigma_ratio must be positive, got {self.sigma_ratio}")

        if not (0.0 <= self.alpha_min <= self.alpha_max <= 1.0):
            raise ValueError(
                f"Invalid alpha range: alpha_min={self.alpha_min}, alpha_max={self.alpha_max}. "
                f"Must satisfy 0 <= alpha_min <= alpha_max <= 1"
            )


@dataclass
class ProcessingConfig:
    """Configuration for image processing.

    Attributes:
        max_input_side: Maximum dimension for input image resize
        desired_height: Desired output height (-1 for original)
        desired_width: Desired output width (-1 for original)
        save_masks: Whether to save mask images
        mask_image_format: Format for saving mask images ("png", "jpg")
        skip_existing: Whether to skip existing outputs
        batch_size: Batch size for processing
    """
    max_input_side: int = 512
    desired_height: int = -1
    desired_width: int = -1
    save_masks: bool = False
    mask_image_format: str = "png"
    skip_existing: bool = True
    batch_size: int = 1

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.max_input_side <= 0:
            raise ValueError(f"max_input_side must be positive, got {self.max_input_side}")

        valid_formats = ["png", "jpg", "jpeg"]
        if self.mask_image_format.lower() not in valid_formats:
            raise ValueError(
                f"Invalid mask_image_format: {self.mask_image_format}. "
                f"Must be one of {valid_formats}"
            )

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")


@dataclass
class PlafConfig:
    """Main configuration class for PLAF.

    This class combines all sub-configurations and provides methods
    for loading/saving from YAML files.

    Attributes:
        radio: RADIO feature extractor configuration
        sam: SAM mask generator configuration
        fusion: Feature fusion configuration
        processing: Image processing configuration
    """
    radio: RadioConfig = field(default_factory=RadioConfig)
    sam: SamConfig = field(default_factory=SamConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PlafConfig":
        """Create config from dictionary."""
        radio = RadioConfig(**config_dict.get("radio", {}))
        sam = SamConfig(**config_dict.get("sam", {}))
        fusion = FusionConfig(**config_dict.get("fusion", {}))
        processing = ProcessingConfig(**config_dict.get("processing", {}))

        # Sync device settings
        if "device" in config_dict:
            device = config_dict["device"]
            radio.device = device
            sam.device = device

        return cls(radio=radio, sam=sam, fusion=fusion, processing=processing)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PlafConfig":
        """Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            PlafConfig instance

        Raises:
            FileNotFoundError: If YAML file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "radio": asdict(self.radio),
            "sam": asdict(self.sam),
            "fusion": asdict(self.fusion),
            "processing": asdict(self.processing),
        }

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML configuration file
        """
        os.makedirs(os.path.dirname(yaml_path) if os.path.dirname(yaml_path) else ".", exist_ok=True)

        with open(yaml_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def save(self, path: str) -> None:
        """Save configuration (auto-detect format from extension)."""
        if path.endswith(".yaml") or path.endswith(".yml"):
            self.to_yaml(path)
        else:
            raise ValueError(f"Unsupported config format: {path}")

    @classmethod
    def load(cls, path: str) -> "PlafConfig":
        """Load configuration (auto-detect format from extension)."""
        if path.endswith(".yaml") or path.endswith(".yml"):
            return cls.from_yaml(path)
        else:
            raise ValueError(f"Unsupported config format: {path}")

    def get_device(self) -> str:
        """Get the device to use for models."""
        return self.radio.device

    def __str__(self) -> str:
        """String representation of configuration."""
        lines = ["PlafConfig:"]
        lines.append(f"  Device: {self.get_device()}")
        lines.append("\n  Radio:")
        for key, value in asdict(self.radio).items():
            lines.append(f"    {key}: {value}")
        lines.append("\n  SAM:")
        for key, value in asdict(self.sam).items():
            lines.append(f"    {key}: {value}")
        lines.append("\n  Fusion:")
        for key, value in asdict(self.fusion).items():
            lines.append(f"    {key}: {value}")
        lines.append("\n  Processing:")
        for key, value in asdict(self.processing).items():
            lines.append(f"    {key}: {value}")
        return "\n".join(lines)


def create_default_config() -> PlafConfig:
    """Create default PLAF configuration."""
    return PlafConfig()


def load_config_or_default(config_path: Optional[str] = None) -> PlafConfig:
    """Load configuration from file or create default.

    Args:
        config_path: Optional path to configuration file

    Returns:
        PlafConfig instance
    """
    if config_path is not None and os.path.exists(config_path):
        return PlafConfig.load(config_path)
    return PlafConfig()


@dataclass
class ADE20KConfig:
    """ADE20K 150-class semantic segmentation config.

    ADE20K has 150 semantic classes (0-149, where 0 is background/unlabeled).

    Attributes:
        class_names: List of 151 class names (index 0 = background, 1-150 = ADE20K classes)
        num_classes: Total number of classes (151 including background)
    """
    class_names: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize class names if not provided."""
        if not self.class_names:
            self.class_names = self._get_default_class_names()

    @property
    def num_classes(self) -> int:
        """Total number of classes including background."""
        return len(self.class_names)

    @staticmethod
    def _get_default_class_names() -> List[str]:
        """Get default ADE20K 150 class names from objectInfo150.txt."""
        return [
            "background",  # 0 (not in original file, added for consistency)
            # Classes 1-150 from ADE20K objectInfo150.txt
            # Using first name from comma-separated list as primary class name
            "wall",  # 1
            "building",  # 2 (building, edifice)
            "sky",  # 3
            "floor",  # 4 (floor, flooring)
            "tree",  # 5
            "ceiling",  # 6
            "road",  # 7 (road, route)
            "bed",  # 8
            "windowpane",  # 9 (windowpane, window)
            "grass",  # 10
            "cabinet",  # 11
            "sidewalk",  # 12 (sidewalk, pavement)
            "person",  # 13 (person, individual, someone, somebody, mortal, soul)
            "earth",  # 14 (earth, ground)
            "door",  # 15 (door, double door)
            "table",  # 16
            "mountain",  # 17 (mountain, mount)
            "plant",  # 18 (plant, flora, plant life)
            "curtain",  # 19 (curtain, drape, drapery, mantle, pall)
            "chair",  # 20
            "car",  # 21 (car, auto, automobile, machine, motorcar)
            "water",  # 22
            "painting",  # 23 (painting, picture)
            "sofa",  # 24 (sofa, couch, lounge)
            "shelf",  # 25
            "house",  # 26
            "sea",  # 27
            "mirror",  # 28
            "rug",  # 29 (rug, carpet, carpeting)
            "field",  # 30
            "armchair",  # 31
            "seat",  # 32
            "fence",  # 33 (fence, fencing)
            "desk",  # 34
            "rock",  # 35 (rock, stone)
            "wardrobe",  # 36 (wardrobe, closet, press)
            "lamp",  # 37
            "bathtub",  # 38 (bathtub, bathing tub, bath, tub)
            "railing",  # 39 (railing, rail)
            "cushion",  # 40
            "base",  # 41 (base, pedestal, stand)
            "box",  # 42
            "column",  # 43 (column, pillar)
            "signboard",  # 44 (signboard, sign)
            "chest of drawers",  # 45 (chest of drawers, chest, bureau, dresser)
            "counter",  # 46
            "sand",  # 47
            "sink",  # 48
            "skyscraper",  # 49
            "fireplace",  # 50 (fireplace, hearth, open fireplace)
            "refrigerator",  # 51 (refrigerator, icebox)
            "grandstand",  # 52 (grandstand, covered stand)
            "path",  # 53
            "stairs",  # 54 (stairs, steps)
            "runway",  # 55
            "case",  # 56 (case, display case, showcase, vitrine)
            "pool table",  # 57 (pool table, billiard table, snooker table)
            "pillow",  # 58
            "screen door",  # 59 (screen door, screen)
            "stairway",  # 60 (stairway, staircase)
            "river",  # 61
            "bridge",  # 62 (bridge, span)
            "bookcase",  # 63
            "blind",  # 64 (blind, screen)
            "coffee table",  # 65
            "toilet",  # 66
            "flower",  # 67
            "book",  # 68
            "hill",  # 69
            "bench",  # 70
            "countertop",  # 71
            "stove",  # 72 (stove, kitchen stove, range, kitchen range, cooking stove)
            "palm",  # 73 (palm, palm tree)
            "kitchen island",  # 74
            "computer",  # 75 (computer, computing machine, computing device, data processor, electronic computer, information processing system)
            "swivel chair",  # 76
            "boat",  # 77
            "bar",  # 78
            "arcade machine",  # 79
            "hovel",  # 80 (hovel, hut, hutch, shack, shanty)
            "bus",  # 81 (bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle)
            "towel",  # 82
            "light",  # 83 (light, light source)
            "truck",  # 84 (truck, motortruck)
            "tower",  # 85
            "chandelier",  # 86 (chandelier, pendant, pendent)
            "awning",  # 87 (awning, sunshade, sunblind)
            "streetlight",  # 88 (streetlight, street lamp)
            "booth",  # 89 (booth, cubicle, stall, kiosk)
            "television",  # 90 (television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box)
            "airplane",  # 91 (airplane, aeroplane, plane)
            "dirt track",  # 92
            "apparel",  # 93 (apparel, wearing apparel, dress, clothes)
            "pole",  # 94
            "land",  # 95 (land, ground, soil)
            "bannister",  # 96 (bannister, banister, balustrade, balusters, handrail)
            "escalator",  # 97 (escalator, moving staircase, moving stairway)
            "ottoman",  # 98 (ottoman, pouf, pouffe, puff, hassock)
            "bottle",  # 99
            "buffet",  # 100 (buffet, counter, sideboard)
            "poster",  # 101 (poster, posting, placard, notice, bill, card)
            "stage",  # 102
            "van",  # 103
            "ship",  # 104
            "fountain",  # 105
            "conveyer belt",  # 106 (conveyer belt, conveyor belt, conveyer, conveyor, transporter)
            "canopy",  # 107
            "washer",  # 108 (washer, automatic washer, washing machine)
            "plaything",  # 109 (plaything, toy)
            "swimming pool",  # 110
            "stool",  # 111
            "barrel",  # 112 (barrel, cask)
            "basket",  # 113 (basket, handbasket)
            "waterfall",  # 114 (waterfall, falls)
            "tent",  # 115 (tent, collapsible shelter)
            "bag",  # 116
            "minibike",  # 117 (minibike, motorbike)
            "cradle",  # 118
            "oven",  # 119
            "ball",  # 120
            "food",  # 121 (food, solid food)
            "step",  # 122 (step, stair)
            "tank",  # 123 (tank, storage tank)
            "trade name",  # 124 (trade name, brand name, brand, marque)
            "microwave",  # 125 (microwave, microwave oven)
            "pot",  # 126 (pot, flowerpot)
            "animal",  # 127 (animal, animate being, beast, brute, creature, fauna)
            "bicycle",  # 128 (bicycle, bike, wheel, cycle)
            "lake",  # 129
            "dishwasher",  # 130 (dishwasher, dish washer, dishwashing machine)
            "screen",  # 131 (screen, silver screen, projection screen)
            "blanket",  # 132 (blanket, cover)
            "sculpture",  # 133
            "hood",  # 134 (hood, exhaust hood)
            "sconce",  # 135
            "vase",  # 136
            "traffic light",  # 137 (traffic light, traffic signal, stoplight)
            "tray",  # 138
            "ashcan",  # 139 (ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin)
            "fan",  # 140
            "pier",  # 141 (pier, wharf, wharfage, dock)
            "crt screen",  # 142
            "plate",  # 143
            "monitor",  # 144 (monitor, monitoring device)
            "bulletin board",  # 145 (bulletin board, notice board)
            "shower",  # 146
            "radiator",  # 147
            "glass",  # 148 (glass, drinking glass)
            "clock",  # 149
            "flag",  # 150
        ]


def load_ade20k_config() -> ADE20KConfig:
    """Load ADE20K 150 class configuration.

    Returns:
        ADE20KConfig with 151 class names (background + 150 ADE20K classes).
    """
    return ADE20KConfig()
