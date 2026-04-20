#!/usr/bin/env python3
"""
3D point cloud text query using PLAF and baseline methods.

This script builds a 3D point cloud from RGB-D sequences and enables
open-vocabulary text queries on the 3D scene with RGB visualization.

Usage:
    # Interactive mode with visualization
    python scripts/evaluation/query_pointcloud_text_scannet.py \\
        --data-root ./data/scannet/scene0001_00 \\
        --method concept_fusion \\
        --num-frames 10 \\
        --visualize

    # Single query mode
    python scripts/evaluation/query_pointcloud_text_scannet.py \\
        --data-root ./data/scannet/scene0001_00 \\
        --method concept_fusion \\
        --query "chair" \\
        --num-frames 10 \\
        --visualize

    # With feature saving/loading
    python scripts/evaluation/query_pointcloud_text_scannet.py \\
        --data-root ./data/scannet/scene0001_00 \\
        --method plaf \\
        --num-frames 50 \\
        --save-features ./features/scene0001_00_plaf.pkl

    # Load pre-computed features and query
    python scripts/evaluation/query_pointcloud_text_scannet.py \\
        --data-root ./data/scannet/scene0001_00 \\
        --method plaf \\
        --load-features ./features/scene0001_00_plaf.pkl \\
        --query "chair" \\
        --output-dir ./results
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time
import threading
import webbrowser
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plaf.core import RadioFeatureExtractor, SamMaskGenerator, FeatureFusion
from plaf.baselines.concept_fusion import ConceptFusionExtractor
from plaf.baselines.openmask3d import OpenMask3DExtractor

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logger = logging.getLogger(__name__)
    logger.warning("Open3D not found. Install with: pip install open3d")

try:
    from flask import Flask, jsonify, Response
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Data for a single RGB-D frame."""
    rgb: np.ndarray  # (H, W, 3)
    depth: np.ndarray  # (H, W) in meters
    intrinsics: np.ndarray  # (3, 3) or (4, 4)
    pose: np.ndarray  # (4, 4) camera-to-world (ScanNet format)
    frame_id: int


@dataclass
class PointObservationRGB:
    """
    Represents a single 3D point observation from 2D with RGB color.
    """
    point_3d: np.ndarray  # (3,)
    feature: torch.Tensor  # (C,)
    mask_id: int
    keyframe_id: int
    pixel_coord: Tuple[int, int]
    confidence: float = 1.0
    color: np.ndarray = field(default_factory=lambda: np.array([128, 128, 128], dtype=np.uint8))  # RGB


@dataclass
class PointCloudData:
    """Container for 3D point cloud data with features."""
    points_3d: np.ndarray  # (N, 3)
    colors_rgb: np.ndarray  # (N, 3) in [0, 1]
    features_3d: torch.Tensor  # (N, C)
    feature_dim: int
    method: str
    num_frames: int

    def save(self, path: str):
        """Save to pickle file."""
        save_dict = {
            'points_3d': self.points_3d,
            'colors_rgb': self.colors_rgb,
            'features_3d': self.features_3d.cpu(),
            'feature_dim': self.feature_dim,
            'method': self.method,
            'num_frames': self.num_frames,
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        logger.info(f"Saved point cloud data to {path}")

    @classmethod
    def load(cls, path: str):
        """Load from pickle file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return cls(
            points_3d=data['points_3d'],
            colors_rgb=data['colors_rgb'],
            features_3d=data['features_3d'],
            feature_dim=data['feature_dim'],
            method=data['method'],
            num_frames=data.get('num_frames', 0),
        )


_VIEWER_HTML_PATH = Path(__file__).parent / "pointcloud_viewer.html"


def _load_viewer_html() -> str:
    """Load the point cloud viewer HTML template from file."""
    return _VIEWER_HTML_PATH.read_text(encoding='utf-8')


class PointCloudWebViewer:
    """Web-based 3D point cloud viewer using Flask + Three.js.

    Starts a local HTTP server that serves a WebGL-based viewer.
    Handles large point clouds via voxel-grid downsampling for display
    while keeping full-resolution data for queries.
    """

    def __init__(self, host: str = 'localhost', port: int = 8080,
                 max_display_points: int = 200000):
        self.host = host
        self.port = port
        self.max_display_points = max_display_points

        # Full-resolution data (for queries)
        self.full_points = None       # (N, 3) float32
        self.full_colors_orig = None  # (N, 3) float32

        # Display data (downsampled for browser)
        self.points = None            # (M, 3) float32, M <= max_display_points
        self.colors = None            # (M, 3) float32
        self.original_colors = None   # (M, 3) float32
        self.display_indices = None   # (M,) int - indices into full arrays
        self.was_downsampled = False

        self.generation = 0
        self.meta = {
            "query": "Loading...", "stats": "",
            "num_points": 0, "generation": 0, "colormap": None,
        }
        self._server_started = False
        self.app = Flask(__name__)
        self._setup_routes()

    @staticmethod
    def _voxel_downsample(points: np.ndarray, max_points: int) -> np.ndarray:
        """Downsample via voxel grid, returning indices to keep.

        Uses a hash-based voxel grid: one point per voxel (first hit).
        Falls back to random subsampling if still over limit.
        """
        n = len(points)
        if n <= max_points:
            return np.arange(n)

        # Compute voxel size from target count and bounding volume
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_size = bbox_max - bbox_min
        # Clamp zero dimensions
        bbox_size = np.maximum(bbox_size, 1e-6)
        volume = np.prod(bbox_size)
        voxel_size = (volume / max_points) ** (1.0 / 3.0)
        voxel_size = max(voxel_size, 1e-8)

        # Quantize to integer voxel coordinates
        voxel_coords = np.floor((points - bbox_min) / voxel_size).astype(np.int64)

        # Find unique voxels (keeps first occurrence via lexsort)
        # Sort by voxel coordinates
        sort_idx = np.lexsort(voxel_coords.T[::-1])
        sorted_voxels = voxel_coords[sort_idx]

        # Find boundaries where voxel changes
        changes = np.ones(len(sorted_voxels), dtype=bool)
        changes[1:] = np.any(sorted_voxels[1:] != sorted_voxels[:-1], axis=1)
        unique_in_sorted = np.where(changes)[0]

        # Map back to original indices
        kept = sort_idx[unique_in_sorted]

        if len(kept) > max_points:
            rng = np.random.RandomState(42)
            kept = np.sort(rng.choice(kept, max_points, replace=False))

        return kept

    def _setup_routes(self):
        viewer = self

        @self.app.route('/')
        def index():
            return _load_viewer_html()

        @self.app.route('/api/meta')
        def meta():
            return jsonify(viewer.meta)

        @self.app.route('/api/points.bin')
        def points():
            if viewer.points is None:
                return Response(b'', mimetype='application/octet-stream')
            return Response(
                viewer.points.tobytes(),
                mimetype='application/octet-stream',
                headers={'Content-Length': str(viewer.points.nbytes)},
            )

        @self.app.route('/api/colors.bin')
        def colors():
            if viewer.colors is None:
                return Response(b'', mimetype='application/octet-stream')
            return Response(
                viewer.colors.tobytes(),
                mimetype='application/octet-stream',
                headers={'Content-Length': str(viewer.colors.nbytes)},
            )

    def show(self, points_3d: np.ndarray, colors_rgb: np.ndarray,
             title: str = "Original Point Cloud"):
        """Display point cloud and open browser. Auto-downsamples if needed."""
        self.full_points = np.ascontiguousarray(points_3d.astype(np.float32))
        self.full_colors_orig = np.ascontiguousarray(colors_rgb.astype(np.float32))

        n_full = len(self.full_points)

        if n_full > self.max_display_points:
            self.display_indices = self._voxel_downsample(
                self.full_points, self.max_display_points
            )
            self.was_downsampled = True
            self.points = np.ascontiguousarray(
                self.full_points[self.display_indices]
            )
            self.colors = np.ascontiguousarray(
                self.full_colors_orig[self.display_indices]
            )
            logger.info(
                f"Downsampled {n_full:,} -> {len(self.points):,} points "
                f"for display (voxel grid)"
            )
        else:
            self.display_indices = None
            self.was_downsampled = False
            self.points = self.full_points.copy()
            self.colors = self.full_colors_orig.copy()

        self.original_colors = self.colors.copy()

        stats = f"Points: {len(self.points):,}"
        if self.was_downsampled:
            stats += f" (of {n_full:,})"

        self.meta = {
            "query": title,
            "stats": stats,
            "num_points": len(self.points),
            "generation": self.generation,
            "colormap": None,
        }

        if not self._server_started:
            self._start_server()
            self._server_started = True
            url = f'http://{self.host}:{self.port}'
            print(f"  Opening browser: {url}")
            threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    def update_colors(self, colors_rgb: np.ndarray, query: str = "",
                      stats: str = "", colormap: str = None):
        """Update point cloud colors (for query results).

        colors_rgb must match the FULL point count. It will be subsampled
        for display if downsampling is active.
        """
        self.generation += 1

        if self.display_indices is not None:
            # Subsample colors for display
            display_colors = colors_rgb[self.display_indices].astype(np.float32)
        else:
            display_colors = np.ascontiguousarray(colors_rgb.astype(np.float32))

        self.colors = np.ascontiguousarray(display_colors)

        display_stats = stats
        if self.was_downsampled:
            display_stats += f" | Display: {len(self.points):,}/{len(self.full_points):,}"

        self.meta = {
            "query": query,
            "stats": display_stats,
            "num_points": len(self.points),
            "generation": self.generation,
            "colormap": colormap,
        }

    def reset_colors(self):
        """Reset to original colors."""
        self.generation += 1
        self.colors = self.original_colors.copy()

        stats = f"Points: {len(self.points):,}"
        if self.was_downsampled:
            stats += f" (of {len(self.full_points):,})"

        self.meta = {
            "query": "Original Point Cloud",
            "stats": stats,
            "num_points": len(self.points),
            "generation": self.generation,
            "colormap": None,
        }

    def _start_server(self):
        """Start Flask server in a background daemon thread."""
        import logging as flask_logging
        flask_log = flask_logging.getLogger('werkzeug')
        flask_log.setLevel(flask_logging.WARNING)

        def run():
            self.app.run(
                host=self.host, port=self.port,
                threaded=True, use_reloader=False,
            )

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        logger.info(f"Web viewer started at http://{self.host}:{self.port}")


class ScanNetLoader:
    """Load RGB-D frames from ScanNet dataset structure."""

    def __init__(self, scene_root: Path, depth_scale: float = 1000.0):
        self.scene_root = Path(scene_root)
        self.depth_scale = depth_scale

        self.color_dir = self.scene_root / "color"
        self.depth_dir = self.scene_root / "depth"
        self.pose_dir = self.scene_root / "pose"
        self.intrinsic_dir = self.scene_root / "intrinsic"

        if not self.color_dir.exists():
            raise ValueError(f"Color directory not found: {self.color_dir}")
        if not self.depth_dir.exists():
            raise ValueError(f"Depth directory not found: {self.depth_dir}")

        self.intrinsics = self._load_intrinsics()
        self.frame_ids = self._get_frame_ids()
        logger.info(f"Found {len(self.frame_ids)} frames in {scene_root}")

    def _load_intrinsics(self) -> np.ndarray:
        """Load camera intrinsics from intrinsic_depth.txt."""
        intrinsic_file = self.intrinsic_dir / "intrinsic_depth.txt"
        if intrinsic_file.exists():
            K = np.loadtxt(intrinsic_file)
        else:
            intrinsic_file = self.intrinsic_dir / "intrinsic_color.txt"
            if not intrinsic_file.exists():
                logger.warning("No intrinsics found, using ScanNet defaults")
                K = np.eye(3)
                K[0, 0] = K[1, 1] = 577.87
                K[0, 2] = 319.5
                K[1, 2] = 239.5
            else:
                K = np.loadtxt(intrinsic_file)

        if K.shape == (4, 4):
            K = K[:3, :3]
        return K

    def _get_frame_ids(self) -> List[int]:
        """Get sorted list of available frame IDs."""
        frame_ids = set()
        for f in self.color_dir.glob("*.jpg"):
            frame_ids.add(int(f.stem))
        for f in self.color_dir.glob("*.png"):
            frame_ids.add(int(f.stem))
        return sorted(frame_ids)

    def load_frame(self, frame_id: int) -> Optional[FrameData]:
        """Load a single RGB-D frame."""
        rgb_file = self.color_dir / f"{frame_id}.jpg"
        if not rgb_file.exists():
            rgb_file = self.color_dir / f"{frame_id}.png"
        if not rgb_file.exists():
            return None

        rgb = np.array(Image.open(rgb_file))
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        elif rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        depth_file = self.depth_dir / f"{frame_id}.png"
        if not depth_file.exists():
            depth_file = self.depth_dir / f"{frame_id}.jpg"
        if not depth_file.exists():
            return None

        depth_img = Image.open(depth_file)
        depth = np.array(depth_img, dtype=np.float32)
        depth = depth / self.depth_scale

        pose_file = self.pose_dir / f"{frame_id}.txt"
        if not pose_file.exists():
            return None
        pose = np.loadtxt(pose_file)

        return FrameData(
            rgb=rgb,
            depth=depth,
            intrinsics=self.intrinsics.copy(),
            pose=pose,
            frame_id=frame_id
        )

    def load_frames(self, num_frames: int = -1, stride: int = 1) -> List[FrameData]:
        """Load multiple frames with optional subsampling."""
        if num_frames > 0:
            selected_ids = self.frame_ids[::stride][:num_frames]
        else:
            selected_ids = self.frame_ids[::stride]

        frames = []
        for fid in tqdm(selected_ids, desc="Loading frames"):
            frame = self.load_frame(fid)
            if frame is not None:
                frames.append(frame)
        return frames


def extract_2d_features_plaf(
    frame: FrameData,
    radio_extractor: RadioFeatureExtractor,
    sam_generator: SamMaskGenerator,
    fusion: FeatureFusion
) -> Tuple[np.ndarray, torch.Tensor]:
    """Extract 2D mask-indexed features using PLAF."""
    radio_features = radio_extractor.extract_features(frame.rgb)
    masks = sam_generator.generate_masks(frame.rgb)
    pixel_features = fusion.fuse_mask_features(
        radio_features.cpu(),
        masks
    )
    return pixel_features.mask_ids, pixel_features.mask_features


def extract_2d_features_conceptfusion(
    frame: FrameData,
    extractor: ConceptFusionExtractor,
    device: str
) -> Tuple[np.ndarray, torch.Tensor]:
    """Extract 2D mask-indexed features using ConceptFusion."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    from plaf.utils.model_loader import resolve_sam_checkpoint

    sam_checkpoint = resolve_sam_checkpoint("vit_h", checkpoint_dir="./checkpoints")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.92,
        min_mask_region_area=200,
    )
    masks = mask_generator.generate(frame.rgb)

    mask_list = []
    for m in masks:
        mask_list.append({
            "segmentation": m["segmentation"].astype(bool),
            "area": int(m["segmentation"].sum()),
            "bbox": m["bbox"]
        })

    features = extractor.extract_pixel_features(frame.rgb, mask_list)

    h, w = frame.rgb.shape[:2]
    mask_ids = np.zeros((h, w), dtype=np.int32)
    mask_features_list = []

    for i, mask_dict in enumerate(mask_list):
        mask = mask_dict["segmentation"]
        mask_ids[mask] = i + 1
        mask_feat = features[mask].mean(dim=0)
        mask_features_list.append(mask_feat)

    mask_union = np.zeros((h, w), dtype=bool)
    for mask_dict in mask_list:
        mask_union = mask_union | mask_dict["segmentation"]
    if (~mask_union).sum() > 0:
        background_feat = features[~mask_union].mean(dim=0)
        mask_features_list.insert(0, background_feat)

    mask_features = torch.stack(mask_features_list)
    return mask_ids, mask_features


def extract_2d_features_openmask3d(
    frame: FrameData,
    extractor: OpenMask3DExtractor,
    device: str
) -> Tuple[np.ndarray, torch.Tensor]:
    """Extract 2D mask-indexed features using OpenMask3D."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    from plaf.utils.model_loader import resolve_sam_checkpoint

    sam_checkpoint = resolve_sam_checkpoint("vit_h", checkpoint_dir="./checkpoints")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    sam.eval()

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.92,
        min_mask_region_area=200,
    )
    masks = mask_generator.generate(frame.rgb)

    mask_list = []
    for m in masks:
        mask_list.append({
            "segmentation": m["segmentation"].astype(bool),
            "area": int(m["segmentation"].sum()),
            "bbox": m["bbox"]
        })

    h, w = frame.rgb.shape[:2]
    mask_ids = np.zeros((h, w), dtype=np.int32)
    mask_features_list = []

    for i, mask_dict in enumerate(mask_list):
        mask = mask_dict["segmentation"]
        mask_ids[mask] = i + 1
        mask_feat = extractor.extract_mask_features(
            frame.rgb, mask, use_sam_refinement=False
        )
        mask_features_list.append(mask_feat.cpu())

    center_h, center_w = h // 2, w // 2
    crop_size = min(224, min(h, w) // 4)
    center_crop = frame.rgb[
        max(0, center_h - crop_size):min(h, center_h + crop_size),
        max(0, center_w - crop_size):min(w, center_w + crop_size)
    ]
    pil_crop = Image.fromarray(center_crop)
    if pil_crop.size[0] < 224 or pil_crop.size[1] < 224:
        pil_crop = pil_crop.resize((224, 224), Image.LANCZOS)

    import clip
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        crop_input = clip_preprocess(pil_crop).unsqueeze(0).to(device)
        bg_feat = clip_model.encode_image(crop_input).float()
        bg_feat = bg_feat / bg_feat.norm(dim=-1, keepdim=True)

    mask_features_list.insert(0, bg_feat.squeeze(0).cpu())
    mask_features = torch.stack(mask_features_list)
    return mask_ids, mask_features


def backproject_frame_rgb(
    frame: FrameData,
    mask_ids: np.ndarray,
    features_2d: torch.Tensor,
    max_depth: float = 5.0,
    stride: int = 4
) -> List[PointObservationRGB]:
    """
    Back-project 2D pixels to 3D world coordinates with RGB colors.

    IMPORTANT: ScanNet pose format is camera-to-world (4x4 matrix).
    p_world = R_cw @ p_cam + t_cw

    Args:
        frame: RGB-D frame with depth, intrinsics, pose, and RGB
        mask_ids: Mask ID map (H, W) from SAM/RADIO fusion
        features_2d: 2D mask features (M, C) where M is number of masks
        max_depth: Maximum depth for back-projection
        stride: Pixel stride for downsampling (reduces point count)

    Returns:
        observations: List of PointObservationRGB with colors
    """
    observations = []

    # Resize mask_ids to match depth if needed
    if mask_ids.shape[:2] != frame.depth.shape[:2]:
        mask_img = Image.fromarray(mask_ids.astype(np.uint16))
        mask_img = mask_img.resize((frame.depth.shape[1], frame.depth.shape[0]), Image.NEAREST)
        mask_ids = np.array(mask_img, dtype=mask_ids.dtype)

    # Resize RGB to match depth if needed
    rgb_resized = frame.rgb
    if rgb_resized.shape[:2] != frame.depth.shape[:2]:
        rgb_resized = np.array(Image.fromarray(frame.rgb).resize(
            (frame.depth.shape[1], frame.depth.shape[0]), Image.BILINEAR
        ))

    # Extract camera parameters
    fx, fy = frame.intrinsics[0, 0], frame.intrinsics[1, 1]
    cx, cy = frame.intrinsics[0, 2], frame.intrinsics[1, 2]

    # ScanNet pose format is camera-to-world (4x4 matrix)
    # p_world = R_cw @ p_cam + t_cw
    R_cw = frame.pose[:3, :3]
    t_cw = frame.pose[:3, 3]

    # Create pixel grid with stride
    h_d, w_d = frame.depth.shape
    v_coords, u_coords = np.meshgrid(
        np.arange(0, h_d, stride),
        np.arange(0, w_d, stride),
        indexing='ij'
    )

    # Flatten
    v_flat = v_coords.ravel()
    u_flat = u_coords.ravel()

    # Sample depth at stride locations
    d_flat = frame.depth[::stride, ::stride].ravel()
    mask_ids_flat = mask_ids[::stride, ::stride].ravel()

    # RGB flattened (sampled at stride)
    rgb_flat = rgb_resized[::stride, ::stride].reshape(-1, 3)

    # Filter by valid depth and mask
    valid = (d_flat > 0) & (d_flat < max_depth) & (mask_ids_flat > 0)

    if not valid.any():
        return observations

    # Apply filters
    v_valid = v_flat[valid]
    u_valid = u_flat[valid]
    d_valid = d_flat[valid]
    mask_ids_valid = mask_ids_flat[valid]
    rgb_valid = rgb_flat[valid]

    # Back-project to camera coordinates
    x_cam = (u_valid - cx) * d_valid / fx
    y_cam = (v_valid - cy) * d_valid / fy
    z_cam = d_valid

    # Transform to world coordinates using camera-to-world pose
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    points_world = (R_cw @ points_cam.T).T + t_cw

    # Create observations
    unique_mask_ids = np.unique(mask_ids_valid)

    for mask_id in unique_mask_ids:
        if mask_id >= len(features_2d):
            continue

        mask_mask = mask_ids_valid == mask_id
        if not mask_mask.any():
            continue

        mask_points = points_world[mask_mask]
        mask_u = u_valid[mask_mask]
        mask_v = v_valid[mask_mask]
        mask_depth = d_valid[mask_mask]
        mask_rgb = rgb_valid[mask_mask]

        # Get feature for this mask
        mask_feature = features_2d[mask_id]

        # Confidence based on depth
        confidence = 1.0 - (mask_depth / max_depth)
        confidence = np.clip(confidence, 0.1, 1.0)

        # Subsample points if too many in same mask
        n_points = len(mask_points)
        max_points_per_mask = 200
        if n_points > max_points_per_mask:
            indices = np.random.choice(n_points, max_points_per_mask, replace=False)
            mask_points = mask_points[indices]
            mask_u = mask_u[indices]
            mask_v = mask_v[indices]
            mask_depth = mask_depth[indices]
            mask_rgb = mask_rgb[indices]
            confidence = confidence[indices]

        for i in range(len(mask_points)):
            observations.append(PointObservationRGB(
                point_3d=mask_points[i],
                feature=mask_feature,
                mask_id=int(mask_id),
                keyframe_id=frame.frame_id,
                pixel_coord=(int(mask_u[i]), int(mask_v[i])),
                confidence=float(confidence[i]),
                color=mask_rgb[i].astype(np.uint8)
            ))

    return observations


def build_point_cloud_rgb(
    frames: List[FrameData],
    method: str,
    max_depth: float = 5.0,
    stride: int = 4,
    device: str = "cuda"
) -> PointCloudData:
    """
    Build 3D point cloud with RGB colors and features from RGB-D frames.

    Returns:
        PointCloudData: Container with points, colors, features
    """
    # Initialize extractors based on method
    if method == "plaf":
        logger.info("Initializing PLAF models...")
        radio_extractor = RadioFeatureExtractor(device=device)
        sam_generator = SamMaskGenerator(device=device)
        fusion = FeatureFusion(device=device)
        feature_dim = 1152

    elif method == "concept_fusion":
        logger.info("Initializing ConceptFusion model...")
        cf_extractor = ConceptFusionExtractor(device=device)
        feature_dim = 512

    elif method == "openmask3d":
        logger.info("Initializing OpenMask3D model...")
        om3d_extractor = OpenMask3DExtractor(device=device)
        feature_dim = 512

    all_points = []
    all_colors = []
    all_features = []

    for frame in tqdm(frames, desc="Extracting features and building point cloud"):
        # Extract 2D features
        if method == "plaf":
            mask_ids, mask_features = extract_2d_features_plaf(
                frame, radio_extractor, sam_generator, fusion
            )
        elif method == "concept_fusion":
            mask_ids, mask_features = extract_2d_features_conceptfusion(
                frame, cf_extractor, device
            )
        elif method == "openmask3d":
            mask_ids, mask_features = extract_2d_features_openmask3d(
                frame, om3d_extractor, device
            )

        # Back-project to 3D with RGB
        observations = backproject_frame_rgb(
            frame, mask_ids, mask_features, max_depth, stride
        )

        for obs in observations:
            all_points.append(obs.point_3d)
            all_colors.append(obs.color)
            all_features.append(obs.feature)

    if not all_points:
        logger.warning("No valid 3D observations!")
        return PointCloudData(
            points_3d=np.zeros((0, 3)),
            colors_rgb=np.zeros((0, 3)),
            features_3d=torch.zeros((0, feature_dim)),
            feature_dim=feature_dim,
            method=method,
            num_frames=len(frames),
        )

    points_3d = np.stack(all_points, axis=0).astype(np.float32)
    colors = np.stack(all_colors, axis=0).astype(np.float32) / 255.0  # Normalize to [0, 1]
    features_3d = torch.stack(all_features)

    logger.info(f"Built point cloud: {len(points_3d)} points, {features_3d.shape[-1]}-dim features")

    return PointCloudData(
        points_3d=points_3d,
        colors_rgb=colors,
        features_3d=features_3d,
        feature_dim=feature_dim,
        method=method,
        num_frames=len(frames),
    )


def encode_text_plaf(query: str, radio_extractor: RadioFeatureExtractor) -> torch.Tensor:
    """Encode text query using RADIO's language adaptor."""
    with torch.no_grad():
        text_features = radio_extractor.encode_text(query)
    # Move to same device as extractor for consistency
    return text_features.squeeze(0).to(radio_extractor.device)


def encode_text_clip(query: str, device: str) -> torch.Tensor:
    """Encode text query using CLIP."""
    import clip
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    text_tokens = clip.tokenize([query], truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)

    return text_features.squeeze(0)


def query_point_cloud(
    point_cloud_data: PointCloudData,
    text_feature: torch.Tensor,
    device: str
) -> np.ndarray:
    """
    Query 3D point cloud by text.

    Returns:
        similarities: Similarity scores (N,) in [0, 1]
    """
    features_3d = point_cloud_data.features_3d

    # Ensure features are on the same device
    features_3d = features_3d.to(device)

    # Normalize features
    features_norm = F.normalize(features_3d.float(), dim=-1)
    text_norm = F.normalize(text_feature.float(), dim=-1)

    # Compute similarities
    similarities = (features_norm @ text_norm).cpu().numpy()

    # Normalize to [0, 1]
    min_val, max_val = similarities.min(), similarities.max()
    if max_val > min_val:
        similarities = (similarities - min_val) / (max_val - min_val)
    else:
        similarities = np.ones_like(similarities) * 0.5

    return similarities


def color_pointcloud_by_similarity(
    colors: np.ndarray,
    similarities: np.ndarray,
    colormap: str = 'jet',
    blend_with_original: bool = True,
    blend_alpha: float = 0.5
) -> np.ndarray:
    """
    Color point cloud by similarity scores with optional RGB blending.
    """
    import matplotlib.pyplot as plt

    # Get colormap
    try:
        cmap = plt.cm.get_cmap(colormap)
    except AttributeError:
        # Matplotlib 3.5+
        cmap = plt.cm.colormaps.get_cmap(colormap)

    similarity_colors = cmap(similarities)[:, :3]

    if blend_with_original:
        # Blend original colors with similarity colors
        result_colors = (1 - blend_alpha) * colors + blend_alpha * similarity_colors
    else:
        result_colors = similarity_colors

    return np.clip(result_colors, 0, 1)


def downsample_point_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    voxel_size: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample point cloud using voxel grid filter.

    Returns:
        downsampled_points, downsampled_colors
    """
    if not HAS_OPEN3D:
        # Simple random downsampling fallback
        n_points = len(points)
        target_points = min(100000, n_points)
        indices = np.random.choice(n_points, target_points, replace=False)
        return points[indices], colors[indices]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Voxel grid filter
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    down_points = np.asarray(pcd_down.points)
    down_colors = np.asarray(pcd_down.colors)

    return down_points, down_colors


def create_open3d_pointcloud(
    points_3d: np.ndarray,
    colors: np.ndarray
) -> 'o3d.geometry.PointCloud':
    """Create Open3D point cloud from numpy arrays."""
    if not HAS_OPEN3D:
        raise ImportError("Open3D is required for visualization")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_pointcloud_open3d(
    points_3d: np.ndarray,
    colors: np.ndarray,
    title: str = "Point Cloud",
    voxel_size: Optional[float] = None,
    max_points: int = 200000
):
    """
    Visualize point cloud using Open3D.
    Automatically downsamples if too many points.
    Returns True if visualization succeeded, False otherwise.
    """
    if not HAS_OPEN3D:
        logger.warning("Open3D not available. Skipping visualization.")
        return False

    # Check if DISPLAY is set
    if not os.environ.get('DISPLAY'):
        logger.warning("DISPLAY environment variable not set. Cannot show visualization window.")
        logger.info("To enable visualization:")
        logger.info("  - If using SSH: Add -X flag (ssh -X user@host)")
        logger.info("  - If using WSL: Install XServer like VcXsrv")
        return False

    # Downsample if too many points
    if len(points_3d) > max_points:
        logger.info(f"Downsampling {len(points_3d)} points to {max_points} for visualization...")
        points_3d, colors = downsample_point_cloud(points_3d, colors)
    elif voxel_size is not None:
        points_3d, colors = downsample_point_cloud(points_3d, colors, voxel_size)

    pcd = create_open3d_pointcloud(points_3d, colors)

    print(f"\nDisplaying: {title}")
    print(f"  Points: {len(points_3d)}")
    print("Controls:")
    print("  - Mouse drag: Rotate")
    print("  - Mouse wheel + drag: Zoom")
    print("  - Shift + drag: Pan")
    print("  - Q or close window: Continue")
    print()

    try:
        # Use non-blocking visualization with callback
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title, width=1280, height=720)
        vis.add_geometry(pcd)

        # Run visualization - this blocks until window is closed
        vis.run()
        vis.destroy_window()
        return True
    except Exception as e:
        logger.warning(f"Could not display Open3D window: {e}")
        logger.info("To enable X11 forwarding:")
        logger.info("  SSH: ssh -X user@hostname")
        logger.info("  Or run with xhost + and set DISPLAY manually")
        return False


def save_point_cloud_ply(points: np.ndarray, colors: np.ndarray, output_path: str):
    """Save point cloud as PLY file."""
    # Ensure colors are in [0, 1]
    if colors.max() <= 1.0:
        colors_display = colors
    else:
        colors_display = colors / 255.0

    if not HAS_OPEN3D:
        # Fallback: save as text
        n = len(points)
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(n):
                c = (colors_display[i] * 255).astype(np.uint8)
                f.write(f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} ")
                f.write(f"{c[0]} {c[1]} {c[2]}\n")
        logger.info(f"Saved point cloud to {output_path}")
    else:
        pcd = create_open3d_pointcloud(points, colors_display)
        o3d.io.write_point_cloud(output_path, pcd)
        logger.info(f"Saved point cloud to {output_path}")


def save_query_result(
    points: np.ndarray,
    colors_original: np.ndarray,
    colors_query: np.ndarray,
    similarities: np.ndarray,
    query: str,
    output_dir: Path
):
    """Save query results including both original and query-colored point clouds."""
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = query.replace(" ", "_").replace("/", "_").replace("\\", "_")[:50]

    # Save original RGB point cloud
    save_point_cloud_ply(
        points, colors_original,
        str(output_dir / f"{safe_name}_original.ply")
    )

    # Save query-colored point cloud
    save_point_cloud_ply(
        points, colors_query,
        str(output_dir / f"{safe_name}_query.ply")
    )

    # Save similarities as numpy
    np.save(str(output_dir / f"{safe_name}_similarities.npy"), similarities)

    # Save metadata
    metadata = {
        'query': query,
        'num_points': len(points),
        'similarity_min': float(similarities.min()),
        'similarity_max': float(similarities.max()),
        'similarity_mean': float(similarities.mean()),
        'similarity_std': float(similarities.std()),
        'high_similarity_count': int((similarities > 0.5).sum()),
    }

    with open(output_dir / f"{safe_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved query results for '{query}' to {output_dir}")


def interactive_query_mode(
    point_cloud_data: PointCloudData,
    text_encoder_fn,
    device: str,
    args,
    web_viewer: Optional[PointCloudWebViewer] = None,
):
    """Interactive query mode that shows results in web browser."""

    print("\n" + "=" * 60)
    print("Interactive 3D Text Query Mode")
    print(f"Point Cloud: {len(point_cloud_data.points_3d):,} points")
    print(f"Feature Dim: {point_cloud_data.feature_dim}")
    if web_viewer:
        print(f"Viewer: http://{web_viewer.host}:{web_viewer.port}")
    print("Commands:")
    print("  <text>   - Query for matching regions")
    print("  reset    - Show original RGB point cloud")
    print("  quit     - Exit")
    print("=" * 60 + "\n")

    query_count = 0

    while True:
        try:
            query = input("Enter text query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if query.lower() == 'reset':
                if web_viewer:
                    web_viewer.reset_colors()
                    print("Reset to original point cloud.")
                continue

            if not query:
                print("Please enter a non-empty query.")
                continue

            query_count += 1
            print(f"\n[{query_count}] Processing query: '{query}'...")

            # Encode text and query
            text_feature = text_encoder_fn(query)
            similarities = query_point_cloud(point_cloud_data, text_feature, device)

            stats_str = (
                f"Similarity: [{similarities.min():.4f}, {similarities.max():.4f}] | "
                f"Mean: {similarities.mean():.4f} | "
                f"High (>0.5): {(similarities > 0.5).sum():,} "
                f"({(similarities > 0.5).sum() / len(similarities) * 100:.1f}%)"
            )
            print(f"  {stats_str}")

            # Color by similarity
            query_colors = color_pointcloud_by_similarity(
                point_cloud_data.colors_rgb, similarities,
                colormap=args.colormap,
                blend_with_original=args.overlay,
                blend_alpha=args.blend_alpha,
            )

            # Update web viewer
            if web_viewer:
                mode_str = " (overlay)" if args.overlay else ""
                web_viewer.update_colors(
                    query_colors,
                    query=f"Query: '{query}'{mode_str}",
                    stats=stats_str,
                    colormap=args.colormap,
                )
                print(f"  Updated viewer: http://{web_viewer.host}:{web_viewer.port}")

            # Save query results if output dir specified
            if args.output_dir:
                save_query_result(
                    point_cloud_data.points_3d,
                    point_cloud_data.colors_rgb,
                    query_colors,
                    similarities,
                    query,
                    Path(args.output_dir),
                )

            # Fallback: Open3D visualization if no web viewer
            if not web_viewer and args.visualize and HAS_OPEN3D:
                mode_str = " (Overlay)" if args.overlay else ""
                shown = visualize_pointcloud_open3d(
                    point_cloud_data.points_3d, query_colors,
                    f"Query: '{query}'{mode_str}",
                )
                if not shown and args.auto_save:
                    default_output = Path("./pointcloud_results")
                    default_output.mkdir(exist_ok=True)
                    save_query_result(
                        point_cloud_data.points_3d,
                        point_cloud_data.colors_rgb,
                        query_colors,
                        similarities,
                        query,
                        default_output,
                    )
                    print(f"  Saved to {default_output} (visualization failed)")

        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error processing query: {e}")
            import traceback
            traceback.print_exc()
            continue


def main():
    if 'HF_HUB_OFFLINE' in os.environ:
        del os.environ['HF_HUB_OFFLINE']

    parser = argparse.ArgumentParser(
        description="3D point cloud text query using PLAF and baseline methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with visualization
  python scripts/evaluation/query_pointcloud_text_scannet.py \\
      --data-root ./data/scannet/scene0001_00 \\
      --method concept_fusion \\
      --num-frames 10 \\
      --visualize

  # Single query mode
  python scripts/evaluation/query_pointcloud_text_scannet.py \\
      --data-root ./data/scannet/scene0001_00 \\
      --method plaf \\
      --query "chair" \\
      --num-frames 20 \\
      --stride 2 \\
      --visualize

  # Save and load features
  python scripts/evaluation/query_pointcloud_text_scannet.py \\
      --data-root ./data/scannet/scene0001_00 \\
      --method plaf \\
      --num-frames 50 \\
      --save-features ./features/scene0001_00_plaf.pkl

  # Load pre-computed features and query
  python scripts/evaluation/query_pointcloud_text_scannet.py \\
      --data-root ./data/scannet/scene0001_00 \\
      --method plaf \\
      --load-features ./features/scene0001_00_plaf.pkl \\
      --query "chair table monitor" \\
      --output-dir ./results
        """
    )

    parser.add_argument("--data-root", type=str, required=True,
                        help="Path to ScanNet scene directory")
    parser.add_argument("--method", type=str, default="concept_fusion",
                        choices=["plaf", "concept_fusion", "openmask3d"],
                        help="Feature extraction method")
    parser.add_argument("--query", type=str, default=None,
                        help="Text query to search for (if not specified, interactive mode)")
    parser.add_argument("--num-frames", type=int, default=10,
                        help="Number of frames to process")
    parser.add_argument("--stride", type=int, default=1,
                        help="Frame stride for sampling")
    parser.add_argument("--pixel-stride", type=int, default=4,
                        help="Pixel stride for back-projection (reduces point count)")
    parser.add_argument("--max-depth", type=float, default=5.0,
                        help="Maximum depth for back-projection (meters)")
    parser.add_argument("--top-k", type=int, default=-1,
                        help="Number of top results to return (-1 for all)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for query results")
    parser.add_argument("--save-features", type=str, default=None,
                        help="Path to save extracted features (.pkl file)")
    parser.add_argument("--load-features", type=str, default=None,
                        help="Path to load pre-extracted features (.pkl file)")
    parser.add_argument("--save-ply", action="store_true",
                        help="Save query results as PLY files (in output-dir)")
    parser.add_argument("--colormap", type=str, default="jet",
                        choices=["jet", "hot", "viridis", "plasma", "inferno", "magma", "turbo"],
                        help="Colormap for similarity visualization")
    parser.add_argument("--overlay", action="store_true",
                        help="Blend similarity colors with original RGB")
    parser.add_argument("--blend-alpha", type=float, default=0.5,
                        help="Blend factor (0.0=original, 1.0=similarity)")
    parser.add_argument("--visualize", action="store_true",
                        help="Enable Open3D visualization (fallback when no web)")
    parser.add_argument("--auto-save", action="store_true",
                        help="Automatically save PLY files to default location if visualization fails")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                        help="Voxel size for downsampling visualization (default: 0.05m)")
    parser.add_argument("--web", action="store_true",
                        help="Enable web-based 3D viewer (default: True)")
    parser.add_argument("--no-web", action="store_true",
                        help="Disable web-based viewer, use Open3D instead")
    parser.add_argument("--web-port", type=int, default=8080,
                        help="Port for web viewer (default: 8080)")
    parser.add_argument("--web-host", type=str, default="localhost",
                        help="Host for web viewer (default: localhost)")
    parser.add_argument("--max-display-points", type=int, default=500000,
                        help="Max points to display in browser; downsamples via voxel grid (default: 500000)")

    args = parser.parse_args()

    # Check if Open3D is available for visualization
    can_visualize = HAS_OPEN3D
    if not can_visualize:
        logger.warning("Open3D not available. Install with: pip install open3d")

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving results to {output_dir}")

    # Load or build point cloud
    if args.load_features:
        logger.info(f"Loading pre-computed features from {args.load_features}")
        point_cloud_data = PointCloudData.load(args.load_features)
        logger.info(f"Loaded {len(point_cloud_data.points_3d)} points, {point_cloud_data.feature_dim}-dim features")
    else:
        # Load frames
        logger.info(f"Loading frames from {args.data_root}")
        loader = ScanNetLoader(Path(args.data_root))
        frames = loader.load_frames(num_frames=args.num_frames, stride=args.stride)
        logger.info(f"Loaded {len(frames)} frames")

        if not frames:
            logger.error("No frames loaded!")
            return

        # Build point cloud
        point_cloud_data = build_point_cloud_rgb(
            frames, args.method, args.max_depth, args.pixel_stride, args.device
        )

        if len(point_cloud_data.points_3d) == 0:
            logger.error("No points in point cloud!")
            return

        # Save features if requested
        if args.save_features:
            point_cloud_data.save(args.save_features)

    # Initialize text encoder
    if args.method == "plaf":
        radio_extractor = RadioFeatureExtractor(device=args.device)
        text_encoder_fn = lambda q: encode_text_plaf(q, radio_extractor)
    else:
        text_encoder_fn = lambda q: encode_text_clip(q, args.device)

    # Print point cloud info
    print("\n" + "=" * 60)
    print("3D Point Cloud Reconstruction Complete")
    print(f"  Points: {len(point_cloud_data.points_3d):,}")
    print(f"  Features: {point_cloud_data.feature_dim}-dim")
    print(f"  Method: {args.method}")
    print("=" * 60)

    # Create web viewer (default) or use Open3D
    use_web = HAS_FLASK and not args.no_web
    web_viewer = None

    if use_web:
        logger.info("Using web-based 3D viewer (Flask + Three.js)")
        web_viewer = PointCloudWebViewer(
            host=args.web_host, port=args.web_port,
            max_display_points=args.max_display_points,
        )
        web_viewer.show(
            point_cloud_data.points_3d,
            point_cloud_data.colors_rgb,
            title="Original RGB Point Cloud",
        )
    elif args.visualize and can_visualize:
        print("\nOpening original RGB point cloud in Open3D...")
        print("Close the window to continue to text query mode.")
        visualize_pointcloud_open3d(
            point_cloud_data.points_3d,
            point_cloud_data.colors_rgb,
            "Original RGB Point Cloud",
            voxel_size=args.voxel_size,
        )

    # Process initial query if specified
    if args.query:
        query = args.query
        print(f"\nProcessing query: '{query}'...")

        text_feature = text_encoder_fn(query)
        similarities = query_point_cloud(point_cloud_data, text_feature, args.device)

        stats_str = (
            f"Similarity: [{similarities.min():.4f}, {similarities.max():.4f}] | "
            f"Mean: {similarities.mean():.4f} | "
            f"High (>0.5): {(similarities > 0.5).sum():,} "
            f"({(similarities > 0.5).sum() / len(similarities) * 100:.1f}%)"
        )
        print(f"  {stats_str}")

        # Color by similarity
        query_colors = color_pointcloud_by_similarity(
            point_cloud_data.colors_rgb, similarities,
            colormap=args.colormap,
            blend_with_original=args.overlay,
            blend_alpha=args.blend_alpha,
        )

        # Update web viewer or show in Open3D
        if web_viewer:
            mode_str = " (overlay)" if args.overlay else ""
            web_viewer.update_colors(
                query_colors,
                query=f"Query: '{query}'{mode_str}",
                stats=stats_str,
                colormap=args.colormap,
            )
        elif args.visualize and can_visualize:
            mode_str = " (Overlay)" if args.overlay else ""
            shown = visualize_pointcloud_open3d(
                point_cloud_data.points_3d, query_colors,
                f"Query: '{query}'{mode_str}",
                voxel_size=args.voxel_size,
            )
            if not shown and args.auto_save:
                default_output = Path("./pointcloud_results")
                default_output.mkdir(exist_ok=True)
                save_query_result(
                    point_cloud_data.points_3d,
                    point_cloud_data.colors_rgb,
                    query_colors,
                    similarities,
                    query,
                    default_output,
                )
                print(f"  Saved to {default_output} (visualization failed)")

        # Save query results
        if args.output_dir:
            save_query_result(
                point_cloud_data.points_3d,
                point_cloud_data.colors_rgb,
                query_colors,
                similarities,
                query,
                Path(args.output_dir),
            )

    # Enter interactive query mode
    interactive_query_mode(
        point_cloud_data, text_encoder_fn, args.device, args,
        web_viewer=web_viewer,
    )


if __name__ == "__main__":
    main()
