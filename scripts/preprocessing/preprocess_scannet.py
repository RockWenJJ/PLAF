#!/usr/bin/env python3
"""
ScanNet data preprocessing for PLAF experiments.

This script:
1. Extracts RGB-D frames, poses, intrinsics from .sens file using ScanNet SensReader
2. Converts ScanNet instance labels to semantic labels (2D)
3. Extracts and processes 3D point clouds with semantic labels
4. Prepares evaluation data with proper directory structure

Usage:
    # Process single scene (2D only)
    python scripts/preprocessing/preprocess_scannet.py --scene scene0000_00

    # Process multiple scenes
    python scripts/preprocessing/preprocess_scannet.py --scenes scene0000_00 scene0001_00 scene0002_00

    # Process multiple scenes with 3D data
    python scripts/preprocessing/preprocess_scannet.py --scenes scene0000_00 scene0001_00 --include-3d

    # Process all scenes
    python scripts/preprocessing/preprocess_scannet.py --all --include-3d

    # Process with frame skip (faster for testing)
    python scripts/preprocessing/preprocess_scannet.py --scenes scene0000_00 --frame-skip 50
"""

import argparse
import gzip
import json
import logging
import os
import shutil
import struct
import sys
import zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import imageio
import numpy as np
import png
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from plyfile import PlyData
    HAS_PLY = True
except ImportError:
    HAS_PLY = False
    logging.warning("plyfile not installed. 3D processing will be disabled. Install with: pip install plyfile")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# ScanNet SensReader Integration
# =============================================================================

COMPRESSION_TYPE_COLOR = {-1: 'unknown', 0: 'raw', 1: 'png', 2: 'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1: 'unknown', 0: 'raw_ushort', 1: 'zlib_ushort', 2: 'occi_ushort'}


class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack('f' * 16, file_handle.read(16 * 4)),
            dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
        self.color_data = file_handle.read(self.color_size_bytes)
        self.depth_data = file_handle.read(self.depth_size_bytes)

    def decompress_depth(self, compression_type):
        if compression_type == 'zlib_ushort':
            return self.decompress_depth_zlib()
        else:
            raise ValueError(f"Unsupported depth compression: {compression_type}")

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == 'jpeg':
            return self.decompress_color_jpeg()
        elif compression_type == 'png':
            return self.decompress_color_png()
        else:
            raise ValueError(f"Unsupported color compression: {compression_type}")

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)

    def decompress_color_png(self):
        return cv2.imdecode(np.frombuffer(self.color_data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.load(filename)

    def load(self, filename):
        with open(filename, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            assert self.version == version, f"Version mismatch: expected {self.version}, got {version}"

            strlen = struct.unpack('Q', f.read(8))[0]
            self.sensor_name = f.read(strlen).decode('utf-8')

            self.intrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack('f' * 16, f.read(16 * 4)),
                dtype=np.float32
            ).reshape(4, 4)

            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]
            num_frames = struct.unpack('Q', f.read(8))[0]

            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        logger.info(f'Exporting {len(self.frames)//frame_skip} depth frames to {output_path}')

        for f in range(0, len(self.frames), frame_skip):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            if image_size is not None:
                depth = cv2.resize(
                    depth, (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            # Save as 16-bit PNG
            with open(os.path.join(output_path, f'{f}.png'), 'wb') as out_file:
                writer = png.Writer(
                    width=depth.shape[1], height=depth.shape[0], bitdepth=16
                )
                depth_list = depth.reshape(-1, depth.shape[1]).tolist()
                writer.write(out_file, depth_list)

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        logger.info(f'Exporting {len(self.frames)//frame_skip} color frames to {output_path}')

        for f in range(0, len(self.frames), frame_skip):
            color = self.frames[f].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(
                    color, (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST
                )
            imageio.imwrite(os.path.join(output_path, f'{f}.jpg'), color)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, 'w') as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt='%f')

    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        logger.info(f'Exporting {len(self.frames)//frame_skip} camera poses to {output_path}')

        for f in range(0, len(self.frames), frame_skip):
            self.save_mat_to_file(
                self.frames[f].camera_to_world,
                os.path.join(output_path, f'{f}.txt')
            )

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        logger.info(f'Exporting camera intrinsics to {output_path}')

        self.save_mat_to_file(
            self.intrinsic_color,
            os.path.join(output_path, 'intrinsic_color.txt')
        )
        self.save_mat_to_file(
            self.extrinsic_color,
            os.path.join(output_path, 'extrinsic_color.txt')
        )
        self.save_mat_to_file(
            self.intrinsic_depth,
            os.path.join(output_path, 'intrinsic_depth.txt')
        )
        self.save_mat_to_file(
            self.extrinsic_depth,
            os.path.join(output_path, 'extrinsic_depth.txt')
        )


def extract_frames_from_sens(
    sens_file: Path,
    output_dir: Path,
    frame_skip: int = 1,
    max_frames: int = None
) -> int:
    """
    Extract RGB-D frames, poses, and intrinsics from .sens file.

    Args:
        sens_file: Path to .sens file
        output_dir: Output directory
        frame_skip: Skip every N frames (1 = extract all)
        max_frames: Maximum number of frames to extract (None = no limit)
    """
    logger.info(f"Loading .sens file: {sens_file}")
    sd = SensorData(str(sens_file))

    total_frames = len(sd.frames)
    frames_to_extract = total_frames // frame_skip
    if max_frames is not None:
        frames_to_extract = min(frames_to_extract, max_frames)

    logger.info(f"Extracting {frames_to_extract}/{total_frames} frames (frame_skip={frame_skip})")

    # Extract color images
    sd.export_color_images(str(output_dir / 'color'), frame_skip=frame_skip)
    # Only keep max_frames
    if max_frames is not None:
        _trim_files(output_dir / 'color', max_frames)

    # Extract depth images
    sd.export_depth_images(str(output_dir / 'depth'), frame_skip=frame_skip)
    if max_frames is not None:
        _trim_files(output_dir / 'depth', max_frames)

    # Extract poses
    sd.export_poses(str(output_dir / 'pose'), frame_skip=frame_skip)
    if max_frames is not None:
        _trim_files(output_dir / 'pose', max_frames)

    # Extract intrinsics (only once)
    sd.export_intrinsics(str(output_dir / 'intrinsic'))

    return frames_to_extract


def _trim_files(directory: Path, max_files: int):
    """Trim directory to only keep first max_files."""
    files = sorted(list(directory.glob("*")))
    for file in files[max_files:]:
        file.unlink()


# =============================================================================
# Label Processing
# =============================================================================


def get_aggregation_mapping(aggregation_file: Path) -> Dict[int, str]:
    """Read aggregation.json to map instance IDs to semantic label names."""
    with open(aggregation_file, 'r') as f:
        data = json.load(f)

    instance_to_label = {}
    for seg_group in data.get('segGroups', []):
        instance_id = seg_group['id']
        label = seg_group.get('label', 'unknown').lower()
        instance_to_label[instance_id] = label

    return instance_to_label


def find_aggregation_file(scene_path: Path, scene_name: str) -> Path:
    """Find the aggregation.json file for a scene."""
    agg_files = (
        list(scene_path.glob(f'{scene_name}*.aggregation.json')) +
        list(scene_path.glob('*_aggregation.json'))
    )

    if not agg_files:
        return None

    # Prefer the vh_clean aggregation if available
    for f in agg_files:
        if 'vh_clean' in f.name:
            return f
    return agg_files[0]


def extract_and_convert_labels(
    scene_path: Path,
    scene_name: str,
    aggregation_file: Path,
    output_dir: Path,
    label_type: str = 'label-filt',
    num_frames: int = None,
    frame_skip: int = 1
) -> List[int]:
    """
    Extract labels from zip file and convert to semantic labels.

    Args:
        scene_path: Path to scene directory
        scene_name: Name of the scene
        aggregation_file: Path to aggregation.json
        output_dir: Output directory for converted labels
        label_type: 'label' or 'label-filt'
        num_frames: Number of frames to process (None = all)
        frame_skip: Skip every N frames to match RGB-D extraction

    Returns:
        List of valid frame indices that have labels
    """
    # Find label zip file
    label_zip = scene_path / f"{scene_name}_2d-{label_type}.zip"
    if not label_zip.exists():
        logger.warning(f"Label zip file not found: {label_zip}")
        return 0

    logger.info(f"Extracting labels from {label_zip.name}")

    # Create temp directory for extraction
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())

    try:
        # Extract zip file
        import zipfile
        with zipfile.ZipFile(label_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Get instance to label mapping
        instance_to_label = get_aggregation_mapping(aggregation_file)
        logger.info(f"Loaded {len(instance_to_label)} instance mappings")

        # Create output directory
        output_labels = output_dir / "labels"
        output_labels.mkdir(parents=True, exist_ok=True)

        # Find extracted label directory
        extracted_label_dir = temp_dir / label_type
        if not extracted_label_dir.exists():
            # Try alternative structure
            extracted_label_dir = temp_dir

        label_files = sorted(list(extracted_label_dir.glob("*.png")), key=lambda f: int(f.stem))

        # Get all label indices (original frame indices in ScanNet)
        label_indices = [int(f.stem) for f in label_files]

        # Apply frame_skip: select indices 0, frame_skip, 2*frame_skip, ...
        selected_indices = [idx for i, idx in enumerate(label_indices) if i % frame_skip == 0]

        # Also limit by num_frames if specified
        if num_frames is not None:
            selected_indices = selected_indices[:num_frames]

        logger.info(f"Found {len(label_indices)} label files, selecting {len(selected_indices)} with frame_skip={frame_skip}")

        # Filter label_files to only those with selected indices
        label_files = [f for f in label_files if int(f.stem) in selected_indices]

        logger.info(f"Converting {len(label_files)} label files...")

        for label_file in tqdm(label_files, desc="Converting labels"):
            # Load instance label
            instance_label = np.array(Image.open(label_file))

            # Create instance ID map (preserve raw instance IDs)
            semantic_label = instance_label.copy()

            # Keep the original filename to match RGB-D output
            output_file = output_labels / label_file.name
            Image.fromarray(semantic_label).save(output_file)

        logger.info(f"Converted {len(label_files)} label files")

        return selected_indices

    finally:
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# 3D Point Cloud Processing
# =============================================================================

def find_ply_files(scene_path: Path, scene_name: str) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the PLY files for point cloud and labels."""
    mesh_files = list(scene_path.glob(f'{scene_name}_vh_clean_*.ply'))
    mesh_files = [f for f in mesh_files if 'labels' not in f.name]

    label_files = list(scene_path.glob('*labels.ply'))

    if mesh_files:
        mesh_files.sort(key=lambda x: x.name, reverse=True)
        mesh_ply = mesh_files[0] if mesh_files else None
    else:
        mesh_ply = None

    labels_ply = label_files[0] if label_files else None

    return mesh_ply, labels_ply


def process_point_cloud(
    mesh_ply: Path,
    labels_ply: Path,
    aggregation_file: Path,
    output_dir: Path,
    scene_name: str
) -> Tuple[int, int]:
    """
    Process 3D point cloud with semantic labels.

    Extracts:
    - points_xyz: (N, 3) XYZ coordinates
    - points_rgb: (N, 3) RGB colors
    - points_label: (N,) semantic labels

    Saves as compressed .npz files.
    """
    if not HAS_PLY:
        logger.warning("Skipping 3D processing: plyfile not installed")
        return 0, 0

    logger.info(f"Processing 3D point cloud...")

    # Load instance to label mapping
    instance_to_label = get_aggregation_mapping(aggregation_file)

    # Load mesh PLY (geometry + color)
    mesh_data = PlyData.read(mesh_ply)
    vertex_data = mesh_data['vertex']

    points_xyz = np.vstack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z']
    ]).T.astype(np.float32)

    points_rgb = np.vstack([
        vertex_data['red'],
        vertex_data['green'],
        vertex_data['blue']
    ]).T.astype(np.uint8)

    num_points = points_xyz.shape[0]
    logger.info(f"Loaded {num_points} points from mesh")

    # Load labels PLY
    labels_data = PlyData.read(labels_ply)
    label_vertex_data = labels_data['vertex']

    instance_labels = np.array(
        label_vertex_data['label'],
        dtype=np.int32
    )

    logger.info(f"Loaded {len(instance_labels)} instance labels")

    # Verify point correspondence
    if len(points_xyz) != len(instance_labels):
        logger.warning(
            f"Point count mismatch: {len(points_xyz)} points vs {len(instance_labels)} labels"
        )
        min_points = min(len(points_xyz), len(instance_labels))
        points_xyz = points_xyz[:min_points]
        points_rgb = points_rgb[:min_points]
        instance_labels = instance_labels[:min_points]

    # Convert instance labels to label name IDs
    label_to_id = {}
    next_id = 1
    semantic_labels = np.zeros_like(instance_labels, dtype=np.uint8)

    unique_instances = np.unique(instance_labels)
    for inst_id in unique_instances:
        if inst_id == 0:
            continue
        label_name = instance_to_label.get(int(inst_id), "unknown")
        if label_name not in label_to_id:
            label_to_id[label_name] = next_id
            next_id += 1
        semantic_labels[instance_labels == inst_id] = label_to_id[label_name]

    # Count points per class
    unique_classes, class_counts = np.unique(semantic_labels, return_counts=True)
    logger.info(f"Semantic classes: {dict(zip(unique_classes.tolist(), class_counts.tolist()))}")
    logger.info(f"Label mapping: {label_to_id}")

    # Create output directory
    output_3d_dir = output_dir / scene_name / "points"
    output_3d_dir.mkdir(parents=True, exist_ok=True)

    # Save as compressed numpy file
    output_file = output_3d_dir / "pointcloud.npz"
    np.savez_compressed(
        output_file,
        xyz=points_xyz,
        rgb=points_rgb,
        labels=semantic_labels
    )

    logger.info(f"Saved 3D point cloud to {output_file}")
    logger.info(f"  - Points: {num_points}")
    logger.info(f"  - Classes: {len(unique_classes)}")

    return num_points, len(unique_classes)


# =============================================================================
# Main Processing Function
# =============================================================================

def prepare_scene_data(
    scene_path: Path,
    output_dir: Path,
    label_type: str = 'label-filt',
    include_3d: bool = False,
    frame_skip: int = 1
) -> Tuple[int, int, Optional[int]]:
    """
    Prepare a single scene for evaluation.

    Args:
        scene_path: Path to scene directory
        output_dir: Output directory
        label_type: Type of labels to use
        include_3d: Whether to process 3D point cloud
        frame_skip: Skip every N frames during extraction

    Returns:
        (num_frames, num_labels, num_points_3d)
    """
    scene_name = scene_path.name

    # Find aggregation file
    agg_file = find_aggregation_file(scene_path, scene_name)
    if agg_file is None:
        logger.warning(f"No aggregation.json found for {scene_name}")
        return 0, 0, None

    # Setup output directory
    scene_output_dir = output_dir / scene_name

    # Step 1: First, extract and convert labels to get valid frame indices
    # This tells us which frames actually have labels
    valid_label_indices = extract_and_convert_labels(
        scene_path, scene_name, agg_file, scene_output_dir, label_type, None, frame_skip
    )

    if not valid_label_indices:
        logger.warning(f"No valid label frames found for {scene_name}")
        return 0, 0, None

    # Step 2: Extract RGB-D frames from .sens file, limited to valid label frames
    sens_file = scene_path / f"{scene_name}.sens"
    if sens_file.exists():
        logger.info(f"Found .sens file: {sens_file.name}")
        num_frames = extract_frames_from_sens(
            sens_file, scene_output_dir, frame_skip, max_frames=len(valid_label_indices)
        )
    else:
        # Check if pre-extracted data exists
        color_dir = scene_path / "color"
        if color_dir.exists():
            logger.info(f"Using pre-extracted frames from {color_dir}")
            num_frames = len(valid_label_indices)
            # Copy only frames that have labels
            for subdir in ['color', 'depth', 'pose']:
                src = scene_path / subdir
                dst = scene_output_dir / subdir
                if src.exists():
                    dst.mkdir(parents=True, exist_ok=True)
                    copied = 0
                    for idx in valid_label_indices:
                        src_file = src / f"{idx}.{('jpg' if subdir == 'color' else 'png' if subdir == 'depth' else 'txt')}"
                        if src == 'pose' and not src_file.exists():
                            src_file = src / f"{idx}.txt"
                        if src_file.exists():
                            shutil.copy2(src_file, dst / src_file.name)
                            copied += 1
            logger.info(f"Copied {num_frames} frames with labels")
        else:
            logger.error(f"No .sens file or pre-extracted data found for {scene_name}")
            return 0, 0, None

    # Step 3: Process 3D point cloud if requested
    num_points_3d = None
    if include_3d:
        mesh_ply, labels_ply = find_ply_files(scene_path, scene_name)

        if mesh_ply is None:
            logger.warning(f"No mesh PLY found for {scene_name}")
        elif labels_ply is None:
            logger.warning(f"No labels PLY found for {scene_name}")
        else:
            logger.info(f"Found PLY files:")
            logger.info(f"  - Mesh: {mesh_ply.name}")
            logger.info(f"  - Labels: {labels_ply.name}")

            num_points_3d, _ = process_point_cloud(
                mesh_ply, labels_ply, agg_file, output_dir, scene_name
            )

    return num_frames, len(valid_label_indices), num_points_3d


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ScanNet data for PLAF experiments"
    )
    parser.add_argument(
        '--scannet-root',
        type=str,
        default='/mnt/MyDisk/04-Data/04-3DReconstruction/05-ScanNet/scannet-v1/scans',
        help='Path to ScanNet scans directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/scannet_eval',
        help='Output directory for evaluation data'
    )
    parser.add_argument(
        '--scenes',
        type=str,
        nargs='+',
        default=None,
        help='List of scenes to process (e.g., --scenes scene0000_00 scene0001_00 scene0002_00)'
    )
    parser.add_argument(
        '--scene',
        type=str,
        default=None,
        help='Single scene to process (use --scenes for multiple scenes)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all scenes in scannet-root'
    )
    parser.add_argument(
        '--label-type',
        type=str,
        default='label-filt',
        choices=['label', 'label-filt'],
        help='Type of labels to convert'
    )
    parser.add_argument(
        '--include-3d',
        action='store_true',
        help='Extract and process 3D point clouds with semantic labels'
    )
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='Skip every N frames during extraction (1 = extract all)'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate output after processing'
    )

    args = parser.parse_args()

    scannet_root = Path(args.scannet_root)
    output_dir = Path(args.output_dir)

    if not scannet_root.exists():
        logger.error(f"ScanNet root not found: {scannet_root}")
        return

    # Check for plyfile if 3D processing is requested
    if args.include_3d and not HAS_PLY:
        logger.error("3D processing requested but plyfile is not installed.")
        logger.error("Install with: pip install plyfile")
        return

    # Determine scenes to process
    if args.scenes:
        # Multiple scenes specified with --scenes
        scenes = [scannet_root / s for s in args.scenes]
    elif args.scene:
        # Single scene specified with --scene
        scenes = [scannet_root / args.scene]
    elif args.all:
        # Process all scenes
        scenes = sorted([d for d in scannet_root.iterdir() if d.is_dir()])
    else:
        logger.error("Please specify --scenes, --scene, or --all")
        return

    logger.info(f"Processing {len(scenes)} scenes")
    if args.include_3d:
        logger.info("3D point cloud processing enabled")

    for scene in scenes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {scene.name}")
        logger.info(f"{'='*60}")

        try:
            num_frames, num_labels, num_points = prepare_scene_data(
                scene, output_dir, args.label_type, args.include_3d, args.frame_skip
            )

            logger.info(f"Prepared {num_frames} RGB-D frames")
            logger.info(f"Prepared {num_labels} label files")
            if num_points is not None:
                logger.info(f"Prepared {num_points} 3D points")

        except Exception as e:
            logger.error(f"Error processing {scene.name}: {e}")
            import traceback
            traceback.print_exc()

    logger.info(f"\nDone! Data saved to {output_dir}")


if __name__ == '__main__':
    main()
