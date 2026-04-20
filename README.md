# PLAF: Pixel-wise Language-Aligned Features for 3D Scene Understanding

**Paper**: "PLAF: Pixel-wise Language-Aligned Feature Extraction for Efficient 3D Scene Understanding" ([arXiv:2604.15770](https://arxiv.org/abs/2604.15770))

This repository implements PLAF, a method for efficient open-vocabulary 3D scene understanding using RADIO v2.5 + SAM.

## Key Features

- **Pixel-wise Language Alignment**: RADIO v2.5 + SIGLIP for dense, accurate features (1152-dim)
- **Mask-Indexed Storage**: >99% compression compared to dense storage
- **Open-Vocabulary**: Zero-shot recognition without fine-tuning
- **Multi-Modal Support**: 2D images, 3D point clouds, text queries
- **Interactive Web Viewer**: Browser-based 3D point cloud viewer with real-time text queries

## Project Structure

```
PLAF/
├── plaf/                      # Core package
│   ├── core/                  # Feature extraction and fusion
│   │   ├── radio_feature_extractor.py    # RADIO v2.5 features
│   │   ├── sam_mask_generator.py          # SAM mask generation
│   │   └── feature_fusion.py              # Mask-guided fusion
│   ├── storage/               # 2D/3D storage with compression
│   │   └── feature_pool_3d.py            # 3D observation pooling
│   ├── baselines/             # ConceptFusion, OpenMask3D
│   │   ├── concept_fusion.py             # CLIP ViT-B/32 (512-dim)
│   │   └── openmask3d.py                 # Multi-level CLIP (512-dim)
│   └── utils/                 # Model loading, config
│       ├── model_loader.py               # RADIO/SAM loading with offline support
│       └── config.py                     # Config management, ADE20K mapping
│
├── scripts/
│   ├── preprocessing/
│   │   └── preprocess_scannet.py          # ScanNet .sens extraction + label conversion
│   └── evaluation/
│       ├── query_image_text.py            # Interactive 2D text-image query
│       ├── query_pointcloud_text_scannet.py  # Interactive 3D text-pointcloud query
│       └── pointcloud_viewer.html         # Three.js WebGL viewer template
│
└── data/                                  # Dataset storage
```

## Installation

### 1. Environment Setup

```bash
conda create -n plaf python=3.10 -y
conda activate plaf

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Install Dependencies

```bash
cd /path/to/PLAF

# Install core + models
pip install -r requirements.txt

# Optional: for 3D point cloud web viewer
pip install flask

# Optional: for ScanNet 3D preprocessing (--include-3d)
pip install plyfile

# Optional: for Open3D desktop viewer (fallback)
pip install open3d
```

### 3. Download Models

```bash
# SAM checkpoint (auto-downloaded on first use, or manually):
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P checkpoints/

# RADIO v2.5 is auto-downloaded via torch.hub on first use
```

## Quick Start

### Prepare ScanNet Data

```bash
# 2D only (images + semantic labels)
python scripts/preprocessing/preprocess_scannet.py \
    --scannet-root /path/to/scannet/scans \
    --output-dir ./data/scannet_eval \
    --scene scene0000_00

# 2D + 3D (includes point clouds with semantic labels)
python scripts/preprocessing/preprocess_scannet.py \
    --scannet-root /path/to/scannet/scans \
    --output-dir ./data/scannet_eval \
    --scene scene0000_00 \
    --include-3d
```

### Interactive 2D Text-Image Query

```bash
# PLAF method (default)
python scripts/evaluation/query_image_text.py \
    --input_image ./image.jpg --method plaf

# ConceptFusion baseline
python scripts/evaluation/query_image_text.py \
    --input_image ./image.jpg --method concept_fusion

# Batch queries (non-interactive)
python scripts/evaluation/query_image_text.py \
    --input_image ./image.jpg --method plaf \
    --queries "chair" "table" "monitor" \
    --output-dir ./results
```

### Interactive 3D Point Cloud Query

```bash
# Web viewer (default, recommended)
python scripts/evaluation/query_pointcloud_text_scannet.py \
    --data-root ./data/scannet_eval/scene0000_00 \
    --method plaf --num-frames 20

# With initial query
python scripts/evaluation/query_pointcloud_text_scannet.py \
    --data-root ./data/scannet_eval/scene0000_00 \
    --method concept_fusion --num-frames 10 \
    --query "chair"

# Save/load features for faster re-querying
python scripts/evaluation/query_pointcloud_text_scannet.py \
    --data-root ./data/scannet_eval/scene0000_00 \
    --method plaf --num-frames 50 \
    --save-features ./features/scene_plaf.pkl

# Load pre-computed features
python scripts/evaluation/query_pointcloud_text_scannet.py \
    --data-root ./data/scannet_eval/scene0000_00 \
    --method plaf \
    --load-features ./features/scene_plaf.pkl \
    --query "chair table monitor" \
    --output-dir ./results
```

## Python API

### Feature Extraction

```python
from plaf.core import RadioFeatureExtractor, SamMaskGenerator, FeatureFusion
import numpy as np
from PIL import Image

# Load image
image = np.array(Image.open("image.jpg"))

# Initialize components
radio_extractor = RadioFeatureExtractor(device="cuda")
sam_generator = SamMaskGenerator(device="cuda")
fusion = FeatureFusion(device="cuda")

# Extract RADIO features (1152-dim)
features = radio_extractor.extract_features(image)  # (H, W, 1152)

# Generate SAM masks
masks = sam_generator.generate_masks(image)

# Fuse features with mask guidance
pixel_features = fusion.fuse_mask_features(features.cpu(), masks)

# Access results
mask_ids = pixel_features.mask_ids         # (H, W) mask indices
mask_features = pixel_features.mask_features  # (num_masks, 1152)
```

### Text-Based Classification

```python
# Encode text queries (uses RADIO's SigLIP adaptor, 1152-dim)
text_features = radio_extractor.encode_text_batch([
    "chair", "table", "wall", "floor"
])  # (4, 1152)

# Compute similarity with mask features
import torch.nn.functional as F
similarity = F.normalize(mask_features, dim=-1) @ F.normalize(text_features, dim=-1).T

# Assign labels
labels = similarity.argmax(dim=-1)  # (num_masks,)
```

### Storage

```python
from plaf.storage import MaskIndexedStorage2D

# Create storage (mask-indexed, >99% compression)
storage = MaskIndexedStorage2D(H=480, W=640, feature_dim=1152)
storage.from_pixel_features(mask_ids, mask_features)
storage.save("output/frame0.pkl")

# Load and reconstruct
storage.load("output/frame0.pkl")
reconstructed = storage.to_pixel_features()
```

## Citation

```bibtex
@article{wen2026plaf,
  title={PLAF: Pixel-wise Language-Aligned Feature Extraction for Efficient 3D Scene Understanding},
  author={Wen, Junjie and He, Junlin and Ma, Fei and Cui, Jinqiang},
  journal={arXiv preprint arXiv:2604.15770},
  year={2026}
}
```

## License

MIT License
