"""
Baseline method implementations for comparison with PLAF.

Implements:
    - ConceptFusion: CLIP global + local features
    - OpenMask3D: CLIP-based 3D segmentation

Reference:
    - ConceptFusion: "ConceptFusion: Open-set Multimodal 3D Mapping" (CVPR 2023)
    - OpenMask3D: "OpenMask3D: Open-Vocabulary 3D Instance Segmentation" (CVPR 2023)
"""

from plaf.baselines.concept_fusion import ConceptFusionExtractor
from plaf.baselines.openmask3d import OpenMask3DExtractor

__all__ = [
    "ConceptFusionExtractor",
    "OpenMask3DExtractor",
]
