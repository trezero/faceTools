"""
Face Detection Tools Package

This package contains tools for face detection, quality analysis, and training image generation.
"""

from .face_quality import FaceQualityAnalyzer
from .generateTrainingFaces import generate_training_faces

__all__ = ['FaceQualityAnalyzer', 'generate_training_faces']
