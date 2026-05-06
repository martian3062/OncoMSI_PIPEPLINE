"""
Pathology Foundation Models

A Python package for interfacing with foundation models for histopathology image analysis.
"""

__version__ = "0.0.2"
__author__ = "Igor Borja"
__email__ = "igorpradoborja@gmail.com"

# Import key modules for easier access
from pathology_foundation_models import dataset
from pathology_foundation_models import models

__all__ = [
    "dataset",
    "models",
]
