"""
SpectraLab: Data Science & PCA Analysis Platform

A data-science-first analysis platform for exploring, reducing, and interpreting
high-dimensional scientific datasets with a focus on Principal Component Analysis.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data.loader import DataLoader
from .data.preprocessor import DataPreprocessor
from .analysis.pca_analyzer import PCAAnalyzer
from .visualization.plotter import Plotter

__all__ = [
    'DataLoader',
    'DataPreprocessor',
    'PCAAnalyzer',
    'Plotter',
]