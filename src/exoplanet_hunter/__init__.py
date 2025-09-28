"""
ExoplanetHunter-TransitML: Machine Learning for Exoplanet Detection from Transit Data

This package provides tools for detecting exoplanets using machine learning
techniques applied to transit light curves.
"""

__version__ = "0.1.0"
__author__ = "ExoplanetHunter Team"

from .data_loader import TransitDataLoader
from .preprocessor import TransitPreprocessor
from .models import TransitClassifier
from .validator import GroupedTimeSeriesValidator

__all__ = [
    "TransitDataLoader",
    "TransitPreprocessor", 
    "TransitClassifier",
    "GroupedTimeSeriesValidator"
]