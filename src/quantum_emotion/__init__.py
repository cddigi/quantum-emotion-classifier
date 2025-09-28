"""
Quantum Emotion Classifier Package

A hybrid quantum-classical machine learning framework for text emotion classification
using PennyLane and PyTorch.
"""

from .classifier import QuantumTextClassifier
from .kernels import QuantumKernel
from .hybrid import HybridModel
from .encoding import TextEncoder
from .utils import create_emotion_dataset, plot_training_history

__version__ = "0.1.0"
__author__ = "Quantum Emotion AI Team"

__all__ = [
    "QuantumTextClassifier",
    "QuantumKernel",
    "HybridModel",
    "TextEncoder",
    "create_emotion_dataset",
    "plot_training_history"
]