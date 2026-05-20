"""nn - DIY Neural Network Framework (NumPy-based)"""
from .tensor import Tensor, cat
from .nn import Module, Linear, Embedding, RMSNorm, Adam, Sequential, ReLU, Tanh, mse_loss
from .gpt import GPT
from .chargpt import train_model, generate_samples
from .cnn import Conv2d, MaxPool2d, AvgPool2d, Flatten, BatchNorm2d, Dropout2d

try:
    from .datasets import (
        DataLoader,
        Dataset,
        datasets,
        Compose,
        Grayscale,
        Normalize,
        RandomRotation,
        Resize,
        ToTensor,
    )
    _HAS_DATASETS = True
except ImportError:
    _HAS_DATASETS = False

__all__ = [
    "Tensor", "cat",
    "Module", "Linear", "Embedding", "RMSNorm", "Adam", "Sequential", "ReLU", "Tanh", "mse_loss",
    "GPT",
    "train_model", "generate_samples",
    "Conv2d", "MaxPool2d", "AvgPool2d", "Flatten", "BatchNorm2d", "Dropout2d",
    "DataLoader",
    "Dataset",
    "datasets",
    "Compose",
    "Grayscale",
    "Normalize",
    "RandomRotation",
    "Resize",
    "ToTensor",
]