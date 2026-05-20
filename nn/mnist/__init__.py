"""nn/mnist - MNIST training and prediction using ai4/nn framework."""

from .train import train, save_model
from .predict import predict, load_model

__all__ = ["train", "save_model", "predict", "load_model"]