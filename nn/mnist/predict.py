"""MNIST prediction using ai4/nn framework."""

from __future__ import annotations

import numpy as np
from PIL import Image
from nn import Tensor, Module, Linear, Conv2d, MaxPool2d, Flatten
from nn.datasets import Compose, Grayscale, Resize, ToTensor, Normalize


class MNISTNet(Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, bias=True)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, bias=True)
        self.pool1 = MaxPool2d(kernel_size=2)
        self.pool2 = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.fc1 = Linear(in_features=64 * 5 * 5, out_features=128, bias=True)
        self.fc2 = Linear(in_features=128, out_features=10, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = x.relu()
        x = self.pool1(self.conv1(x))
        x = x.relu()
        x = self.pool2(self.conv2(x))
        x = self.flatten(x)
        x = x.relu()
        x = self.fc1(x)
        return self.fc2(x)


transform = Compose([
    Grayscale(),
    Resize((28, 28)),
    ToTensor(),
    Normalize((0.5,), (0.5,)),
])


def load_model(model: MNISTNet, path: str) -> None:
    """Load model parameters."""
    params = np.load(path, allow_pickle=True).item()
    param_list = list(model.parameters())
    for i, key in enumerate(sorted(params.keys(), key=lambda x: int(x.split('_')[1]))):
        param_list[i].data = params[key].copy()


def predict(image_path: str, model_path: str = "nn/mnist/model.npy") -> int:
    """Predict digit from image."""
    model = MNISTNet()
    load_model(model, model_path)

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).numpy()

    x = Tensor(image, requires_grad=True)
    logits = model.forward(x)

    pred = np.argmax(logits.data)
    print(f"Predicted: {pred}")
    return pred


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Usage: python nn/mnist/predict.py <image_path>")