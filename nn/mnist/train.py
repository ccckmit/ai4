"""MNIST training using ai4/nn framework."""

from __future__ import annotations

import os
import numpy as np
from nn import Tensor, Module, Linear, Conv2d, MaxPool2d, Flatten, ReLU, Adam
from nn.datasets import datasets, DataLoader, Compose, RandomRotation, ToTensor, Normalize


class MNISTNet(Module):
    """CNN classifier: 2 conv layers + 2 fully connected layers."""

    def __init__(self) -> None:
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=3, bias=True)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, bias=True)
        self.pool1 = MaxPool2d(kernel_size=2)
        self.pool2 = MaxPool2d(kernel_size=2)
        self.flatten = Flatten()
        self.fc1 = Linear(in_features=64 * 5 * 5, out_features=128, bias=True)
        self.fc2 = Linear(in_features=128, out_features=10, bias=True)
        self.relu = ReLU()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def train() -> None:
    """Train MNIST model using ai4/nn."""
    transform = Compose([
        RandomRotation(10),
        ToTensor(),
        Normalize((0.5,), (0.5,)),
    ])

    trainSet = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    trainLoader = DataLoader(trainSet, batch_size=64, shuffle=True)

    model = MNISTNet()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        total = 0
        correct = 0
        for images, labels in trainLoader:
            images_np = images.numpy()
            labels_np = labels.numpy()

            x = Tensor(images_np, requires_grad=True)
            logits = model(x)

            logits_data = logits.data
            max_logits = np.max(logits_data, axis=-1, keepdims=True)
            exps = np.exp(logits_data - max_logits)
            probs = exps / np.sum(exps, axis=-1, keepdims=True)

            batch_size = labels_np.shape[0]
            loss = 0.0
            for i in range(batch_size):
                loss -= np.log(probs[i, labels_np[i]] + 1e-10)
            loss /= batch_size

            loss_tensor = Tensor(loss, (logits,), True)

            optimizer.zero_grad()
            loss_tensor.backward()

            optimizer.step()

            predictions = np.argmax(logits_data, axis=1)
            total += labels_np.shape[0]
            correct += np.sum(predictions == labels_np)

        print(f"Epoch {epoch+1}: {100*correct/total:.2f}%")

    os.makedirs("nn/mnist", exist_ok=True)
    save_model(model, "nn/mnist/model.npy")
    print("Model saved to nn/mnist/model.npy")


def save_model(model: MNISTNet, path: str) -> None:
    """Save model parameters."""
    params = {}
    for i, p in enumerate(model.parameters()):
        params[f"param_{i}"] = p.data.copy()
    np.save(path, params)


if __name__ == "__main__":
    train()