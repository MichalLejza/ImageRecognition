from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MnistDataset(Dataset):
    def __init__(self, imagePath: str, labelPath: str, transform=None):
        self.imagePath = imagePath
        self.labelPath = labelPath
        self.images = load_mnist_images(imagePath)
        self.labels = load_mnist_labels(labelPath)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def showImageDistribution(self):
        labelList = list(Counter(self.labels.numpy()).keys())
        labelCount = list(Counter(self.labels.numpy()).values())
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labelList, labelCount, color='skyblue')
        plt.xticks(range(10))  # Set x-ticks to label values
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title('Distribution of Labels')
        plt.grid(axis='y')
        for bar in bars:
            yval = bar.get_height()  # Get the height of the bar
            plt.text(bar.get_x() + bar.get_width() / 2, yval, str(yval), ha='center', va='bottom', fontsize=10)
        plt.show()

    def printSize(self) -> None:
        print("Dataset sizes: ")
        print(self.images.shape)
        print(self.labels.shape)

    def printImage(self, idx: int) -> None:
        print("Label: ", self.labels[idx].numpy())
        for i in range(self.images.shape[2]):
            for j in range(self.images.shape[3]):
                if self.images[idx][0][i][j] > 0:
                    print("\033[31m", end="")
                print(format(self.images[idx][0][i][j].numpy(), ".1f"), end='')
                print("\033[0m ", end="")
            print()
        print()


def load_mnist_images(filepath):
    with open(filepath, 'rb') as f:
        f.read(16)  # Skip the header
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(-1, 28, 28).astype(np.float32) / 255.0  # Normalize
    return torch.tensor(images).unsqueeze(1)  # Add channel dimension for PyTorch


def load_mnist_labels(filepath):
    with open(filepath, 'rb') as f:
        f.read(8)  # Skip the header
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return torch.tensor(labels, dtype=torch.long)
