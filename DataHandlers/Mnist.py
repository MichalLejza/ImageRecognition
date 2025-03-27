from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from DataHandlers import get_dataset_path, SLASH
from DataHandlers.Dataset import CustomDataset


class MnistDataset(CustomDataset):
    def __init__(self, kind: str = 'MNIST', train: bool = False, test: bool = False, transform=None) -> None:
        super().__init__(train, test, transform)
        self._kind = kind
        self._path = get_dataset_path('EMNIST') + SLASH + kind + SLASH
        self.make_classes()
        self.make_classes_index()
        self.make_index_classes()
        if train:
            self._load_train_data()
        if test:
            self._load_val_data()

    def make_classes(self) -> None:
        txt_file = next(Path(self._path).glob("*.txt"), None)
        if txt_file:
            with txt_file.open("r", encoding="utf-8") as file:
                for line in file:
                    parts = line.strip().split(" ")
                    if len(parts) == 2:
                        self._classes.append(chr(int(parts[1])))
        else:
            print("Nie znaleziono pliku .txt w folderze.")

    def images_shape(self) -> tuple:
        return self._images.shape

    def _load_train_data(self):
        self._images = self._load_images(self._path + ('train-images' if self._train else 'test-images'))
        self._labels = self._load_labels(self._path + ('train-labels' if self._train else 'test-labels'))

    def _load_val_data(self):
        self._images = self._load_images(self._path + ('train-images' if self._train else 'test-images'))
        self._labels = self._load_labels(self._path + ('train-labels' if self._train else 'test-labels'))

    @staticmethod
    def _load_images(filepath: str):
        with open(filepath, 'rb') as f:
            f.read(16)
            f.seek(0, 2)
            file_size = f.tell()
            num_images = (file_size - 16) // (28 * 28)
            f.seek(16)
            images = []
            for _ in tqdm(range(num_images), desc="Loading MNIST images"):
                buffer = f.read(28 * 28)
                image = np.frombuffer(buffer, dtype=np.uint8).reshape(28, 28).astype(np.float32)
                images.append(image)
        images = np.stack(images)
        return torch.tensor(images).unsqueeze(1)

    @staticmethod
    def _load_labels(filepath: str):
        with open(filepath, 'rb') as f:
            f.read(8)
            f.seek(0, 2)
            file_size = f.tell()
            num_labels = file_size - 8
            f.seek(8)
            labels = []
            for _ in tqdm(range(num_labels), desc=f"Loading MNIST labels"):
                label = np.frombuffer(f.read(1), dtype=np.uint8)[0]
                labels.append(label)
        return torch.tensor(labels, dtype=torch.long)

    def plot_eight_images(self, random: bool = False) -> None:
        plt.figure(figsize=(15, 10))
        indexes = np.array([i for i in range(8)])
        if random:
            indexes = np.random.randint(0, len(self._labels), size=8)
        for i in range(len(indexes)):
            plt.subplot(2, 4, i + 1)
            plt.imshow(self._images[indexes[i]][0].t(), interpolation="nearest", cmap="Greys")
            plt.title("Label: " + self._index_classes[self._labels[indexes[i]].item()])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot(self, index: int = 0) -> None:
        plt.figure(figsize=(10, 10))
        plt.imshow(self._images[index][0].t(), interpolation="nearest", cmap="Greys")
        plt.title("Label: " + self._index_classes[self._labels[index].item()])
        plt.axis('off')
        plt.show()
