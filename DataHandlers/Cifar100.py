import os
import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt

from DataHandlers import get_dataset_path, SLASH
from DataHandlers.Dataset import CustomDataset


class Cifar100Dataset(CustomDataset):
    def __init__(self, train=False, test=False, transform=None):
        super().__init__(train, test, transform)
        self._path = get_dataset_path('CIFAR100')
        self.make_classes()
        self.make_classes_index()
        self.make_index_classes()
        if train:
            self._load_train_data()
        if test:
            self._load_val_data()

    def make_classes(self) -> None:
        meta_path = os.path.join(self._path, "meta")
        with open(meta_path, "rb") as f:
            meta_dict = pickle.load(f, encoding="bytes")
        self._classes =  [name.decode("utf-8") for name in meta_dict[b"fine_label_names"]]

    def images_shape(self) -> tuple:
        return self._images.shape

    def _load_val_data(self) -> None:
        filepath = self._path + SLASH + 'test'
        with open(filepath, "rb") as f:
            batch = pickle.load(f, encoding="bytes")

        self._images = torch.tensor(batch[b"data"].reshape(-1, 3, 32, 32), dtype=torch.float32)
        self._labels = torch.tensor(batch[b"fine_labels"], dtype=torch.long)

    def _load_train_data(self) -> None:
        filepath = self._path + SLASH + 'train'
        with open(filepath, "rb") as f:
            batch = pickle.load(f, encoding="bytes")

        self._images = torch.tensor(batch[b"data"].reshape(-1, 3, 32, 32), dtype=torch.float32)
        self._labels = torch.tensor(batch[b"fine_labels"], dtype=torch.long)

    @staticmethod
    def _load_pickle_file(filepath: str) -> dict:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        data = {key.decode('utf=8'): value for key, value in data.items()}
        return data

    def plot(self, idx: int) -> None:
        image = self._images[idx]
        plt.imshow(image.permute(1, 2, 0).numpy() / 255.0)
        plt.title(self._classes[self._labels[idx]])
        plt.axis('off')
        plt.show()

    def plot_eight_images(self, random: bool = False) -> None:
        plt.figure(figsize=(15, 10))
        indexes = np.array([i for i in range(8)])
        if random:
            indexes = np.random.randint(0, len(self._labels), size=8)

        for i in range(len(indexes)):
            plt.subplot(2, 4, i + 1)
            plt.imshow(self._images[indexes[i]].permute(1, 2, 0).numpy() / 255.0)
            plt.title(self._classes[self._labels[indexes[i]]])
            plt.axis('off')
        plt.tight_layout()
        plt.show()
