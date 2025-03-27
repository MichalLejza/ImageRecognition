import pickle

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from DataHandlers import get_dataset_path, SLASH
from DataHandlers.Dataset import CustomDataset


class Cifar10Dataset(CustomDataset):
    def __init__(self, train: bool = False, test: bool = False, transform=None) -> None:
        super().__init__(train, test, transform)
        self._path = get_dataset_path('CIFAR10') + SLASH
        self.make_classes()
        self.make_classes_index()
        self.make_index_classes()
        if train:
            self._load_train_data()
        if test:
            self._load_val_data()

    def make_classes(self) -> None:
        self._classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    def images_shape(self) -> tuple:
        return self._images.shape

    def _load_train_data(self) -> None:
        images, labels = [], []
        for i in tqdm(range(1, 6), desc="Loading CIFAR-10 Train Data"):
            file_path = self._path + f"data_batch_{i}"
            batch_data = Cifar10Dataset._load_pickle_file(file_path)
            images.append(batch_data['data'])
            labels.extend(batch_data['labels'])
        images = np.vstack(images).reshape(-1, 3, 32, 32)
        self._images = torch.tensor(images, dtype=torch.float32)
        self._labels = torch.tensor(labels, dtype=torch.long)

    def _load_val_data(self) -> None:
        filepath = self._path + 'test_batch'
        batch_data = Cifar10Dataset._load_pickle_file(filepath)
        images = batch_data['data'].reshape(-1, 3, 32, 32)
        labels = batch_data['labels']
        self._images = torch.tensor(images, dtype=torch.float32)
        self._labels = torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def _load_pickle_file(filepath: str) -> dict:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        data = {key.decode('utf=8'): value for key, value in data.items()}
        return data

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

    def plot(self, idx: int = 0) -> None:
        image = self._images[idx]
        plt.imshow(image.permute(1, 2, 0).numpy() / 255.0)
        plt.title(self._classes[self._labels[idx]])
        plt.axis('off')
        plt.show()
