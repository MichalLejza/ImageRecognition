import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from DataHandlers import *
from DataHandlers.Dataset import CustomDataset


class Caltech256Dataset(CustomDataset):
    def __init__(self, train=False, test=False, transform=None):
        super().__init__(train, test, transform)
        self._path = get_dataset_path('CALTECH256')
        self._images = []
        self.make_classes()
        self.make_classes_index()
        self.make_index_classes()
        if train:
            self._load_train_data()
        if test:
            self._load_val_data()

    def make_classes(self) -> None:
        dir_paths = os.listdir(self._path + SLASH + 'all')
        for dir_path in dir_paths:
            dir_name = dir_path.split('.')[-1]
            self._classes.append(dir_name)

    def images_shape(self) -> tuple:
        return len(self._labels), 3, 224, 224

    def __getitem__(self, idx: int) -> tuple:
        image_path = self._images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self._labels[idx]
        if self._transform is not None:
            image = self._transform(image)
        return image, label

    def _load_train_data(self) -> None:
        dir_paths = os.listdir(self._path + SLASH + 'train')
        labels: list = []
        for dir_path in tqdm(dir_paths, desc='Loading Caltech256 Train Data'):
            dir_name = dir_path.split('.')[-1]
            for file in os.listdir(self._path + SLASH + 'train' + SLASH + dir_path):
                image_path = self._path + SLASH + 'train' + SLASH + dir_path + SLASH + file
                self._images.append(image_path)
                labels.append(self._classes_index[dir_name])
        self._labels = torch.tensor(labels, dtype=torch.long)


    def _load_val_data(self) -> None:
        dir_paths = os.listdir(self._path + SLASH + 'val')
        labels: list = []
        for dir_path in tqdm(dir_paths, desc='Loading Caltech256 Validation Data'):
            dir_name = dir_path.split('.')[-1]
            for file in os.listdir(self._path + SLASH + 'val' + SLASH + dir_path):
                image_path = self._path + SLASH + 'val' + SLASH + dir_path + SLASH + file
                self._images.append(image_path)
                labels.append(self._classes_index[dir_name])
        self._labels = torch.tensor(labels, dtype=torch.long)


    def plot(self, idx: int) -> None:
        image, label = self.__getitem__(idx)
        plt.imshow(image)
        plt.title(self._index_classes[label.item()])
        plt.axis('off')
        plt.show()

    def plot_eight_images(self, random: bool = False) -> None:
        plt.figure(figsize=(15, 10))
        indexes = np.array([i for i in range(8)])
        if random:
            indexes = np.random.randint(0, len(self._labels), size=8)
        for i in range(len(indexes)):
            plt.subplot(2, 4, i + 1)
            image, label = self.__getitem__(indexes[i].item())
            plt.imshow(image)
            plt.title(f"Label: {self._index_classes[label.item()]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
