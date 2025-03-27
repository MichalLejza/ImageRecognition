import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

from DataHandlers import *
from DataHandlers.Dataset import CustomDataset


class ImageNetDataset(CustomDataset):
    def __init__(self, train=False, test=False, transform=None):
        super().__init__(train=train, test=test, transform=transform)
        self._images = []
        self._path = get_dataset_path('IMAGENET')
        self._folder_meaning: dict = {}
        self.make_classes()
        self.make_classes_index()
        self.make_index_classes()

        if train:
            self._load_train_data()
        if test:
            self._load_val_data()

    def images_shape(self) -> tuple:
        return len(self._images), 3, 224, 224

    def make_classes(self) -> None:
        with open(self._path + SLASH + 'folder_names.txt', 'r') as f:
            for line in f:
                folder, number, description = line.strip().split(' ')
                self._folder_meaning[folder] = description
                self._classes.append(description)

    def __getitem__(self, item):
        image_path = self._images[item]
        label = self._labels[item]
        image = Image.open(image_path).convert('RGB')
        if self._transform is not None:
            image = self._transform(image)
        return image, label

    def _load_train_data(self) -> None:
        labels: list = []
        for name in tqdm(os.listdir(self._path + SLASH + 'train'), desc='Loading Image Net Train Data'):
            folderPath = self._path + SLASH + 'train' + SLASH + name
            if os.path.isdir(folderPath):
                for img in os.listdir(folderPath):
                    imgPath = os.path.join(folderPath, img)
                    self._images.append(imgPath)
                    index = self._classes_index[self._folder_meaning[name]]
                    labels.append(index)
        self._labels = torch.tensor(labels, dtype=torch.long)

    def _load_val_data(self) -> None:
        labels: list = []
        for name in tqdm(os.listdir(self._path + SLASH + 'val'), desc='Loading Image Net Validation Data'):
            imgPath = os.path.join(self._path + SLASH + 'val', name)
            self._images.append(imgPath)
        with open(self._path + SLASH + 'val_labels.txt', 'r') as f:
            for line in f:
                labels.append(self._classes_index[int(line.strip()) - 1])
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
            image, label = self.__getitem__(indexes[i])
            plt.imshow(image)
            plt.title(f"Label: {self._index_classes[label.item()]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
