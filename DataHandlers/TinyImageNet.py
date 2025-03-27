import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
from DataHandlers import *
from DataHandlers.Dataset import CustomDataset


class TinyImageNetDataset(CustomDataset):
    def __init__(self, train: bool = False, test: bool = False, transform=None):
        super().__init__(train, test, transform)
        self._images = []
        self._path = get_dataset_path('TINYIMAGENET')
        self._folder_meaning: dict = {}
        self.make_classes()
        self.make_classes_index()
        self.make_index_classes()

        if train:
            self._load_train_data()
        if test:
            self._load_val_data()

    def make_classes(self) -> None:
        folder_meaning: dict = {}
        with open(self._path + SLASH + 'words.txt', 'r') as f:
            for line in f:
                folder_name, description = line.strip().split('\t', 1)
                description = description.split(',')[0] if ',' in description else description
                folder_meaning[folder_name] = description
        for name in os.listdir(self._path + SLASH + 'train'):
            if name in folder_meaning:
                self._folder_meaning[name] = folder_meaning[name]
                self._classes.append(folder_meaning[name])

    def images_shape(self) -> tuple:
        return len(self._images), 3, 32, 32

    def num_classes(self) -> int:
        return self._classes.__len__()

    def __getitem__(self, idx: int) -> tuple:
        image_path = self._images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self._labels[idx]
        if self._transform is not None:
            image = self._transform(image)
        return image, label

    def _load_train_data(self) -> None:
        path = get_dataset_path('TINYIMAGENET') + SLASH + 'train'
        labels: list = []
        for name in tqdm(os.listdir(path), desc='Loading TinyImageNet Train Data'):
            folderPath = path + SLASH + name + SLASH + 'images'
            if os.path.isdir(folderPath):
                for img in os.listdir(folderPath):
                    imgPath = os.path.join(folderPath, img)
                    self._images.append(imgPath)
                    index = self._classes_index[self._folder_meaning[name]]
                    labels.append(index)
        self._labels = torch.tensor(labels, dtype=torch.long)

    def _load_val_data(self) -> None:
        path = get_dataset_path('TINYIMAGENET') + SLASH + 'val' + SLASH
        labels: list = []
        with open(path + 'val_annotations.txt', 'r') as f:
            for line in tqdm(f, desc='Loading TinyImageNet Validation Data'):
                imgPath = path + SLASH + 'images' + SLASH + line.split("\t")[0]
                self._images.append(imgPath)
                index = self._classes_index[self._folder_meaning[line.split("\t")[1]]]
                labels.append(torch.tensor(index))
        self._labels = torch.tensor(labels, dtype=torch.long)

    def plot_eight_images(self, random: bool = False):
        plt.figure(figsize=(15, 10))
        indexes = np.array([i for i in range(8)])
        if random:
            indexes = np.random.randint(0, len(self._images), size=8)
        for i in range(len(indexes)):
            plt.subplot(2, 4, i + 1)
            image = Image.open(self._images[indexes[i]]).convert('RGB')
            plt.imshow(image)
            plt.title(f"Label: {self._index_classes[self._labels[indexes[i]].item()]}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot(self, idx: int) -> None:
        image_path = self._images[idx]
        image = Image.open(image_path).convert('RGB')
        plt.imshow(image)
        plt.title(self._index_classes[self._labels[idx].item()])
        plt.axis('off')
        plt.show()
