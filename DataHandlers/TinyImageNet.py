import os
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm
from . import *


class TinyImageNetDataset(Dataset):
    def __init__(self, train: bool = False, test: bool = False, val: bool = False, transform=None):
        if (train + test + val) != 1:
            raise ValueError('Error, when choosing Tiny Image Net type: choose train or test or val')

        self.__transform = transform
        self.__folderMeaning: dict = self.__get_folder_meaning()
        self.__namesMapped: dict = self.__map_names(self.__folderMeaning)

        if train:
            self.__images, self.__labels = self.__load_train(transform=transform)
        if test:
            self.__images, self.__labels = self.__load_test(transform=transform)
        if val:
            self.__images, self.__labels = self.__load_val(transform=transform)

    def images_shape(self) -> tuple:
        """
        :return: Shape of images tensor
        """
        return self.__images.shape

    def num_classes(self) -> int:
        """
        :return: Number of classes
        """
        return self.__namesMapped.__len__()

    def get_data_loader(self, batch_size: int = 64, shuffle: bool = False):
        dataset = TensorDataset(self.__images, self.__labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    def __str__(self) -> str:
        """
        Description of __str__ to be made
        :return:
        """
        info = f'Tiny Image Net Dataset\n'
        info += f'Tensor images type: {type(self.__images)}\n'
        info += f'Number of images: {self.__images.shape[0]}\n'
        info += f'Number of channels {self.__images.shape[1]}\n'
        info += f'Shape of images: {self.__images.shape[2]} x {self.__images.shape[3]}\n'
        return info

    def __len__(self) -> int:
        """
        description of __len__
        """
        return self.__images.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        """
        Description of __getitem__
        """
        image = self.__images[idx]
        label = self.__labels[idx]
        if self.__transform is not None:
            image = self.__transform(image)
        return image, label

    @staticmethod
    def __map_names(names: dict) -> dict:
        """
        Description of _map_names
        """
        index = 0
        mapped = {}
        for folder in names.keys():
            mapped[folder] = index
            index += 1
        return mapped

    @staticmethod
    def __get_folder_meaning() -> dict:
        """
        Description of _get_folder_meaning
        """
        foldersPath = get_dataset_path('IMAGENET') + SLASH + 'train'
        filePath = get_dataset_path('IMAGENET') + SLASH + 'words.txt'
        folder_descriptions = {}
        folder_meaning = {}
        with open(filePath, 'r') as f:
            for line in f:
                folder_name, description = line.strip().split('\t', 1)
                description = description.split(',')[0] if ',' in description else description
                folder_descriptions[folder_name] = description

        for name in os.listdir(foldersPath):
            if name in folder_descriptions:
                folder_meaning[name] = folder_descriptions[name]
        return folder_meaning

    @staticmethod
    def __load_train(transform) -> tuple:
        """
        Description of _load_train_data
        """
        path = get_dataset_path('IMAGENET') + SLASH + 'train'
        images = []
        labels = []
        for name in tqdm(os.listdir(path), desc='Loading Train Data'):
            folderPath = path + SLASH + name + SLASH + 'images'
            if os.path.isdir(folderPath):
                for img in os.listdir(folderPath):
                    imgPath = os.path.join(folderPath, img)
                    try:
                        image = Image.open(imgPath).convert('RGB')
                        images.append(transform(image))
                        labels.append(name)
                    except Exception as e:
                        print(f"Error loading {img}: {e}")
        return torch.stack(images), labels

    @staticmethod
    def __load_test(transform) -> tuple:
        """
        Description of _load_test_data
        """
        path = get_dataset_path('IMAGENET') + SLASH + 'test' + SLASH + 'images' + SLASH
        images = []
        for img in tqdm(os.listdir(path), desc='Loading Test Data'):
            imgPath = os.path.join(path, img)
            try:
                image = Image.open(imgPath).convert('RGB')
                images.append(transform(image))
            except Exception as e:
                print(f"Error loading {img}: {e}")
        return torch.stack(images), None

    @staticmethod
    def __load_val(transform) -> tuple:
        """
        Description of _load_val_data
        :return:
        """
        path = get_dataset_path('IMAGENET') + SLASH + 'val' + SLASH
        images = []
        labels = []
        with open(path + 'val_annotations.txt', 'r') as f:
            for line in tqdm(f, desc='Loading Validation Data'):
                imgPath = path + SLASH + 'images' + SLASH + line.split("\t")[0]
                try:
                    image = Image.open(imgPath).convert('RGB')
                    images.append(transform(image))
                    labels.append(line.split("\t")[1])
                except Exception as e:
                    print(f"Error loading {imgPath}: {e}")
        return torch.stack(images), labels

    def plot_eight_images(self, random: bool = False):
        """
        Description of plot_eight_images
        """
        plt.figure(figsize=(15, 10))
        indexes = np.array([i for i in range(8)])
        if random:
            indexes = np.random.randint(0, self.__images.shape[0], size=8)
        for i in range(len(indexes)):
            plt.subplot(2, 4, i + 1)
            image = self.__images[indexes[i]].permute(1, 2, 0).numpy()
            plt.imshow(image)
            if self.__labels is not None:
                plt.title(f"Label: {self.__folderMeaning[self.__labels[indexes[i]]]}")
        plt.tight_layout()
        plt.show()

    def plot_image(self, idx: int) -> None:
        """
        Description of plot_image
        """
        image = self.__images[idx]
        image = image.permute(1, 2, 0).numpy()
        plt.imshow(image)
        if self.__labels is not None:
            plt.title(self.__folderMeaning[self.__labels[idx]])
        plt.show()
