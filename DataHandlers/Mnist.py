from collections import Counter
from . import get_dataset_path, SLASH
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    def __init__(self, kind: str = 'Classic', train: bool = False, test: bool = False, transform=None) -> None:
        if train == test:
            raise ValueError('Error while choosing MNIST dataset type: train and test values are the same')
        self.__kind = kind
        self.__transform = transform
        path = get_dataset_path('EMNIST') + SLASH + kind + SLASH + ('train-' if train else 'test-')
        self.__images = self.__load_images(path + 'images')
        self.__labels = self.__load_labels(path + 'labels')
        self.__classes = len(list(Counter(self.__labels.numpy()).keys()))

    def images_shape(self) -> tuple:
        """
        :return: Shape of images tensor
        """
        return self.__images.shape

    def num_classes(self) -> int:
        """
        :return: Number of classes
        """
        return self.__classes

    def __str__(self) -> str:
        info = f'Mnist Dataset type: {self.__kind}\n'
        info += f'Tensor images type: {type(self.__images)}\n'
        info += f'Number of images: {self.__images.shape[0]}\n'
        info += f'Number of channels {self.__images.shape[1]}\n'
        info += f'Shape of images: {self.__images.shape[2]} x {self.__images.shape[3]}\n'
        return info

    def __len__(self) -> int:
        """
        :return:
        """
        return self.__images.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        """

        :param idx:
        :return:
        """
        image = self.__images[idx]
        label = self.__labels[idx]
        if self.__transform is not None:
            image = self.__transform(image)
        return image, label

    def plot_class_dist(self) -> None:
        """
        Description of plot_class_dist to be made
        """
        labelList = list(Counter(self.__labels.numpy()).keys())
        labelCount = list(Counter(self.__labels.numpy()).values())
        plt.figure(figsize=(10, 6))
        bars = plt.bar(labelList, labelCount, color='skyblue')
        plt.xticks(range(10))
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.title('Distribution of Labels')
        plt.grid(axis='y')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval, str(yval), ha='center', va='bottom', fontsize=10)
        plt.show()

    def plot_eight_images(self, random: bool = False) -> None:
        """

        :param random:
        :return:
        """
        plt.figure(figsize=(15, 10))
        indexes = np.array([i for i in range(8)])
        if random:
            indexes = np.random.randint(0, len(self.__labels), size=8)
        for i in range(len(indexes)):
            plt.subplot(2, 4, i + 1)
            plt.imshow(self.__images[indexes[i]][0].t(), interpolation="nearest", cmap="Greys")
            plt.title("Label: " + self.__return_label(self.__labels[indexes[i]]))
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_image(self, index: int = 0) -> None:
        """

        :param index:
        :return:
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(self.__images[index][0].t(), interpolation="nearest", cmap="Greys")
        plt.title("Label: " + self.__return_label(self.__labels[index]))
        plt.axis('off')
        plt.show()

    @staticmethod
    def __return_label(number) -> str:
        """

        :param number:
        :return:
        """
        number = number.numpy()
        if number < 10:
            return str(number)
        elif number < 36:
            return chr(ord('A') + number - 10)
        else:
            return chr(ord('a') + number - 36)

    @staticmethod
    def __load_images(filepath: str):
        """

        :param filepath:
        :return:
        """
        with open(filepath, 'rb') as f:
            f.read(16)
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(-1, 28, 28).astype(np.float32)
        return torch.tensor(images).unsqueeze(1)

    @staticmethod
    def __load_labels(filepath: str):
        """

        :param filepath:
        :return:
        """
        with open(filepath, 'rb') as f:
            f.read(8)
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return torch.tensor(labels, dtype=torch.long)
