import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from collections import Counter
from tqdm import tqdm
from DataHandlers import get_dataset_path, SLASH


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

    def get_data_loader(self, batch_size: int = 64, shuffle: bool = False):
        dataset = TensorDataset(self.__images, self.__labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    def __str__(self) -> str:
        """
        Description of __str__ to be made
        :return:
        """
        info = f'Mnist Dataset type: {self.__kind}\n'
        info += f'Tensor images type: {type(self.__images)}\n'
        info += f'Number of images: {self.__images.shape[0]}\n'
        info += f'Number of channels {self.__images.shape[1]}\n'
        info += f'Shape of images: {self.__images.shape[2]} x {self.__images.shape[3]}\n'
        return info

    def __len__(self) -> int:
        """
        :return: to be made
        """
        return self.__images.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        """
        Description pf __getitem__ to be made
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
        Description of plot_eight_images to be made
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
        Description of plot_image to be made
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
        Description of return_label to be made
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
        Description of __load_images to be made
        :param filepath:
        :return:
        """
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
    def __load_labels(filepath: str):
        """
        Description on __load_labels to be made
        :param filepath:
        :return:
        """
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
