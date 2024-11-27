from collections import Counter
from . import *
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class MnistDataset(Dataset):
    """
    Hej, to jest jakaś tam zmiana
    """
    def __init__(self, kind: str='Classic', train: bool=False, test: bool=False, transform=None) -> None:
        if train == test:
            raise ValueError('Error while choosing dataset type: train and test values are the same')
        slash: str = '/' if OS_NAME == 'Darwin' else '\\'
        self.imagePath = get_dataset_path('EMNIST') + slash + kind + slash + ('train-images' if train else 'test-images')
        self.labelPath = get_dataset_path('EMNIST') + slash + kind + slash + ('train-labels' if train else 'test-labels')
        self.images = self._load_images(self.imagePath)
        self.labels = self._load_labels(self.labelPath)
        self.transform = transform
        self.xdd = 'XDDD'


    def __len__(self) -> int:
        """
        :return:
        """
        return len(self.labels)


    def __size__(self) -> tuple:
        """
        :return:
        """
        return self.images.shape


    def __getitem__(self, idx: int) -> tuple:
        """

        :param idx:
        :return:
        """
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


    def __num_classes__(self) -> int:
        """

        :return:
        """
        return len(list(Counter(self.labels.numpy()).keys()))


    def plotClassDist(self) -> None:
        """

        :return: None
        """
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


    def plotEightImages(self, random: bool=False, transform: bool=False) -> None:
        """

        :param transform:
        :param random:
        :return:
        """
        plt.figure(figsize=(15, 10))
        indexes = np.array([i for i in range(8)])
        if random:
            indexes = np.random.randint(0, len(self.labels), size=8)
        for i in range(len(indexes)):
            plt.subplot(2, 4, i + 1)
            plt.imshow(self.images[indexes[i]][0].t(), interpolation="nearest", cmap="Greys")
            plt.title("Label: " + self.returnLabel(self.labels[indexes[i]]))
            plt.axis('off')
        plt.tight_layout()
        plt.show()


    def plotImage(self, index: int=0) -> None:
        """

        :param index:
        :return:
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(self.images[index][0].t(), interpolation="nearest", cmap="Greys")
        plt.title("Label: " + self.returnLabel(self.labels[index]))
        plt.axis('off')
        plt.show()


    @staticmethod
    def returnLabel(number) -> str:
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
    def _load_images(filepath: str):
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
    def _load_labels(filepath: str):
        """

        :param filepath:
        :return:
        """
        with open(filepath, 'rb') as f:
            f.read(8)
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return torch.tensor(labels, dtype=torch.long)
