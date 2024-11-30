import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
from . import get_dataset_path, OS_NAME, SLASH


class Cifar10Dataset(Dataset):
    """
    Description of Cifar 10 DataBase to be made.
    """

    def __init__(self, train: bool = False, test: bool = False, transform=None) -> None:
        if train == test:
            raise ValueError('Error while choosing CIFAR10 dataset type: train and test values are the same')

        self.path = get_dataset_path('CIFAR10') + SLASH
        self.transform = transform
        if train:
            self.images, self.labels = self._load_train_data(self.path)
        else:
            self.images, self.labels = self._load_test_data(self.path)

    def __size__(self) -> None:
        """
        :return:
        """
        return self.images.shape

    def __num__classes(self) -> int:
        """
        :return:
        """
        pass

    def __len__(self) -> int:
        """
        :return:
        """
        return len(self.labels)

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

    @staticmethod
    def _load_train_data(dir_path: str) -> tuple:
        """

        :return:
        """
        images, labels = [], []
        for i in range(1, 6):
            file_path = dir_path + f"data_batch_{i}"
            batch_data = Cifar10Dataset._load_pickle_file(file_path)
            images.append(batch_data['data'])
            labels.extend(batch_data['labels'])
        images = np.vstack(images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def _load_test_data(dir_path: str) -> tuple:
        """

        :return:
        """
        filepath = dir_path + 'test_batch'
        batch_data = Cifar10Dataset._load_pickle_file(filepath)
        images = batch_data['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = batch_data['labels']
        return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def _load_pickle_file(filepath: str) -> dict:
        """

        :param filepath:
        :return:
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        data = {key.decode('utf=8'): value for key, value in data.items()}
        return data

    def plotEightImages(self, random: bool = False) -> None:
        """

        :param random:
        :return:
        """
        pass

    def plotImage(self) -> None:
        """

        :return:
        """
        pass
