import pickle
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from . import get_dataset_path, SLASH


class Cifar10Dataset(Dataset):
    def __init__(self, train: bool = False, test: bool = False, transform=None) -> None:
        if train == test:
            raise ValueError('Error while choosing CIFAR10 dataset type: train and test values are the same')

        self.__path = get_dataset_path('CIFAR10') + SLASH
        self.__classes = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        self.__transform = transform
        if train:
            self.__images, self.__labels = self.__load_train_data(self.__path)
        else:
            self.__images, self.__labels = self.__load_test_data(self.__path)

    def images_shape(self) -> tuple:
        """
        :return: Shape of images tensor
        """
        return self.__images.shape

    def num_classes(self) -> int:
        """
        :return: Number of classes
        """
        return len(self.__classes)

    def __str__(self) -> str:
        info = 'Cifar-10 Dataset\n'
        info += f'Tensor images type: {type(self.__images)}\n'
        info += f'Number of images: {self.__images.shape[0]}\n'
        info += f'Number of channels {self.__images.shape[1]}\n'
        info += f'Shape of images: {self.__images.shape[2]} x {self.__images.shape[3]}\n'
        return info

    def __len__(self) -> int:
        """
        :return: Number of samples
        """
        return self.__images.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        """
        :param idx: index of the array of images
        :return: images and labels at given index
        """
        image = self.__images[idx]
        label = self.__labels[idx]
        if self.__transform is not None:
            image = self.__transform(image)
        return image, label

    @staticmethod
    def __load_train_data(dir_path: str) -> tuple:
        """
        Description of load_train_data() to be made
        :return:
        """
        images, labels = [], []
        for i in range(1, 6):
            file_path = dir_path + f"data_batch_{i}"
            batch_data = Cifar10Dataset.__load_pickle_file(file_path)
            images.append(batch_data['data'])
            labels.extend(batch_data['labels'])
        images = np.vstack(images).reshape(-1, 3, 32, 32)
        return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def __load_test_data(dir_path: str) -> tuple:
        """
        Description of load_test_data() to be made
        :return: test images and labels
        """
        filepath = dir_path + 'test_batch'
        batch_data = Cifar10Dataset.__load_pickle_file(filepath)
        images = batch_data['data'].reshape(-1, 3, 32, 32)
        labels = batch_data['labels']
        return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def __load_pickle_file(filepath: str) -> dict:
        """
        Description of load_pickle_file() to be made
        :param filepath: path of the pickle file with batch.
        :return: Dictionary with images and labels
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        data = {key.decode('utf=8'): value for key, value in data.items()}
        return data

    def plot_eight_images(self, random: bool = False) -> None:
        """
        Description of plot_eight_images() to be made
        :param random: If true, randomly choose 8 images to plot, if not, plot first 8 images
        :return: None
        """
        plt.figure(figsize=(15, 10))
        indexes = np.array([i for i in range(8)])
        if random:
            indexes = np.random.randint(0, len(self.__labels), size=8)
        for i in range(len(indexes)):
            plt.subplot(2, 4, i + 1)
            plt.imshow(self.__images[indexes[i]].permute(1, 2, 0).numpy() / 255.0)
            plt.title(self.__classes[self.__labels[indexes[i]]])
        plt.tight_layout()
        plt.show()

    def plot_image(self, idx: int = 0) -> None:
        """
        Description of plot_image() to be made
        :param idx: index of the image to be plotted
        :return: None
        """
        plt.imshow(self.__images[idx] / 255.0)
        plt.title(self.__classes[self.__labels[idx]])
        plt.show()
