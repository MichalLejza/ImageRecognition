from abc import ABC, abstractmethod
from torch.utils.data import Dataset, TensorDataset, DataLoader


class CustomDataset(Dataset, ABC):
    def __init__(self, train: bool = False, test: bool = False, transform=None) -> None:
        if train == test:  # We cant have both train and test data. We must choose
            raise ValueError('Error while choosing dataset type: train and test values are the same')
        self._transform = transform  # only trnasform on images, used in __getitem__
        self._path: str = ""  # path to dataset given dataset
        self._images = None  # list of images, either a tensor or a list of paths to images
        self._labels = None  # tensor of labels
        self._classes: list = []  # list of classes, only names of classes
        self._classes_index: dict = {}  # dict of classes and their index
        self._index_classes: dict = {}  # dict of index and classes

    def __len__(self) -> int:
        # function to return length of dataset (number of images)
        return self._images.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        # function to return image and label, initial function is for datasets where we hold images as tensor
        # image as pytorch tensor
        image = self._images[idx]
        # label
        label = self._labels[idx]
        # apply transofrm on image
        if self._transform is not None:
            image = self._transform(image)
        return image, label

    def num_classes(self) -> int:
        # function to return number of classes
        return len(self._classes)

    def get_data_loader(self, batch_size: int = 64, shuffle: bool = False):
        dataset = TensorDataset(self._images, self._labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return loader

    @abstractmethod
    def make_classes(self) -> None:
        pass

    @abstractmethod
    def make_classes_index(self) -> None:
        pass

    @abstractmethod
    def make_index_classes(self) -> None:
        pass

    @abstractmethod
    def images_shape(self) -> tuple:
        pass

    @abstractmethod
    def _load_train_data(self) -> None:
        pass

    @abstractmethod
    def _load_val_data(self) -> None:
        pass

    @abstractmethod
    def plot(self, idx: int) -> None:
        pass

    @abstractmethod
    def plot_eight_images(self, random: bool = False) -> None:
        pass
