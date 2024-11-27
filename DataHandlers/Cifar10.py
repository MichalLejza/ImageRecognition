import os
import pickle

import numpy as np


def load_batch(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = batch[b'labels']
    return images, labels


def load_cifar10(folder):
    train_images, train_labels = [], []
    for i in range(1, 6):
        file = os.path.join(folder, f'data_batch_{i}')
        images, labels = load_batch(file)
        train_images.append(images)
        train_labels.extend(labels)
    train_images = np.concatenate(train_images)
    train_labels = np.array(train_labels)

    test_file = os.path.join(folder, 'test_batch')
    test_images, test_labels = load_batch(test_file)
    test_labels = np.array(test_labels)

    return train_images, train_labels, test_images, test_labels


class Cifar10Database:
    def __init__(self, dirPath, transform=None):
        self.dirPath = dirPath
        self.images = None
        self.labels = None
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


