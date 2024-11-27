import os

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class DataHandler(Dataset):
    def __init__(self, root: str, kind: str, transform=None):
        self.root = root
        self.transform = transform
        self.kind = kind
        self.data = []
        self.labels = []
        self.names = {}
        self.namesMapped = {}
        self.load_folder_names()
        if kind == 'train':
            self.load_train_data()
        elif kind == 'val':
            self.load_val_data()
        elif kind == 'test':
            self.load_test_data()

    def load_folder_names(self):
        index = 0
        with open('mapping.txt', 'r') as file:
            for line in file:
                # Split the line into folder name and description
                folder_name, description = line.strip().split('\t')
                self.names[folder_name] = description
                self.namesMapped[folder_name] = index
                index += 1

    def load_train_data(self):
        folders = os.listdir(os.path.join(self.root, 'train'))
        for folder in folders:
            if folder != ".DS_Store":
                images = os.listdir(os.path.join(self.root, "train", folder, "images"))
                for image in images:
                    imagePath = os.path.join(self.root, "train", folder, "images", image)
                    self.data.append(imagePath)
                    self.labels.append(folder)

    def load_test_data(self):
        images = os.listdir(os.path.join(self.root, "test", "images"))
        for image in images:
            imagePath = os.path.join(self.root, "test", "images", image)
            self.data.append(imagePath)
            self.labels.append(-1)

    def load_val_data(self):
        val_dir = os.path.join(self.root, "val", "images")
        with open(os.path.join(self.root, "val", "val_annotations.txt"), "r") as f:
            annotations = {line.split("\t")[0]: line.split("\t")[1] for line in f}
            for img_name, folder in annotations.items():
                img_path = os.path.join(val_dir, img_name)
                self.data.append(img_path)
                self.labels.append(folder)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]
        label = self.namesMapped[self.labels[index]]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def displayImage(self, index):
        img_path = self.data[index]
        label = self.names[self.labels[index]]
        image = Image.open(img_path).convert('RGB')
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.axis("off")  # Hide axis
        plt.show()
