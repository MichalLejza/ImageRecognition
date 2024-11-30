import os

from PIL import Image
from torch.utils.data import Dataset


class TinyImageNetDataset(Dataset):
    """

    """
    def __init__(self, root_dir, split, transform=None):
        """

        :param root_dir:
        :param split:
        :param transform:
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.data = []
        self.labels = []
        self.load_data()

    def load_data(self):
        # Load train data
        if self.split == "train":
            classes = os.listdir(os.path.join(self.root_dir, "train"))
            for class_id in classes:
                if class_id == ".DS_Store":  # Ignore .DS_Store file
                    continue
                class_dir = os.path.join(self.root_dir, "train", class_id, "images")
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    self.data.append(img_path)
                    self.labels.append(class_id)

        elif self.split == "val":
            val_dir = os.path.join(self.root_dir, "val", "images")
            with open(os.path.join(self.root_dir, "val", "val_annotations.txt"), "r") as f:
                annotations = {line.split("\t")[0]: line.split("\t")[1] for line in f}
            for img_name, class_id in annotations.items():
                img_path = os.path.join(val_dir, img_name)
                self.data.append(img_path)
                self.labels.append(class_id)

        elif self.split == "test":
            test_dir = os.path.join(self.root_dir, "test", "images")
            for img_name in os.listdir(test_dir):
                img_path = os.path.join(test_dir, img_name)
                self.data.append(img_path)
                self.labels.append(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label
