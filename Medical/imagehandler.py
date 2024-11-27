import h5py
import numpy as np
from matplotlib import pyplot as plt
import os
import re


class ImageHandler:
    def __init__(self, dirPath: str):
        self.dirPath = dirPath
        self.images = {}
        self._uploadImages()

    def _uploadImages(self):
        files = os.listdir(self.dirPath)
        for file in files:
            if file.endswith('.h5'):
                match = re.match(r'volume_(\d+)_slice_(\d+)\.h5', file)
                if match:
                    volume_number = match.group(1)
                    if int(volume_number) not in self.images:
                        self.images[int(volume_number)] = []
                    self.images[int(volume_number)].append(os.path.join(self.dirPath, file))

    def displayImage(self, index, slice):
        files = self.images[index]
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Extract slice number from filename
        file = files[slice]
        with h5py.File(file, 'r') as h5_file:
            image_data = h5_file['image'][:]  # Adjust the key based on your .h5 structure
            min_value = np.min(image_data)
            max_value = np.max(image_data)
            normalized = (image_data - min_value) / (max_value - min_value)
        plt.figure()
        plt.imshow(normalized)
        plt.axis('off')  # Hide axes
        plt.show()

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    path = "/Users/michallejza/Desktop/Data/BRATS/BraTS2020_training_data/content/data"
    handler = ImageHandler(path)
    handler.displayImage(2, 70)
