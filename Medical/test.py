import h5py
import numpy as np
from matplotlib import pyplot as plt

# Path to the HDF5 file
file_path = '/Users/michallejza/Downloads/archive/BraTS2020_training_data/content/data/volume_1_slice_70.h5'

with h5py.File(file_path, 'r') as f:
    # List all datasets in the file
    print("Datasets in the file:")
    for key in f.keys():
        print(key)

    # Load a specific dataset (modify the dataset name as necessary)
    images = f['image'][:]  # Example key, replace with actual dataset key
    labels = f['mask'][:]  # Example key for labels, replace as necessary
    # Display the first image (modify the index if needed)
    first_image = images
    min_value = np.min(first_image)
    max_value = np.max(first_image)

    normalized_image = (first_image - min_value) / (max_value - min_value)

    # Display the normalized image
    plt.imshow(normalized_image)
    # Display the image
    plt.axis('off')  # Hide axes
    plt.title('First Image from BraTS Dataset')
    plt.show()
