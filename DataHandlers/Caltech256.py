import os
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm
from DataHandlers import *


class Caltech256Dataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

