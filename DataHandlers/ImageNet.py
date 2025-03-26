import os
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm
from DataHandlers import *


class ImageNetDataset(Dataset):
    def __init__(self, train=False, test=False, transform=None):
        pass

