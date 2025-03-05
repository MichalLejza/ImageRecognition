import os
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tqdm import tqdm
from DataHandlers import *


class SVHNDataset(Dataset):
    def __init__(self, transform=None, train=True, val=False, test=False):
        self.__train = train
        self.__val = val
        self.__test = test
        self.__transform = transform

