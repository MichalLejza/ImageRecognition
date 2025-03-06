import torch
from torch import nn
from torch.optim.adam import Adam
from ResNet import ResNet
from DataHandlers.Cifar10 import Cifar10Dataset


class ModelBuilder:
    def __init__(self, resnet:ResNet):
        self.resnet = resnet
        self.train_data =Cifar10Dataset(train=True)
        self.test_data = Cifar10Dataset(train=False)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.resnet.parameters(), lr=0.001)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
