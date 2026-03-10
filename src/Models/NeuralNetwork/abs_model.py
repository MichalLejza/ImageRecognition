from abc import ABC, abstractmethod
import torch.nn as nn


class AbsModel(nn.Module, ABC):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def print_info(self):
        pass
