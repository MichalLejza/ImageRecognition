import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, data_set_size, num_classes: int):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(data_set_size[1], 64, kernel_size=3, stride=1, padding=1),  # Dostosowane do 32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 *  (data_set_size[2] // (2 ** 3)) * (data_set_size[3] // (2 ** 3)), 4096),  # Zmniejszono liczbę neuronów
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2048),  # Kolejna redukcja
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),  # Liczba klas dla CIFAR-10
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x