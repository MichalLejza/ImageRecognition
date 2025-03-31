import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # (224x224 -> 55x55)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (55x55 -> 27x27)

            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),  # (27x27 -> 27x27)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # (27x27 -> 13x13)

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),  # (13x13 -> 13x13)
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),  # (13x13 -> 13x13)
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),  # (13x13 -> 13x13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # (13x13 -> 6x6)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)  # Warstwa wyjściowa
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Spłaszczenie do wektora
        x = self.classifier(x)
        return x