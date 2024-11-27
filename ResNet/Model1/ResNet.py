import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from ResBlock import *
from DataHandler import *


class ResNet(nn.Module):
    def __init__(self, num_classes=201):
        super(ResNet, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.layer4 = self._make_layer(256, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.train_data = DataHandler('/Users/michallejza/Desktop/Data/IMAGENET/tiny-imagenet-200', kind='train',
                                      transform=transform)
        self.test_data = DataHandler('/Users/michallejza/Desktop/Data/IMAGENET/tiny-imagenet-200', kind='test',
                                     transform=transform)
        self.val_data = DataHandler('/Users/michallejza/Desktop/Data/IMAGENET/tiny-imagenet-200', kind='val',
                                    transform=transform)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def _make_layer(self, out_channels, blocks, stride):
        layers = [ResBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x size is Bx3x64x64
        out = self.conv1(x)  # Bx32x64x64
        out = self.bn1(out)  # Bx32x64x64
        out = F.relu(out)  # Bx32x64x64
        out = self.layer1(out)  # Bx32x64x64
        out = self.layer2(out)  # Bx64x32x32
        out = self.layer3(out)  # Bx128x16x16
        out = self.layer4(out)  # Bx256x8x8
        out = self.avg_pool(out)  # Bx256x1x1
        out = torch.flatten(out, 1)  # Bx256
        out = self.fc(out)  # Bx200
        return out

    def trainModel(self, epochs):
        train_loader = DataLoader(self.train_data, batch_size=64, shuffle=True, drop_last=True)
        for epoch in range(epochs):
            self.train()
            running_loss = 0.0

            for images, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss / len(train_loader):.4f}')
            print('Training completed.')

    def testModel(self):
        pass

    def predictImage(self, index):
        pass
