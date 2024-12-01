import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from DataHandlers.Cifar10 import Cifar10Dataset


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Dostosowane do 32x32
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
            nn.Linear(256 * 4 * 4, 4096),  # Zmniejszono liczbę neuronów
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


class AlexNetTrainer:
    def __init__(self, batch_size: int  = 64, transform = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.model = AlexNet().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_data = Cifar10Dataset(train=True, transform=transform)
        self.test_data = Cifar10Dataset(test=True, transform=transform)

    def train(self, epochs: int = 10):
        self.model.train()
        train_loader = self.train_data.get_data_loader(batch_size=self.batch_size, shuffle=False)
        for epoch in range(epochs):
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}: '):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'  Loss: {running_loss / len(train_loader):.4f}')
            self.test()
        print('Training completed.')

    def test(self):
        self.model.eval()
        test_loader = self.test_data.get_data_loader(batch_size=self.batch_size, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
