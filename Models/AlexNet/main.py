import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from DataHandlers.Cifar10 import Cifar10Dataset
from Models.AlexNet.Model1 import AlexNet


class AlexNetTrainer:
    def __init__(self, batch_size: int  = 64, transform = None):
        self.train_data = Cifar10Dataset(train=True, transform=transform)
        self.test_data = Cifar10Dataset(test=True, transform=transform)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.model = AlexNet(self.train_data.images_shape(), self.train_data.num_classes()).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

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



if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    net = AlexNetTrainer(batch_size=64, transform=transform)
    net.train(10)
    net.test()
