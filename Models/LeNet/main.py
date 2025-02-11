import time
import torch
import torch.nn as nn
from tqdm import tqdm
from DataHandlers.Mnist import MnistDataset
from Models.LeNet.Model1 import LeNet


class CNN(nn.Module):
    def __init__(self, batch_size: int, epochs: int, kind: str = 'Classic'):
        super(CNN, self).__init__()
        self.train_set = MnistDataset(kind=kind, train=True)
        self.test_set = MnistDataset(kind=kind, test=True)

        self.epochs = epochs
        self.batch_size = batch_size
        self.num_classes = self.train_set.num_classes()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LeNet(data_set_size=self.train_set.images_shape(), num_classes=self.num_classes).to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def train_model(self) -> None:
        start = time.time()
        train_loader = self.train_set.get_data_loader(batch_size=64, shuffle=False)
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}: '):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'  Loss: {running_loss / len(train_loader):.4f}')
            self.test_model()
        print('Training completed.')
        end = time.time()
        duration = end - start
        print(f"Program took {duration:.4f} seconds to run.")

    def test_model(self) -> None:
        test_loader = self.test_set.get_data_loader(batch_size=64, shuffle=False)
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Testing network: '):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

    def save_model(self, dirPath: str) -> None:
        torch.save(self.model.state_dict(), dirPath + '\\Model4.pth')

    def load_model(self, dirPath: str) -> None:
        self.model.load_state_dict(torch.load(dirPath + '\\Model4.pth'))

    def predict(self, image):
        self.model.eval()
        image = torch.tensor(image).float().unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
            _, predicted = torch.max(output.data, 1)
        return predicted


if __name__ == '__main__':
    model = CNN(batch_size=64, epochs=10, kind='Classic')
    model.train_model()