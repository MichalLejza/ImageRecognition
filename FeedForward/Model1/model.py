import torch.nn as nn
from torch.utils.data import TensorDataset

from handler import *


class MLP(nn.Module):
    def __init__(self, fc: tuple, batch_size: int, epochs: int, num_classes: int, data_path: str):
        super().__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.trainSet = MnistDataset(imagePath=data_path+"\\train-images", labelPath=data_path+"\\train-labels")
        self.testSet = MnistDataset(imagePath=data_path+"\\test-images", labelPath=data_path+"\\test-labels")

        self.fc1 = nn.Linear(28 * 28, fc[0], bias=True)
        self.fc2 = nn.Linear(fc[0], fc[1], bias=True)
        self.fc3 = nn.Linear(fc[1], fc[2], bias=True)
        self.fc4 = nn.Linear(fc[2], num_classes, bias=True)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        print(self.device)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x

    def train_model(self):
        encoded = torch.nn.functional.one_hot(self.trainSet.labels, num_classes=self.num_classes)
        train_dataset = TensorDataset(self.trainSet.images, encoded.float())
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.epochs):
            self.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss / len(train_loader):.4f}')
        print('Training completed.')

    def test_model(self) -> None:
        test_dataset = TensorDataset(self.testSet.images, self.testSet.labels)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total  # Calculate accuracy
        print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

    def save_model(self, dirPath: str) -> None:
        torch.save(self.state_dict(), dirPath + '/Model2.pth')

    def load_model(self, dirPath: str) -> None:
        self.load_state_dict(torch.load(dirPath + '/Model2.pth'))

    def predict(self, idx: int) -> None:
        image = self.testSet.images[idx]
        label = self.testSet.labels[idx]
        image = image.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            output = self(image.to(self.device))
            _, predicted = torch.max(output.data, 1)
        image = image.squeeze()
        plt.imshow(image.numpy().transpose(), cmap='gray')
        plt.axis('off')
        plt.title(f"Przewidywana liczba: {predicted[0].numpy()}   Faktyczna Liczba: {label}")
        plt.show()

    def displayImage(self, idx):
        image = self.testSet.images[idx].squeeze()
        label = self.testSet.labels[idx]
        plt.imshow(image.numpy(), cmap='gray')
        plt.axis('off')
        plt.title(f"{label}")
        plt.show()


if __name__ == '__main__':
    model = MLP(fc=(1024, 512, 256), batch_size=64, epochs=20, num_classes=10,
                data_path="C:\\Users\\Michał\\Desktop\\Data\\EMNIST\\MNIST")
    print(model)
    model.train_model()
    model.test_model()

    while True:
        i = int(input('Which image do you want to predict?: '))
        model.predict(i)
