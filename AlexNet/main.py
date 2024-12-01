from torchvision import transforms

from Network import AlexNetTrainer


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    net = AlexNetTrainer(batch_size=64, transform=transform)
    net.train(10)
