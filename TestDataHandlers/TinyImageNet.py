from torchvision import transforms
from DataHandlers.TinyImageNet import TinyImageNetDataset


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = TinyImageNetDataset(train=True, transform=transform)
    print(len(dataset))
    print(dataset.images.shape)
