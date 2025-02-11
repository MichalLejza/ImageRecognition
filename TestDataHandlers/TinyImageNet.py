from torchvision import transforms
from DataHandlers.TinyImageNet import TinyImageNetDataset


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = TinyImageNetDataset(train=True, transform=transform)
    dataset1 = TinyImageNetDataset(test=True, transform=transform)
    dataset2 = TinyImageNetDataset(val=True, transform=transform)
    dataset1.plot_image(0)
    print(dataset)
    print(dataset1)
    print(dataset2)
