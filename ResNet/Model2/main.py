from torchvision import transforms

from DataHandlers.TinyImageNet import TinyImageNetDataset


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Ensure images are 64x64
        transforms.ToTensor(),  # Convert to PyTorch tensor (C, H, W)
    ])
    dataset = TinyImageNetDataset(train=True, transform=transform)
    print(len(dataset))
    print(dataset.images.shape)
    dataset.plot_image(0)
    dataset.plot_eight_images(random=True)
