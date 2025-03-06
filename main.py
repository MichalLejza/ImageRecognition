from DataHandlers.TinyImageNet import TinyImageNetDataset
from ResNet.ResNet import ResNet
from torchvision.transforms import transforms


if __name__ == '__main__':
    # resnet18 = ResNet(bottleneck=False, basic=True, layers=[2, 2, 2, 2], num_classes=1000)
    # resnet18.print_info()
    # resnet34 = ResNet(bottleneck=False, basic=True, layers=[3, 4, 6, 3], num_classes=1000)
    # resnet34.print_info()
    resnet50 = ResNet(bottleneck=True, basic=False, layers=[3, 4, 6, 3], num_classes=1000)
    resnet50.print_info()

    transform = transforms.Compose([transforms.ToTensor()])
    data = TinyImageNetDataset(train=True, transform=transform)
    data.plot_image(0)
    data.plot_eight_images(random=True)
    print(data.images_shape())
