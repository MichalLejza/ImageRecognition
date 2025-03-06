import torch

from DataHandlers.TinyImageNet import TinyImageNetDataset
from ResNet.ResNet import ResNet
from torchvision.transforms import transforms


if __name__ == '__main__':
    resnet18 = ResNet(bottleneck=False, basic=True, layers=[2, 2, 2, 2], num_classes=10)
    # resnet18.print_info()
    # resnet34 = ResNet(bottleneck=False, basic=True, layers=[3, 4, 6, 3], num_classes=10)
    y = resnet18(torch.randn(1, 3, 224, 224))
    print(y.size())
    # resnet50 = ResNet(bottleneck=True, basic=False, layers=[3, 4, 6, 3], num_classes=1000)
    # resnet50.print_info()
