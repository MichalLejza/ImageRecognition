import torch
import torch.nn as nn
from ResNet.Blocks.BasicBlock import BasicBlock
from ResNet.Blocks.BottleneckBlock import BottleneckBlock


class ResNet(nn.Module):
    def __init__(self, bottleneck: bool=False, basic: bool=False, layers: list=None, num_classes: int=10):
        super(ResNet, self).__init__()
        # Check if provided arguments are proper
        if bottleneck == basic or list is None or len(layers) != 4:
            raise Exception('Wrong Arguments Provided for ResNet Class')

        self.in_channels = 64
        self.expansion = 1 if basic else 4

        # First conv Layer, identical to all ResNet versions
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Main blocks of ResNet
        if bottleneck:
            self.layer1 = self.__make_layer_bottleneck(64, layers[0])
            self.layer2 = self.__make_layer_bottleneck(128, layers[1], stride=2)
            self.layer3 = self.__make_layer_bottleneck(256, layers[2], stride=2)
            self.layer4 = self.__make_layer_bottleneck(512, layers[3], stride=2)
        if basic:
            self.layer1 = self.__make_layer_basic(64, layers[0])
            self.layer2 = self.__make_layer_basic(128, layers[1], stride=2)
            self.layer3 = self.__make_layer_basic(256, layers[2], stride=2)
            self.layer4 = self.__make_layer_basic(512, layers[3], stride=2)

        # End layer
        self.avgpool = nn.AvgPool2d((1, 1))
        self.fc = nn.Linear(512 * self.expansion, num_classes, bias=True)

    def __make_layer_basic(self, out_channels, num_blocks, stride=1):
        # Basic Block list
        layers: list = []
        #
        downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion))
        if stride == 1 or self.in_channels == out_channels * self.expansion:
            downsample = None
        #
        layers.append(BasicBlock(in_channels=self.in_channels, out_channels=out_channels, stride=stride, downsample=downsample))
        #
        self.in_channels = out_channels * self.expansion
        #
        for i in range(num_blocks - 1):
            layers.append(BasicBlock(in_channels=self.in_channels, out_channels=out_channels))
        #
        return nn.Sequential(*layers)

    def __make_layer_bottleneck(self, out_channels, num_blocks, stride=1):
        #
        layers: list= []
        #
        downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion))
        if stride == 1 or self.in_channels == out_channels * self.expansion:
            downsample = None
        #
        layers.append(BottleneckBlock(in_channels=self.in_channels, out_channels=out_channels, stride=stride, downsample=downsample))
        #
        self.in_channels = out_channels * self.expansion
        #
        for i in range(num_blocks - 1):
            layers.append(BottleneckBlock(in_channels=self.in_channels, out_channels=out_channels))
        #
        return nn.Sequential(*layers)

    def foward(self, x):
        #
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def print_info(self):
        print(self)
        print(f'Number of parameters: {sum(p.numel() for p in self.parameters())}')
        print(f'Number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}')
