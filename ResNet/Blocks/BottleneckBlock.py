import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    """
    Bottleneck block for ResNet50, ResNet101, and ResNet152
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int=1, downsample=None, expansion: int=4):
        super(BottleneckBlock, self).__init__()
        # Expansion factor of the last Convolution Layer. Most commonly it is 4
        self.expansion = expansion

        # First Conv Layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second Conv Layer
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Third Conv Layer
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # Downsample,
        self.downsample = downsample

    def forward(self, x):
        # Input Identity
        input_identity = x if self.downsample is None else self.downsample(x)

        # First Conv Layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second Conv Layer
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # Third Conv Layer
        out = self.conv3(out)
        out = self.bn3(out)
        # Add input to output
        out += input_identity

        # Apply relu
        out = F.relu(out)

        return out