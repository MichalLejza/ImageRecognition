import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic block for ResNet used in models: ResNet18, ResNet34.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # First Convolution Layer. In ResNet18 and ResNet34 it is 3x3 with padding 1. Other
        # paramaters differ based on which vlock you are building.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second Convolution Layer, pay attention that the stride is always 1 and padding is 1.
        # size of kernels is 3x3
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsample, in ResNet we add a skip connections part, and we have to determine if we
        # just add input to output or if we first apply 1x1 conv.
        self.downsample = downsample

    def forward(self, x):
        # Input Identity, basically we copy the input
        input_identity = x
        # We apply downsample. If this block is first in a sequence, downsample is None, so we don't do anything
        # but if this block is not first, we have to apply downsample which is a 1x1 conv and batchnorm.
        if self.downsample is not None:
            input_identity = self.downsample(x)

        # First Convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Second Convolution, we don't apply relu at first
        out = self.conv2(out)
        out = self.bn2(out)

        # Add input to output
        out += input_identity

        # Apply relu
        out = F.relu(out)

        return out
