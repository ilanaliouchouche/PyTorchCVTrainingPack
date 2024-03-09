from typing import Optional
import torch
from torch import nn
from functools import reduce

class ResNet50(nn.Module):
    """
    ResNet50 model.
    """

    in_channels: int
    conv1: nn.Sequential
    layers: nn.Sequential
    fc: nn.Sequential

    def __init__(self, 
                 num_classes: int,
                 in_channels: int = 3) -> None:
        """
        Constructor for the ResNet50 class.

        Args:
            num_classes (int): Number of classes in the dataset.
            in_channels (int): Number of input channels.
        """

        super().__init__()

        self.res_in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layers = nn.Sequential(
            self._make_layer(64, 3),
            self._make_layer(128, 4, stride=2),
            self._make_layer(256, 6, stride=2),
            self._make_layer(512, 3, stride=2)
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * 4, num_classes)
        )

    def _make_layer(self, 
                    out_channels: int, 
                    blocks: int, 
                    stride: int = 1) -> nn.Sequential:
        """
        Create a layer of the ResNet50 model.

        Args:
            out_channels (int): Number of output channels in each block.
            blocks (int): Number of blocks in the layer.
            stride (int): Stride for the first block.

        Returns:
            nn.Sequential: Layer of the ResNet50 model.
        """

        downsample = None
        if stride != 1 or out_channels * ResBlock.expansion != self.res_in_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.res_in_channels, out_channels * ResBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResBlock.expansion),
            )

        init_layer = ResBlock(self.res_in_channels, out_channels, stride, downsample)
        self.res_in_channels = out_channels * ResBlock.expansion

        layers = reduce(lambda acc, _: acc + [ResBlock(self.res_in_channels, out_channels)], 
                        range(1, blocks), 
                        [init_layer])

        return nn.Sequential(*layers)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ResNet50 model.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the ResNet50 model.
        """

        if x.shape[2] != 224 or x.shape[3] != 224:
            raise ValueError(f"Input tensor must have shape (N, C, 224, 224) but got {x.shape}")

        x = self.conv1(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc(x)

        return x
    
class ResBlock(nn.Module):
    """
    Bottleneck block for the ResNet50 model.
    """

    conv1: nn.Sequential
    conv2: nn.Sequential
    conv3: nn.Sequential
    downsample: Optional[nn.Module]
    relu: nn.ReLU
    # static
    expansion: int

    expansion = 4

    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 stride: int = 1, 
                 downsample: Optional[nn.Module] = None) -> None:
        """
        Constructor for the Bottleneck class.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for the first block.
            downsample (Optional[nn.Module]): Downsample layer.
        """

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Bottleneck.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)  

        return out
