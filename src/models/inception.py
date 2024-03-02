from torch import nn
import torch


class InceptionModule(nn.Module):

    def __init__(self, 
                 in_channels: int, 
                 out_1x1_channels: int,
                 reduce_3x3_channels: int, 
                 out_3x3_channels: int,
                 reduce_5x5_channels: int, 
                 out_5x5_channels: int,
                 out_pool_channels: int) -> None:
        """
        Constructor for the InceptionModule class.

        Args:
            in_channels (int): Number of input channels.
            out_1x1_channels (int): Number of output channels for the 1x1 convolution.
            reduce_3x3_channels (int): Number of output channels for the reduction 1x1 convolution before the 3x3 convolution.
            out_3x3_channels (int): Number of output channels for the 3x3 convolution.
            reduce_5x5_channels (int): Number of output channels for the reduction 1x1 convolution before the 5x5 convolution.
            out_5x5_channels (int): Number of output channels for the 5x5 convolution.
            out_pool_channels (int): Number of output channels for the 1x1 convolution after max pooling.
        """

        super(InceptionModule, self).__init__()

        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1_channels, kernel_size=1),
            nn.ReLU()
        )

        self.conv_3x3_block = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(reduce_3x3_channels, out_3x3_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv_5x5_block = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(reduce_5x5_channels, out_5x5_channels, kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.max_pool_block = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, 
                x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the InceptionModule.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        out_1x1 = self.conv_1x1(x)
        out_3x3 = self.conv_3x3_block(x)
        out_5x5 = self.conv_5x5_block(x)
        out_max_pool = self.max_pool_block(x)

        outputs = torch.cat([out_1x1, out_3x3, out_5x5, out_max_pool], 1)
        return outputs

class InceptionAuxiliaryClassifier(nn.Module):
    """
    Auxiliary classifier for the Inception model.
    """

    def __init__(self, 
                 in_channels: int, 
                 num_classes: int) -> None:
        """
        Constructor for the InceptionAuxiliaryClassifier class.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of classes for the classification task.
        """

        super(InceptionAuxiliaryClassifier, self).__init__()

        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the InceptionAuxiliaryClassifier.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        x = self.pool(x)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Inception(nn.Module):
    """
    Inception model.
    """

    def __init__(self, 
                 in_channels: int, 
                 num_classes: int) -> None:
        """
        Constructor for the Inception class.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of classes for the classification task.
        """

        super(Inception, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )


        self.inception1 = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.inception2 = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.inception3 = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4 = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.inception5 = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.inception6 = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.inception7 = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.inception8 = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception9 = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        self.auxiliary_classifier1 = InceptionAuxiliaryClassifier(512, num_classes)
        self.auxiliary_classifier2 = InceptionAuxiliaryClassifier(528, num_classes)

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Inception model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception1(x)
        x = self.inception2(x)
        out_aux1 = self.auxiliary_classifier1(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        out_aux2 = self.auxiliary_classifier2(x)
        x = self.inception6(x)
        x = self.inception7(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.classifier(x)

        return x, out_aux1, out_aux2



        
