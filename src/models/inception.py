from typing import Tuple
from torch import nn
import torch

class InceptionModule(nn.Module):
    """
    Inception module.
    """

    conv_1x1: nn.Sequential
    conv_3x3_block: nn.Sequential
    conv_5x5_block: nn.Sequential
    max_pool_block: nn.Sequential    

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

        super().__init__()

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
                x: torch.Tensor) -> torch.Tensor:
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

    pool: nn.AvgPool2d
    conv: nn.Conv2d
    relu: nn.ReLU
    fc1: nn.Linear
    fc2: nn.Linear
    dropout: nn.Dropout

    def __init__(self, 
                 in_channels: int, 
                 num_classes: int,
                 in_after_flat: int) -> None:
        """
        Constructor for the InceptionAuxiliaryClassifier class.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of classes for the classification task.
        """

        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(in_after_flat, 1024)
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

    conv1: nn.Sequential
    conv2: nn.Sequential
    inception1: InceptionModule
    inception2: InceptionModule
    inception3: InceptionModule
    inception4: InceptionModule
    inception5: InceptionModule
    inception6: InceptionModule
    inception7: InceptionModule
    inception8: InceptionModule
    inception9: InceptionModule
    auxiliary_classifier1: InceptionAuxiliaryClassifier
    auxiliary_classifier2: InceptionAuxiliaryClassifier
    classifier: nn.Sequential

    def __init__(self,  
                 num_classes: int,
                 in_channels: int = 3) -> None:
        """
        Constructor for the Inception class.

        Args:
            num_classes (int): Number of classes for the classification task.
            in_channels (int): Number of input channels.
        """

        super().__init__()

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

        self.auxiliary_classifier1 = InceptionAuxiliaryClassifier(512, num_classes, 8192)
        self.auxiliary_classifier2 = InceptionAuxiliaryClassifier(528, num_classes, 8192)

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=1),
            nn.Flatten(),
            nn.Linear(495616, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
        )

        self.apply(self._init_weights)
    
    def _init_weights(self, m) -> None:
        """
        Initialize the weights of the model.

        Args:
            m (nn.Module): Module of the model.
        """

        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Inception model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        if x.shape[2] != 224 or x.shape[3] != 224:
            raise ValueError(f'Input tensor must have shape (N, C, 224, 224), but got {x.shape}')

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        if self.training:
            out_aux1 = self.auxiliary_classifier1(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        if self.training:
            out_aux2 = self.auxiliary_classifier2(x)
        x = self.inception7(x)
        x = self.inception8(x)
        x = self.inception9(x)
        x = self.classifier(x)
        if self.training:
            return out_aux1, out_aux2, x
        return x


class BinaryInceptionLoss(nn.BCEWithLogitsLoss):
    """
    Binary cross-entropy loss for the Inception model, it need the logits of
    auxiliary classifiers and the main classifier.
    """

    def __init__(self) -> None:
        """
        Constructor for the BinaryInceptionLoss class.
        """

        super().__init__()

    def forward(self, 
                input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BinaryInceptionLoss.

        Args:
            input (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Output tensor.
        """

        if len(input) != 3:
            raise ValueError(f'Input tensor must have 3 elements, but got {len(input)}')
        
        out_aux1, out_aux2, out_main = input
        loss_aux1 = super().forward(out_aux1, target)
        loss_aux2 = super().forward(out_aux2, target)
        loss_main = super().forward(out_main, target)

        return 0.3 * loss_aux1 + 0.3 * loss_aux2 + loss_main
    