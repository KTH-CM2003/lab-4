import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """
    A double convolutional block used in the U-Net architecture.
    
    This module applies two consecutive convolutional layers, each followed by
    batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        #--- CODE START ---
        self.conv = None
        #--- CODE END ---

    def forward(self, x):
        """
        Forward pass of the DoubleConv module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output after applying the double convolution.
        """
        #--- CODE START ---
        return None
        #--- CODE END ---

class UNET(nn.Module):
    """
    U-Net architecture for image segmentation tasks.

    This implementation follows the original U-Net paper with some modifications.
    It consists of a contracting path (downsampling) and an expansive path (upsampling),
    with skip connections between the corresponding layers.

    Args:
        in_channels (int): Number of input channels (default: 3 for RGB images).
        out_channels (int): Number of output channels (default: 1 for binary segmentation).
        features (list): List of feature dimensions for each level of the U-Net (default: [64, 128, 256, 512]).
    """
    def __init__(
            self, in_channels=3, out_channels=3, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #--- CODE START ---
        # Down part of UNET
        for feature in features:
            pass

        # Up part of UNET
        for feature in reversed(features):
            pass

        self.bottleneck = None
        self.final_conv = None
        #--- CODE END ---

    def forward(self, x):
        """
        Forward pass of the U-Net.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output segmentation map.
        """
        #--- CODE START ---
        skip_connections = []

        for down in self.downs:
            pass
        
        # Apply bottleneck
        x = self.bottleneck(x)
        #--- CODE END ---

        skip_connections = skip_connections[::-1]
        #--- CODE START --- 
        # Upsample and concatenate skip connections, look carefully at architecture to implement this
        
        
        #--- CODE END ---
        return self.final_conv(x)
