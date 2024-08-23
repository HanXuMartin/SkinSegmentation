import torch
from torch import Tensor, nn


class ConvBNReLU(nn.Module):
    def __init__(self, 
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 groups: int = 1,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels, 
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x: Tensor):
        return self.conv(x)
    

class ConvTransBNReLU(nn.Module):
    def __init__(self, 
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 output_padding: int,
                 groups: int = 1,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.convtrans = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=input_channels, 
                out_channels=output_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.convtrans(x)