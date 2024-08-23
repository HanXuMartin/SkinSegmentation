import torch
from torch import Tensor, nn

from .baseblock import ConvBNReLU


class VGGBasicBlock(nn.Module):
    def __init__(self, 
                 num_convs: int,
                 input_channels: int,
                 output_channels:int,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layers: list = []
        for layer_idx in range(num_convs):
            if layer_idx == 0:
                layers.append(
                    ConvBNReLU(
                        input_channels=input_channels,
                        output_channels=output_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ))
            else:
                layers.append(
                    ConvBNReLU(
                        input_channels=output_channels,
                        output_channels=output_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    ))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)
    def forward(self, input: Tensor) -> Tensor:
        return self.block(input)
    
vgg: dict = {"16": [2, 2, 3, 3, 3]}

class VGG16Pyramid(nn.Module):
    def __init__(self, 
                 input_channel: int = 3,
                 output_channels: list = [64, 128, 256, 512, 512],
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layer_nums = vgg["16"]
        self.blocks = nn.ModuleList()
        self.stem = nn.Sequential(*[
            ConvBNReLU(input_channels=input_channel,
                      output_channels=output_channels[0],
                      kernel_size=3,
                      stride=1,
                      padding=1),
            ConvBNReLU(input_channels=output_channels[0],
                      output_channels=output_channels[0],
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ])
        for i, num_convs in enumerate(layer_nums):
            if i == 0:
                self.blocks.append(self.stem)
            else:
                self.blocks.append(
                    VGGBasicBlock(
                        num_convs=layer_nums[i],
                        input_channels=output_channels[i-1],
                        output_channels=output_channels[i],
                    ))
    def forward(self, input: Tensor) -> Tensor:
        outputs = []
        for block in self.blocks:
            inter = block(input)
            outputs.append(inter)
            input = inter
        return outputs