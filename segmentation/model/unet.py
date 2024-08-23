from typing import List

import torch
from torch import nn

from .baseblock import ConvBNReLU


class UnetDownSampleBlock(nn.Module):
    def __init__(self, 
                 input_channels: int,
                 output_channels: int,
                 if_maxpool: bool = True,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        down_conv_list = []
        if if_maxpool:
            down_conv_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        down_conv_list.append(
            ConvBNReLU(
                input_channels=input_channels,
                output_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1)
        )
        down_conv_list.append(
            ConvBNReLU(
                input_channels=output_channels,
                output_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1)
        )
        self.downsample = nn.Sequential(*down_conv_list)

    def forward(self, x):
        return self.downsample(x) 

class UnetUpSampleBlock(nn.Module):
    def __init__(self, 
                 input_channels: int,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        up_conv_list = []
        mid_channels = int(input_channels/2)
        up_conv_list.append(
            ConvBNReLU(
                input_channels=input_channels,
                output_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1)
        )
        up_conv_list.append(
            ConvBNReLU(
                input_channels=mid_channels,
                output_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1)
        )
        up_conv_list.append(
            nn.ConvTranspose2d(
                in_channels=mid_channels,
                out_channels=int(mid_channels/2),
                kernel_size=2,
                stride=2)
        )
        self.upsample = nn.Sequential(*up_conv_list)
    def forward(self, x):
        return self.upsample(x)

class Unet(nn.Module):
    def __init__(self, 
                 input_channels: int = 3,
                 output_channels: int = 1,
                 stages: int = 4,
                 downsample_channels: List[int] = [64, 128, 256, 512],
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stages = stages
        downsample_blocks = []
        for i in range(stages):
            if i == 0:
                downsample_blocks.append(
                    UnetDownSampleBlock(
                        input_channels=input_channels,
                        output_channels=downsample_channels[i],
                        if_maxpool=False
                    )
                )
            else:
                downsample_blocks.append(
                    UnetDownSampleBlock(
                        input_channels=downsample_channels[i-1],
                        output_channels=downsample_channels[i],
                    )
                )
        self.downsample = nn.ModuleList(downsample_blocks)
        self.bridge = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBNReLU(
                input_channels=downsample_channels[-1],
                output_channels=downsample_channels[-1]*2,
                kernel_size=3,
                stride=1,
                padding=1),
            ConvBNReLU(
                input_channels=downsample_channels[-1]*2,
                output_channels=downsample_channels[-1]*2,
                kernel_size=3,
                stride=1,
                padding=1),    
            nn.ConvTranspose2d(
                in_channels=downsample_channels[-1]*2,
                out_channels=downsample_channels[-1],
                kernel_size=2,
                stride=2)
        )

        upsample_channels = downsample_channels[::-1]
        upsample_blocks = []
        for i in range(stages):
            if i == 0:
                upsample_blocks.append(self.bridge)
            else:
                upsample_blocks.append(
                    UnetUpSampleBlock(input_channels=upsample_channels[i-1]*2)
                )
        self.upsamples = nn.ModuleList(upsample_blocks)
        self.endblock = nn.Sequential(
            ConvBNReLU(
                input_channels=downsample_channels[0]*2,
                output_channels=downsample_channels[0],
                kernel_size=3,
                stride=1,
                padding=1),
            ConvBNReLU(
                input_channels=downsample_channels[0],
                output_channels=downsample_channels[0],
                kernel_size=3,
                stride=1,
                padding=1),
            ConvBNReLU(
                input_channels=downsample_channels[0],
                output_channels=1,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        outputs = []
        for i in range(self.stages):
            if i == 0:
                outputs.append(self.downsample[i](x))
            else:
                outputs.append(self.downsample[i](outputs[-1]))
        outputs = outputs[::-1]

        for i in range(self.stages):
            if i == 0:
                temp = self.upsamples[i](outputs[i])
                temp = torch.concat((outputs[i], temp), dim=1)
            else:
                temp = self.upsamples[i](temp)
                temp = torch.concat((outputs[i], temp), dim=1)
        
        output = self.endblock(temp)
        return output

