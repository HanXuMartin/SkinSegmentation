import torch
from torch import Tensor, nn

from .baseblock import ConvBNReLU, ConvTransBNReLU
from .vgg import VGG16Pyramid 


class FCN(nn.Module):
    def __init__(self, 
                 stage_channels: list = [64, 128, 256, 512, 512],
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.backbone = VGG16Pyramid()
        self.convtrans = nn.ModuleList()
        for i, channel in enumerate(stage_channels):
            if i == 0:
                self.convtrans.append(
                    ConvTransBNReLU(
                        input_channels=64,
                        output_channels=32,
                        kernel_size=3,
                        padding=1,
                        output_padding=1,
                        stride=2
                    )
                )
            else:
                self.convtrans.append(
                    ConvTransBNReLU(
                        input_channels=stage_channels[i],
                        output_channels=stage_channels[i-1],
                        kernel_size=3,
                        padding=1,
                        output_padding=1,
                        stride=2
                    )
                )
        self.head = nn.Sequential(*[
            nn.Conv2d(
                in_channels=32, 
                out_channels=1,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        ])

    def forward(self, input: Tensor) -> Tensor:
        inputs = self.backbone(input)
        assert len(inputs) == len(self.convtrans)
        for i in range(len(inputs)):
            if i == 0:
                x = self.convtrans[len(inputs)-i-1](inputs[len(inputs)-i-1])
            else:
                x = self.convtrans[len(inputs)-i-1](inputs[len(inputs)-i-1] + x)
        x = self.head(x)
        return x