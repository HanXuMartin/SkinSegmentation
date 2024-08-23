from typing import Sequence, Union

import numpy as np
import torch
from torch import Tensor


class Batchlization:
    def __init__(self, input_size) -> None:
        self.input_size = input_size
        pass
    def __call__(self, inputs: Tensor) -> Tensor:
        # print(inputs.shape)
        inputs = inputs.unsqueeze(dim=0)
        # print(inputs.shape)
        return inputs

class Transpose:
    """transpose input ndarray from [N, H, W, C] to [N, C, H, W]"""
    def __call__(self, inputs: Tensor) -> Tensor:
        return inputs.permute(0, 3, 1, 2)
    

class ToTensor:
    def __call__(self, data) -> torch.Any:
        return to_tensor(data)

def to_tensor(
    data: Union[torch.Tensor, np.ndarray, Sequence, int,
                float]) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.

    Returns:
        torch.Tensor: the converted data.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence):
        return torch.tensor(data, dtype=torch.int64)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')