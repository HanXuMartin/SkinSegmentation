from typing import Union

import cv2
import numpy as np


class Normalize:
    """Normalize one image with mean and std.
    code from MMCV
    """

    def __init__(self, 
                 mean: Union [np.ndarray, list], 
                 std: Union [np.ndarray, list], 
                 to_rgb: bool = False) -> None:
        if not isinstance(mean, np.ndarray):
            mean = np.array(mean)
        if not isinstance(std, np.ndarray):
            std = np.array(std)    
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """image: one 3-channel image of ndarray
        """
        image = image.copy().astype(np.float32)
        return self.imnormalize_(image, self.mean, self.std, self.to_rgb)

    def imnormalize_(self, 
                     image: np.ndarray, 
                     mean: np.ndarray, 
                     std: np.ndarray, 
                     to_rgb: bool = False) -> np.ndarray:
        """Inplace normalize an image with mean and std.

        Args:
            image (ndarray): image to be normalized.
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.

        Returns:
            ndarray: The normalized image.
        """
        # cv2 inplace normalization does not accept uint8
        assert image.dtype != np.uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)  # inplace
        cv2.subtract(image, mean, image)  # inplace
        cv2.multiply(image, stdinv, image)  # inplace
        return image
    


class NormimageList(Normalize):
    """Normalize a list of images with mean and std.
    """

    def __call__(self, image_list: list) -> list:
        """image: one 3-channel image

        Returns:
            list[ndarray]
        """
        return [self.imnormalize_(image.copy().astype(np.float32), self.mean, self.std, self.to_rgb) 
                for image in image_list]

