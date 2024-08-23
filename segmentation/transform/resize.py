import os

import cv2
import numpy as np


class Resize:
    def __init__(self,
                 resize: tuple,
                 backend = "cv2") -> None:
        assert len(resize) == 2, "size shoule has a length of 2"
        self.resize = resize
        self.backend = backend 

    def __call__(self, ori_image: np.ndarray) -> np.ndarray:
        return cv2.resize(ori_image, self.resize)
