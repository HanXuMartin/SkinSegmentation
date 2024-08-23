import cv2
import numpy as np


class ImgBinary:
    def __init__(self,
                 camp: int = 127) -> None:
        self.camp = camp
        pass

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.threshold(image, thresh=self.camp, maxval=255, type=cv2.THRESH_BINARY)[1]