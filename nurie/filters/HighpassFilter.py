import numpy as np
import cv2
from .Filter import Filter


class HighpassFilter(Filter):
    def __init__(self):
        super(HighpassFilter, self).__init__()

    def __repr__(self):
        return 'HighpassFilter: {}'

    def __call__(self, image):
        kernel = np.array([[-1, -1, -1],
                           [-1, 8, -1],
                           [-1, -1, -1]], np.float32)
        kernel /= kernel.std()
        image_filtered = cv2.filter2D(image, -1, kernel)
        return image_filtered.astype(np.uint8)
