import numpy as np
import cv2
from .Filter import Filter


class GaussianBlur(Filter):
    def __init__(self, kernel_size=5):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size

    def __repr__(self):
        return f'GaussianBlur: {{"kernel_size":{self.kernel_size}}}'

    def __call__(self, image):
        image_filtered = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
        return image_filtered.astype(np.uint8)
