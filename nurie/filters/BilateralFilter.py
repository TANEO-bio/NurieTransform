import numpy as np
import cv2
from .Filter import Filter


class BilateralFilter(Filter):
    def __init__(self, kernel_size=5, sigma=75):
        super(BilateralFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __repr__(self):
        return f'BilateralFilter: {{"kernel_size":{self.kernel_size},"sigma":{self.sigma}}}'

    def __call__(self, image):
        image_filtered = cv2.bilateralFilter(image, self.kernel_size, self.sigma, self.sigma)
        return image_filtered.astype(np.uint8)
