import numpy as np
import cv2
from .Filter import Filter


class AverageFilter(Filter):
    def __init__(self, kernel_size=5):
        super(AverageFilter, self).__init__()
        self.kernel_size = kernel_size

    def __repr__(self):
        return f'AverageFilter: {{"kernel_size":{self.kernel_size}}}'

    def __call__(self, image):
        kernel = np.ones((self.kernel_size, self.kernel_size)) / self.kernel_size ** 2
        image_filtered = cv2.filter2D(image, -1, kernel)
        return image_filtered.astype(np.uint8)
