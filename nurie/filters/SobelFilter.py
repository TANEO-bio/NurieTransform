import numpy as np
import cv2
from .Filter import Filter


class SobelFilter(Filter):
    def __init__(self, mode='xy', kernel_size=5):
        super(SobelFilter, self).__init__()
        self.mode = mode
        self.kernel_size = kernel_size

    def __repr__(self):
        return f'SobelFilter: {{"mode":"{self.mode}","kernel_size":{self.kernel_size}}}'

    def __call__(self, image):
        x, y = 0, 0
        if self.mode == 'x':
            x = 1
        elif self.mode == 'y':
            y = 1
        elif self.mode == 'xy':
            x, y = 1, 1

        image_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=self.kernel_size)
        image_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=self.kernel_size)
        image_filtered = np.sqrt(x * image_x.astype(float) ** 2 + y * image_y.astype(float) ** 2)
        return image_filtered.astype(np.uint8)
