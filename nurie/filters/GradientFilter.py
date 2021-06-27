import numpy as np
import cv2
from .Filter import Filter


class GradientFilter(Filter):
    def __init__(self, mode='xy'):
        super(GradientFilter, self).__init__()
        self.mode = mode

    def __repr__(self):
        return f'GradientFilter: {{"mode":"{self.mode}"}}'

    def __call__(self, image):
        x, y = 0, 0

        if self.mode == 'x':
            x = 1
        elif self.mode == 'y':
            y = 1
        elif self.mode == 'xy':
            x, y = 1, 1

        kernel = np.array([[1, 0, -1],
                           [1, 0, -1],
                           [1, 0, -1]], np.float32)
        kernel /= kernel.std()
        image_x = cv2.filter2D(image, -1, kernel)
        image_y = cv2.filter2D(image, -1, kernel.T)
        image_filtered = np.sqrt(x * image_x.astype(float) ** 2 + y * image_y.astype(float) ** 2)
        return image_filtered.astype(np.uint8)
