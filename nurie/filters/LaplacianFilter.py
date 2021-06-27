import numpy as np
import cv2
from .Filter import Filter


class LaplacianFilter(Filter):
    def __init__(self):
        super(LaplacianFilter, self).__init__()

    def __repr__(self):
        return 'LaplacianFilter: {}'

    def __call__(self, image):
        kernel = np.array([
                            [-1, -3, -4, -3, -1],
                            [-3,  0,  6,  0, -3],
                            [-4,  6, 20,  6, -4],
                            [-3,  0,  6,  0, -3],
                            [-1, -3, -4, -3, -1]
                            ], np.float32)
        kernel /= kernel.std()
        image_filtered = cv2.filter2D(image, -1, kernel)
        return image_filtered.astype(np.uint8)
