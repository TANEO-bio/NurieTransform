import numpy as np
import cv2
from .Filter import Filter


class MedianBlur(Filter):
    def __init__(self, kernel_size=5):
        super(MedianBlur, self).__init__()
        self.kernel_size = kernel_size

    def __repr__(self):
        return f'MedianBlur: {{"kernel_size":{self.kernel_size}}}'

    def __call__(self, image):
        image_filtered = cv2.medianBlur(image, self.kernel_size)
        return image_filtered.astype(np.uint8)
