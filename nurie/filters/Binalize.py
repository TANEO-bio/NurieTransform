import numpy as np
from .Filter import Filter


class Binalize(Filter):
    def __init__(self, threshold=25):
        super(Binalize, self).__init__()
        self.threshold = threshold

    def __repr__(self):
        return f'Binalize: {{"threshold":{self.threshold}}}'

    def __call__(self, image):
        return (image >= self.threshold).astype(np.uint8)
