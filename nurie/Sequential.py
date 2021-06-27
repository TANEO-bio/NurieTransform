from typing import List
from ast import literal_eval

from .filters import Filter


class Sequential(object):
    def __init__(self, filters: List[Filter]):
        self.filters = filters
        self.dict_params = [(str(s).split(': ')[0], literal_eval(str(s).split(': ')[1])) for s in self.filters]

    def __repr__(self):
        rep = "\n".join([f'{filt}: {params}' for filt, params in self.dict_params])
        return rep

    def params(self):
        return self.dict_params

    def fit(self, image):
        image_filtered = image.copy()
        for method in self.filters:
            image_filtered = method(image_filtered)
        return image_filtered
