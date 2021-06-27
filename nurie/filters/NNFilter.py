import numpy as np
import cv2
import torch
from .models.UNet.model import UNet
from .models.ENet.model import ENet
from .Filter import Filter


class NNFilter(Filter):
    def __init__(self, mode='UNet', ckpt=None):
        self.ckpt = ckpt
        self.mode = mode
        if self.mode == 'UNet':
            self.model = UNet(num_classes=2)
        elif self.mode == 'ENet':
            self.model = ENet(num_classes=2)
        self.model.load_state_dict(torch.load(self.ckpt), strict=False)

    def __repr__(self):
        return f'NNFilter: {{"mode":"{self.mode}", "ckpt":"{self.ckpt}"}}'

    def __call__(self, image):
        shape_tmp = image.shape
        image = np.stack([image, image, image], axis=-1).astype(np.uint8)
        image = cv2.resize(image, (400, 400)) / 255
        image = image.transpose(2,0,1)[np.newaxis, :, :, :]
        image = torch.from_numpy(image.astype(np.float32)).clone()
        image = image.float()

        with torch.no_grad():
            result = self.model.forward(image)

        image_filtered = result.argmax(axis=1)[0].numpy().astype(np.uint8)
        image_filtered = cv2.resize(image_filtered, shape_tmp[::-1])

        return image_filtered.astype(np.uint8)
