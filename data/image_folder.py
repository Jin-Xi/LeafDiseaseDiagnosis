# Copyright (C) 2023 Xi Jin
# Fake it. Util make it. Start it, Util love it

import os

from PIL import Image
import torch.utils.data as data


# 可用图像类型
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


