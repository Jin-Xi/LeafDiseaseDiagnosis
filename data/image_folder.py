# Copyright (C) 2023 Xi Jin
# Fake it. Util make it. Start it, Util love it
"""A modified image folder class
modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import os
from abc import ABC, abstractmethod

from PIL import Image
import pandas as pd
import torch.utils.data as data


# 可用图像类型
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s ——> 没有此目录' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset, ABC):

    def __init__(self, root, transform=None, target_transform=None, return_paths=False,
                 loader=default_loader):
        df = pd.read_csv(self.root)
        if len(df) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.df = df
        self.transform = transform
        self.target_transform = target_transform
        self.return_paths = return_paths
        self.loader = loader

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    # from tqdm import tqdm
    # dataset = ImageFolder(root="E:\\2023DiseaseDiagnose\\datasets\\PlantVillage")
    # for data in tqdm(dataset):
    #     pass
    # TODO: 数据集单元测试
    pass

    