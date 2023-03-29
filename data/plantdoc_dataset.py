# Copyright (C) 2023 Xi Jin
# Fake it. Util make it. Start it, Util love it

import os
import random

from PIL import Image
import torch

from data.base_dataset import BaseDataset
from data.image_folder import ImageFolder, default_loader
from utils.utils import class2index


# TODO:torch transform


class PlantDocDataset(ImageFolder):
    def __init__(self, opt, transform=None, target_transform=None):
        super().__init__(self, opt)
        # 获取类别信息
        img_classes = os.listdir(opt.image_root)
        self.index2class, self.class2index = class2index(img_classes)

        self.image_tuple = []
        for class_name in img_classes:
            class_path = os.path.join(opt.root, class_name)
            for path in os.listdir(class_path):
                self.image_tuple.append((os.path.join(class_path, path), torch.tensor(self.class2index[class_name])))

        self.loader = default_loader
        # FIXME: 写一个本数据集专用的transform方法
        self.transform = transform
        self.target_transform = target_transform

    # TODO:完成数据迭代方法
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.image_tuple)