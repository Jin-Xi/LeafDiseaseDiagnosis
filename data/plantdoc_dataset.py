# Copyright (C) 2023 Xi Jin
# Fake it. Util make it. Start it, Util love it

import os
import random

from PIL import Image
from collections import Counter
import torch
import pandas as pd
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset
from data.image_folder import ImageFolder, default_loader
from utils.utils import class2index


def get_defaut_transforms(load_size, crop_size, is_train=True):
    train_trans = transforms.Compose([
                transforms.Scale(load_size),
                transforms.RandomSizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_trans = transforms.Compose([
            #Higher scale-up for inception
            transforms.Scale(load_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    if is_train:
        return train_trans
    return val_trans


class PlantDocDataset(ImageFolder):
    def __init__(self, opt, transform=None):
        ImageFolder.__init__(self, opt)
        self.class_number = len(Counter(self.df.lable.to_list()))
        self.loader = default_loader
        self.transform = get_defaut_transforms(opt.load_size, opt.crop_size, opt.is_train)

    # 完成数据迭代方法
    def __getitem__(self, index):
        img_path, label_index = self.df.loc[index].img_src, self.df.loc[index].one_hot
        img = self.transform(self.loader(img_path))
        label = torch.zeros(self.class_number)
        label[label_index] = 1
        return img, label

    def __len__(self):
        return len(self.df)