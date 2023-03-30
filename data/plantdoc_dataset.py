# Copyright (C) 2023 Xi Jin
# Fake it. Util make it. Start it, Util love it

import os
import random

from PIL import Image
import torch
from torchvision.transforms import transforms

from data.base_dataset import BaseDataset
from data.image_folder import ImageFolder, default_loader
from utils.utils import class2index


def get_defaut_transforms(img_size, is_train=True):
    train_trans = transforms.Compose([
                transforms.RandomSizedCrop(max(img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_trans = transforms.Compose([
            #Higher scale-up for inception
            transforms.Scale(int(max(img_size)/224*256)),
            transforms.CenterCrop(max(img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    return train_trans, val_trans


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
        self.transform = get_defaut_transforms(opt.img_size, opt.is_train)
        self.target_transform = target_transform

    # TODO:完成数据迭代方法
    def __getitem__(self, index):
        img_path, label = self.image_tuple[index]
        img = self.loader(img_path)

        pass

    def __len__(self):
        return len(self.image_tuple)