# Copyright (C) 2023 Xi Jin
# Fake it. Util make it. Start it, Util love it

import os
import random

from PIL import Image

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset


class PlantDocDataset(BaseDataset):
    def __init__(
        self,
        opt
    ):
        BaseDataset.__init__(self, opt)
        # 获取类别信息
        img_classes = os.listdir(opt.image_root)
        self.num_class = len(img_classes)
        class2index = {}
        for index, name in enumerate(img_classes):
            class2index[name] = index
        self.index2class = {index: name for index, name in enumerate(img_classes)}

        self.image_tuple = []
        for class_name in img_classes:
            class_path = os.path.join(img_root, class_name)
            for path in os.listdir(class_path):
                self.image_tuple.append((os.path.join(class_path, path), torch.tensor(class2index[class_name])))

        self.class_num_count = {}
        # 统计不同类别图像的数量
        for class_name in img_classes:
            class_path = os.path.join(img_root, class_name)
            self.class_num_count[class_name] = len(os.listdir(class_path))

        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.image_tuple[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target, self.num_class)
        return img, target

    def __len__(self):
        return len(self.image_tuple)