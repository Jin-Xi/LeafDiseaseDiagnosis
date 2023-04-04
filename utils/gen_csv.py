# Copyright (C) 2023 Xi Jin
# Fake it. Util make it. Start it, Util love it


import os
import sys
from collections import Counter

import torch
from torch.nn import functional as F
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


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


def find_target_dataset(target_dataset, scope='./'):
    """给一个数据集的名字，查找数据集的目录

    Args:
        target_dataset (str): 目标数据集的名字
        scope (str): 查找的目标目录 
    """
    # dirpath：当前查找的目录；dirname：当前的文件夹名字；filenames：？？
    for dirpath, dirname, filenames in os.walk(scope):
        if target_dataset in dirname:
            return os.path.join(dirpath, target_dataset)
    return None


def gen_csv(dataset_name='PlantVillage'):
    """通过数据文件夹生成数据csv文件

    Args:
        dataset_name (str, optional):数据集名字. Defaults to 'PlantVillage'.
    """
    pwd = find_target_dataset(dataset_name)
    images = make_dataset(pwd)
    # 取出labels
    labels = []
    for img in images:
        if sys.platform != "win32":
            labels.append(img.split('/')[-2])
        else:
            labels.append(img.split('\\')[-2])
    # 统计labels
    count = Counter(labels.copy())

    # 标签序列化
    onehot_encoding = F.one_hot(torch.arange(0, len(count.keys()))).tolist()

    # 构建label2encoding字典
    label2encoding = {}
    for i, label in enumerate(count.keys()):
        label2encoding[label] = onehot_encoding[i]
    df = pd.DataFrame(columns=['img_src', 'label', 'one_hot'])
    for index in range(len(images)):
        df.loc[len(df)] = [images[index], labels[index], label2encoding[labels[index]]]

    # 保存csv文件
    df.to_csv('./datasets/'+dataset_name+'.csv')


if __name__ == '__main__':
    gen_csv('PlantDoc')