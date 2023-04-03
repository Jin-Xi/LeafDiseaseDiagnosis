# Copyright (C) 2023 Xi Jin
# Fake it. Util make it. Start it, Util love it


import os
import sys
from collections import Counter

import torch
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


def gen_csv(dataset_src='PlantVillage', dataset_name='PlantVillage'):
    """通过数据文件夹生成数据csv文件

    Args:
        stc_dataset (str, optional): _description_. Defaults to 'PlantVillage'.
    """
    pwd = os.getcwd() + '/datasets/' + dataset_src
    images = make_dataset(pwd)
    # 取出labels
    labels = []
    for img in images:
        labels.append(img.split('/')[-2])
    # 统计labels
    count = Counter(labels)

    # 标签序列化
    labels = list(count.keys())
    onehot_encoding = torch.nn.functional.one_hot(torch.arange(0, len(count.keys()))).tolist()

    # 构建label2encoding字典
    label2encoding = {}
    for i, label in enumerate(labels):
        label2encoding[label] = onehot_encoding[i]
    df = pd.DataFrame(columns=['img_src', 'label', 'one_hot'])
    for index, (img, label) in enumerate(zip(images, labels)):
        df.loc[index] = [img, label, label2encoding[label]]

    # 保存csv文件
    df.to_csv('./datasets/'+dataset_name+'.csv')


if __name__ == '__main__':
    gen_csv('PlantDoc/train')