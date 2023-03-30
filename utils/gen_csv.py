# Copyright (C) 2023 Xi Jin
# Fake it. Util make it. Start it, Util love it


import os
from collections import Counter

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from data.image_folder import IMG_EXTENSIONS
from data.image_folder import is_image_file, make_dataset


def gen_csv(dataset_src='PlantVillage'):
    """通过数据文件夹生成数据csv文件

    Args:
        stc_dataset (str, optional): _description_. Defaults to 'PlantVillage'.
    """
    pwd = os.getcwd() + '\\datasets\\' + dataset_src
    images = make_dataset(pwd)
    # 取出labels
    labels = []
    for img in images:
        labels.append(img.split('\\')[-2])
    # 统计labels
    count = Counter(labels)
    # TODO:标签序列化器初始化
    label_binarizer = MultiLabelBinarizer()
    onehot_labels = label_binarizer.fit_transform(list(count.keys()))
    
    # 构建一个标签到onehot的映射
    labels2array = {}
    for label, arr in zip(label_binarizer.classes_, onehot_labels):
        labels2array[label] = arr.tolist()
    n_classes = len(label_binarizer.classes_)


if __name__ == '__main__':
    gen_csv()