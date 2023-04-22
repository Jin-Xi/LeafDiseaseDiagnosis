# This code's architecture inspire by CycleGAN.
# Copyright (c) 2023 Xijin.

"""This package includes all the modules related to data loading and preprocessing
 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import importlib

import torch.utils.data
from data.base_dataset import BaseDataset
from data.image_folder import ImageFolder 


def find_dataset_using_name(dataset_name):
    """
    
    """
    """通过类名导入"data/[dataset_name]_dataset.py"，并实例化。
    在本函数中DatasetNameDataset()的数据集对象会被实例化，
    它是BaseDataset的子类，并且该子类数据集名字不区分大小写

    Patameters:
        dataset_name: str, 数据集名

    Raises:
        NotImplementedError: 该数据集类没有继承于BaseDataset，或者没有该类，类名不能带有_，不区分大小写

    Returns:
        _type_: BaseDataset() 实例
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        # 如果类名匹配，并且该类是BaseDataset的子类
        if name.lower() == target_dataset_name.lower() \
           and (issubclass(cls, BaseDataset) or issubclass(cls, ImageFolder)):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("在 %s.py, 应该有一个继承于BaseDataset并且和小写名称的 %s 匹配的子类." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt):
    """Create a dataset given the option.
    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'
    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt)
    dataset = data_loader.load_data()
    return dataset


class CustomDatasetDataLoader:
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class
        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads)
        )

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data