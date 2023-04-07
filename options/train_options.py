# Copyright (C) 2023 Xi Jin
# Fake it. Util make it. Start it, Util love it


from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def __init__(self):
        super().__init__()
        self.isTrain = None

    def initialize(self, parser=None):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--is_train', required=False, help='锻炼模式！！')
        parser.add_argument('--optimizer_name', required=False, help='优化器: Adam | SGD | ...')
        parser.add_argument('--lr_policy', required=False, help='学习率更新策略:step | plateau | cosine')
        parser.add_argument('--continue_train', required=False, help='继续上一次的训练')
        parser.add_argument('--load_iters', required=True, default=0, help='继续上一次的训练')
        parser.add_argument('--epoch', required=False, default=100, help='继续上一次的训练 epoch')
        parser.add_argument('--learning_rate', required=False, default=0.1, help='学习率')
        parser.add_argument('--learning_rate', required=False, default=0.1, help='学习率')

        self.isTrain = True
        return parser