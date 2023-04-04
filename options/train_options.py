# Copyright (C) 2023 Xi Jin
# Fake it. Util make it. Start it, Util love it


from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser=None):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--is_train', required=True, help='锻炼模式！！')

        self.isTrain = True
        return parser