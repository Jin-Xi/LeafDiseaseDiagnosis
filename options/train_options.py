# Copyright (C) 2023 Xi Jin
# Fake it. Util make it. Start it, Util love it


from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # TODO:完成多折交叉验证训练