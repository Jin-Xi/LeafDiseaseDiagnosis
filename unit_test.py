from data.plantdoc_dataset import PlantDocDataset
from options.train_options import TrainOptions


def main():
    opt = TrainOptions()
    opt.gather_options()
    dataset = PlantDocDataset(opt)


if __name__ == '__main__':
    main()
    