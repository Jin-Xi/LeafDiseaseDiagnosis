from data import create_dataset
from models import create_model
from options.train_options import TrainOptions


def main():
    opt = TrainOptions()
    opt.gather_options()
    model = create_model(opt)
    dataset = create_dataset(opt)


if __name__ == '__main__':
    main()
    