import argparse

def args_parser():
    """ load hyperparameters"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, default='run0', help="name of current run"
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="learning_rate",
    )
    parser.add_argument(
        "--batch", type=int, default=256, help="batch size"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="num layers"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden_size"
    )
    parser.add_argument(
        "--batch_norm", type=int, default=1, help="whether to conduct batch norm"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="drop_out ratio"
    )
    parser.add_argument(
        "--optimizer", type=str, default='SGD', help="type of optimizer"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="patience of early stopping"
    )
    parser.add_argument(
        "--epoch", type=int, default=10, help="training epochs"
    )
    args = parser.parse_args()

    return args
