import argparse
import logging
import math

import coloredlogs


def main_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Game input', type=str)
    parser.add_argument('--num_gpus', help='If set, use GPUs', type=int, default=0)

    parser.add_argument(
            '--minimal_nw', help='If set, use minimal network for training', action='store_true', dest='minimal_nw'
    )
    parser.set_defaults(minimal_nw=False)

    parser.add_argument('--num_sim', help='Set number of simulations in chess self-play', type=int, default=int(5e5))
    parser.add_argument('--num_officers', help='Set number of workers in chess self-play', type=int, default=1)
    return parser.parse_args()


def init_logger(filename, mode='a'):
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(filename=filename, mode=mode)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.
    return log


def manage_gpus(num_gpus, config, log_in_tensorboard):
    if 0 < num_gpus:
        num_gpus_per_worker = num_gpus / (
                config.train_on_gpu
                + config.num_workers * config.selfplay_on_gpu
                + log_in_tensorboard * config.selfplay_on_gpu
                + config.use_last_model_value * config.reanalyse_on_gpu
        )
        if 1 < num_gpus_per_worker:
            num_gpus_per_worker = math.floor(num_gpus_per_worker)
    else:
        num_gpus_per_worker = 0
    return num_gpus_per_worker


class DotDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError
