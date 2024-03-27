from fastai.vision.all import *
import yaml
from pathlib import Path
import argparse

from stages.preparation import get_im_load


def train_model(config_path):

    with open(config_path) as file:
        config = yaml.safe_load(file)

    im_load = get_im_load(config=config)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', dest='config', required=True)
    args = arg_parser.parse_args()

    train_model(config_path=args.config)
