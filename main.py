import os
import argparse

from config import get_cfg_defaults

import lstm


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='.yml config path',
    )


def main(cfg):
    

    if cfg.MODEL == 'LSTM':
        lstm.main()

if __name__ == '__main__':
    args = parse_arguments()
    cfg = get_cfg_defaults()

    if os.path.exists(args.config):
        cfg.merge_from_file(args.config)