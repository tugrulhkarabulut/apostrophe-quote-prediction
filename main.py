import os
import argparse

from config import get_cfg_defaults

import lstm
import bert
import t5

modules = {'LSTM': lstm, 'BERT': bert, 'T5': t5}

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='.yml config path',
    )

def main(cfg):
    modules[cfg.MODEL].main(cfg)

if __name__ == '__main__':
    args = parse_arguments()
    cfg = get_cfg_defaults()

    if os.path.exists(args.config):
        cfg.merge_from_file(args.config)