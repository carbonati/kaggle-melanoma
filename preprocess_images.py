import os
import argparse

from melanoma.utils.generic_utils import load_config_from_yaml
from melanoma.data.preprocess import preprocess_images


def main(args):
    config = load_config_from_yaml(args.config_filepath)
    preprocess_images(**config['input'],
                      **config['output'],
                      **config['data'],
                      **config.get('default', {}))
    print('QED.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath',
                        '-f',
                        default='input/preprocess_images_config.yaml',
                        type=str,
                        help='Path to image preprocessing configuration file.')
    args = parser.parse_args()

    main(args)
