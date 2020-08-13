import os
import json
import pickle
import datetime
import argparse
import pandas as pd
from itertools import product

from melanoma.utils import generic_utils as utils
from melanoma.utils import data_utils
from melanoma.utils.generic_utils import load_config_from_yaml
from melanoma.data.preprocess import generate_cropped_coords


def main(config):
    """Cropped image generation."""
    config = load_config_from_yaml(args.config_filepath)
    print('Generating and saving cropped image coordinates to disk.')
    generate_cropped_coords(config)
    print('QED.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath',
                        '-f',
                        default='input/generate_cropped_images.yaml',
                        type=str,
                        help='Path to cropped image configuration file.')
    args = parser.parse_args()
    main(args)
