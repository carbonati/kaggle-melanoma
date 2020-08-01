import os
import argparse

from melanoma.utils.generic_utils import load_config_from_yaml
from melanoma.data import preprocess


def main(args):
    config = load_config_from_yaml(args.config_filepath)
    if str(config['version']) == '2018':
        print(f'Preprocessing 2018 ISIC data')
        preprocess.prepare_isic_2018(**config['input'])
    elif str(config['version']) == '2019':
        print(f'Preprocessing 2019 ISIC data')
        preprocess.prepare_isic_2019(**config['input'])
    else:
        raise ValueError(f"Unrecognized version config['version']")
    print('QED.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath',
                        '-f',
                        default='input/preprocess_external_data_config.yaml',
                        type=str,
                        help='Path to external data configuration file.')
    args = parser.parse_args()
    main(args)
