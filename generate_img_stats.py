import os
import json
import pickle
import datetime
import argparse
import pandas as pd

from melanoma.utils import data_utils
from melanoma.utils.generic_utils import load_config_from_yaml
from melanoma.data.preprocess import generate_img_stats


def main(config):
    """Image normalization stats generation."""
    config = load_config_from_yaml(args.config_filepath)

    cv_folds_dir = config['input']['cv_folds']
    image_dir = config['input']['image_dir']
    meta_dir = os.path.join(image_dir, 'metadata')
    image_name = image_dir.strip('/').split('/')[-1]
    output_dir = os.path.join(cv_folds_dir, image_name)
    if not os.path.exists(output_dir):
        print(f'Generating output directory {output_dir}')
        os.makedirs(output_dir)

    stratify = config['data'].get('stratify')
    df_mela = data_utils.load_data(config['input']['train'],
                                   duplicate_path=config['input'].get('duplicates'),
                                   cv_folds_dir=config['input'].get('cv_folds'))


    fold_ids = list(set(df_mela['fold'].tolist()))
    for fold in fold_ids:
        fold_dir = os.path.join(output_dir, f'fold_{fold}')
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        df_train = df_mela.loc[(df_mela['fold'] != fold) &
                               (df_mela['fold'] != 'test')].reset_index(drop=True)

        fold_stats = {}
        if stratify is not None:
            for col in stratify:
                fold_stats[col] = {}
                df_train = data_utils.fill_na(df_train, col, how='missing')
                values = df_train[col].unique()
                for v in values:
                    if v == 'missing':
                        # use all samples to compute image statistics for missing values
                        df_value = df_train
                    else:
                        df_value = df_train.loc[df_train[col] == v]
                    fold_stats[col][v] = generate_img_stats(df_value,
                                                            meta_dir,
                                                            name=f'fold {fold} ({col} == {v})')
        fold_stats_full = generate_img_stats(df_train,
                                             meta_dir,
                                             name=f'fold {fold}')
        fold_stats = dict(fold_stats, **fold_stats_full)

        with open(os.path.join(fold_dir, 'img_stats.json'), 'w') as f:
            json.dump(str(fold_stats), f)
        print(f'Saved to fold {fold} img stats to {fold_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath',
                        '-f',
                        default='input/generate_img_stats_config.yaml',
                        type=str,
                        help='Path to image statistics configuration file.')
    args = parser.parse_args()
    main(args)
