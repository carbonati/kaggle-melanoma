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
from melanoma.data.preprocess import generate_img_stats


def main(config):
    """Image normalization stats generation."""
    config = load_config_from_yaml(args.config_filepath)

    cv_folds_dir = config['input']['cv_folds']
    image_dir = config['input']['image_dir']
    image_name = image_dir.strip('/').split('/')[-1]
    output_dir = os.path.join(cv_folds_dir, image_name)
    if not os.path.exists(output_dir):
        print(f'Generating output directory {output_dir}')
        os.makedirs(output_dir)

    stratify = config['data'].get('stratify')
    min_samples = config['data'].get('min_samples', 50)
    df_mela = data_utils.load_data(config['input']['train'],
                                   duplicate_path=config['input'].get('duplicates'),
                                   cv_folds_dir=config['input'].get('cv_folds'),
                                   external_filepaths=config['input'].get('external_filepaths'),
                                   keep_prob=config['default'].get('keep_prob', 1))
    df_mela['image_dir'] = df_mela['source'].map(config['input']['image_map'])
    df_mela['meta_dir'] = df_mela['image_dir'].apply(lambda x: os.path.join(x, 'metadata'))

    fold_ids = config['default'].get('fold_ids', list(set(df_mela['fold'].tolist())))
    for fold in fold_ids:
        fold_dir = os.path.join(output_dir, f'fold_{fold}')
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)
        df_train = df_mela.loc[(df_mela['fold'] != fold) &
                               (df_mela['fold'] != 'test')].reset_index(drop=True)

        fold_stats = {}
        if stratify is not None:
            groups = list(product(*[list(df_train[col].unique()) for col in stratify]))
            for group in groups:
                n = len(group)
                for i in range(n):
                    col = stratify[i]
                    v = group[i]
                    if col not in fold_stats.keys():
                        fold_stats[col] = {}
                    if v not in fold_stats[col].keys():
                        df_train = data_utils.fill_na(df_train, col, how='unknown')
                        if v in ['missing', 'unknown']:
                            # use all samples to compute image statistics for missing values
                            df_value = df_train
                        else:
                            df_value = df_train.loc[df_train[col] == v]
                            if len(df_value) < min_samples:
                                df_value = df_train
                        fold_stats[col][v] = generate_img_stats(df_value,
                                                                name=f'fold {fold} = ({col} == {v})')
                    for j in range(i+1, n):
                        # subset the train data to samples with multiple conditions
                        # need to handle when a value is missing/unknown
                        sub_cols = stratify[i:j+1]
                        sub_group = group[i:j+1]
                        conditions = lambda df: [
                            df[sub_cols[i]] == sub_group[i]
                            for i
                            in range(len(sub_cols))
                            if sub_group[i] not in ['missing', 'unknown']
                        ]
                        df_sub = utils.subset_df(df_train, conditions)
                        if len(df_sub) < min_samples:
                            df_sub = df_train
                        # add the image stats for the subset to our dict
                        sub_dict = utils.deep_get(fold_stats, *[v for l in zip(sub_cols[:j], sub_group[:j]) for v in l])
                        if sub_cols[j] not in sub_dict.keys():
                            sub_dict[sub_cols[j]] = {}
                        name = f'fold {fold} - ' + ' & '.join([f'({sub_cols[i]} == {sub_group[i]})' for i in range(len(sub_cols))])
                        sub_dict[sub_cols[j]][sub_group[j]] = generate_img_stats(
                            df_sub,
                            name=name
                        )

        fold_stats_full = generate_img_stats(df_train,
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
