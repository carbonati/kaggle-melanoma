import os
import json
import pickle
import datetime
import argparse
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from melanoma.utils.generic_utils import load_config_from_yaml
from melanoma.utils import data_utils


def main(config):
    """CV fold generation."""
    config = load_config_from_yaml(args.config_filepath)

    # generate output directory to save cv folds
    session_dt = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    if config.get('tag'):
        session_fn = f"cv_folds_{config['tag']}_{session_dt}"
    else:
        session_fn = f"cv_folds_{session_dt}"
    session_dir = os.path.join(config['output']['root'], session_fn)
    if not os.path.exists(session_dir):
        print(f'Generating cv session directory @ {session_dir}')
        os.makedirs(session_dir)

    df_train = data_utils.load_data(**config['input'])
    df_patient = df_train.groupby('patient_id').apply(data_utils.patient_agg_fnc).reset_index()

	# fill mean age
    df_patient = data_utils.fill_na(df_patient, 'age_mean', how='mean')

    preprocess_config = config.get('preprocess', {})
    for k, v in preprocess_config.items():
        df_patient[k] = data_utils.bin_continuous_feature(df_patient[k], **v)
        for v in range(df_patient[k].max().astype(int), df_patient[k].min().astype(int)-1, -1):
            if sum(df_patient[k] == v) < config['cv_folds'].get('num_folds', 10) * 2:
                df_patient.loc[df_patient[k] == v, k] = v - 1

    # compute most frequent site per patient
    if 'anatom_site_general_challenge' in config['cv_folds'].get('stratify_val', []):
        df_anat = data_utils.get_df_agg(df_train,
                                        'anatom_site_general_challenge',
                                        'image_name',
                                        how='idxmax')
        df_patient = pd.merge(df_patient,
                              df_anat[['patient_id', 'anatom_site_general_challenge']],
                              how='left',
                              on='patient_id')
        df_patient = data_utils.fill_na(df_patient, 'anatom_site_general_challenge', how='missing')

    cv_folds = data_utils.generate_cv_folds(df_patient, **config['cv_folds'])
    df_train['fold'] = data_utils.get_fold_col(df_train, cv_folds)
    df_fold = df_train.groupby('fold').apply(data_utils.fold_agg_fnc).reset_index()
    print(df_fold)
    print(f"max-min rate : {df_fold['pos_rate'].max() - df_fold['pos_rate'].min():.6f}")
    print(f"min pos rate : {df_fold['pos_rate'].min():.6f}")
    print(f"max pos rate : {df_fold['pos_rate'].max():.6f}")

    with open(os.path.join(session_dir, 'cv_folds.p'), 'wb') as f:
        pickle.dump(cv_folds, f)
    with open(os.path.join(session_dir, 'config.json'), 'w') as f:
        json.dump(config, f)
    df_train.to_csv(os.path.join(session_dir, 'cv_folds.csv'), index=False)
    print(f'Saved cv folds & config to {session_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath',
                        '-f',
                        default='input/generate_cv_folds_config.yaml',
                        type=str,
                        help='Path to cv generation configuration file.')
    args = parser.parse_args()
    main(args)
