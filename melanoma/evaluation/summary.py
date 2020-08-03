import os
import json
import glob
import datetime
import numpy as np
import pandas as pd

from utils import data_utils, model_utils
from evaluation.metrics import compute_auc


def get_session_attr(config):
    attr = {
        'cv_folds': [config['input']['cv_folds'].split('/')[-1]],
        'backbone': [config['model']['backbone']],
        'sampler': [config['sampler']['method']],
        'optim': [config['optimizer'].get('method')],
        'lr': [config['optimizer']['params']['lr']],
        'scheduler': [config['scheduler'].get('method')],
        'scheduler_params': [config['scheduler'].get('params')],
        'criterion': [config['criterion']['method']],
        'criterion_params': [config['criterion'].get('params', {})],
        'batch_size': [config['batch_size']],
        'class_weight': [config.get('class_weight', config['criterion'].get('sample_weight', False))],
        'norm_cols': [config['data'].get('norm_cols')],
        'max_norm': [config['trainer'].get('max_norm')],
        'fp_16': [config.get('fp_16', False)],
        'random_state': [config['random_state']]
    }
    return attr


def get_ckpt_dt(model_name):
    dt_str = '-'.join(model_name.strip('/').split('_')[-2:])
    dt = datetime.datetime.strptime(dt_str, '%Y%m%d-%H%M')
    return dt


def generate_df_scores(exp_dirs, df_panda=None):
    df_scores = None
    for exp_dir in exp_dirs:
        for root, dirs, files in os.walk(exp_dir):
            for model_name in dirs:
                fold_dirs = glob.glob(os.path.join(root, model_name, 'fold_*'))
                try:
                    config = model_utils.load_config(os.path.join(root, model_name))
                except:
                    continue
                for fold_dir in fold_dirs:
                    hist_filepath = os.path.join(fold_dir, 'history.csv')
                    if os.path.exists(hist_filepath):
                        df_hist = pd.read_csv(hist_filepath)
                        fold = int(fold_dir.split('_')[-1])
                        best_step = df_hist['val_auc_score'].idxmax()
                        best_val_auc_step = best_step + 1
                        best_val_loss_step = df_hist['val_loss'].idxmin() + 1

                        df_best = df_hist.loc[best_step].to_frame().T
                        df_best['best_val_auc_step'] = best_val_auc_step
                        df_best['best_val_loss_step'] = best_val_loss_step
                        df_best['experiment'] = exp_dir.split('/')[-1]
                        df_best['model_name'] = model_name
                        df_best['fold'] = fold

                        try:
                            ckpt_filepath = glob.glob(
                                os.path.join(
                                    root,
                                    model_name,
                                    f'fold_{fold}',
                                    f'ckpt_{best_val_auc_step:04d}_*'
                                )
                            )[0]
                        except:
                            continue
                        df_best['ckpt'] = ckpt_filepath.split('/')[-1]
                        df_best = df_best.reset_index(drop=True)
                        df_best = pd.concat((df_best,
                                             pd.DataFrame(get_session_attr(config))), axis=1)
                        df_scores = pd.concat((df_scores, df_best), ignore_index=True)

    df_scores['dt'] = df_scores['model_name'].apply(get_ckpt_dt)
    df_scores = df_scores.loc[df_scores['val_auc_score'].sort_values(ascending=False).index]
    df_scores = df_scores.reset_index(drop=True)
    return df_scores
