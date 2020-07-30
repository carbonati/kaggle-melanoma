import os
import json
import glob
import numpy as np
import pandas as pd

from utils import data_utils
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
        'class_weight': [config.get('class_weight', False)],
        'fp_16': [config.get('fp_16', False)],
        'sample_random': [config['data'].get('sample_random', False)],
        'random_state': [config['random_state']]
    }
    return attr


def generate_df_scores(exp_dirs, df_panda=None):
    df_scores = None
    for exp_dir in exp_dirs:
        for root, dirs, files in os.walk(exp_dir):
            for model_name in dirs:
                fold_dirs = glob.glob(os.path.join(root, model_name, 'fold_*'))
                try:
                    config = data_utils.load_config(os.path.join(root, model_name))
                except:
                    continue
                for fold_dir in fold_dirs:
                    hist_filepath = os.path.join(fold_dir, 'history.csv')
                    if os.path.exists(hist_filepath):
                        df_hist = pd.read_csv(hist_filepath)
                        best_step = df_hist['val_auc_score'].idxmax()
                        step = best_step + 1
                        fold = int(fold_dir.split('_')[-1])

                        df_best = df_hist.loc[best_step].to_frame().T
                        df_best['step'] = step
                        df_best['experiment'] = exp_dir.split('/')[-1]
                        df_best['model_name'] = model_name
                        df_best['fold'] = fold

                        try:
                            ckpt_filepath = glob.glob(
                                os.path.join(
                                    root,
                                    model_name,
                                    f'fold_{fold}',
                                    f'ckpt_{step:04d}_*'
                                )
                            )[0]
                        except:
                            continue
                        df_best['ckpt'] = ckpt_filepath.split('/')[-1]
                        df_best = df_best.reset_index(drop=True)
                        df_best = pd.concat((df_best,
                                             pd.DataFrame(get_session_attr(config))), axis=1)
                        df_scores = pd.concat((df_scores, df_best), ignore_index=True)

    df_scores = df_scores.loc[df_scores['val_auc_score'].sort_values(ascending=False).index]
	df_scores = df_scores.reset_index(drop=True)
    return df_scores
