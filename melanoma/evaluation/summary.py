import os
import json
import glob
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import data_utils, model_utils, train_utils
from utils import generic_utils as utils
from evaluation.metrics import compute_auc

METRICS = [
    'roc_auc_score',
    'accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'average_precision_score',
    'specificity_score',
    'fpr_score',
    'pos_rate',
    'log_loss',
]

def get_session_attr(config):
    attr = {
        'cv_folds': [config['input']['cv_folds'].split('/')[-1]],
        'backbone': [config['model']['params']['backbone']],
        'sampler': [config['sampler']['method']],
        'optim': [config['optimizer'].get('method')],
        'initial_lr': [config['optimizer']['params']['lr']],
        'pool': [config['model']['params']['pool_params']],
        'output_net': [config['model']['params'].get('output_net_params')],
        'scheduler': [config['scheduler'].get('method')],
        'scheduler_params': [config['scheduler'].get('params')],
        'criterion': [config['criterion']['method']],
        'criterion_params': [config['criterion'].get('params', {})],
        'transforms': [list(config['augmentations']['transforms'].keys())],
        'tta_val': [config['augmentations']['tta_val']],
        'tta_test': [config['augmentations']['tta_test']],
        'train_only': [config['augmentations'].get('train_only')],
        'batch_size': [config['batch_size']],
        'class_weight': [config.get('class_weight', config['criterion'].get('params', {}).get('class_weight'))],
        'norm_cols': [config['data']['params'].get('norm_cols')],
        'post_norm': [config['augmentations'].get('post_norm')],
        'reg_params': [config['trainer'].get('reg_params')],
        'max_norm': [config['trainer'].get('max_norm')],
        'distributed': [config.get('distributed', False)],
        'fp_16': [config.get('fp_16', False)],
        'opt_level': [config.get('opt_level', None)],
        'num_bags': [config.get('num_bags', 1)],
        'config_filepath': [config.get('config_filepath')],
        'random_state': [config['random_state']]
    }
    return attr


def get_ckpt_dt(model_name):
    dt_str = '-'.join(model_name.strip('/').split('_')[-2:])
    dt = datetime.datetime.strptime(dt_str, '%Y%m%d-%H%M')
    return dt


def compute_summary_scores(fold_dir, metrics=None, group='val'):
    if metrics is None:
        metrics = METRICS
    filepath = os.path.join(fold_dir, f'{group}_predictions.csv')
    if os.path.exists(filepath):
        df_val = pd.read_csv(filepath)
        if 'prediction_raw' not in df_val.columns:
            df_val = df_val.rename(columns={'prediction': 'prediction_raw'})
        scores = train_utils.compute_scores(df_val['target'],
                                            df_val['prediction_raw'],
                                            metrics=metrics)
        df_scores = pd.DataFrame([list(scores.values())], columns=scores.keys())
        df_scores.columns = [f'{group}_{c}' for c in df_scores.columns]
    else:
        df_scores = pd.DataFrame([[None] * len(metrics)],
                                 columns=[f'{group}_{c}' for c in metrics])
    return df_scores


def generate_df_scores(exp_dirs, df_panda=None):
    df_scores = None
    for exp_dir in tqdm(exp_dirs):
        for root, dirs, files in os.walk(exp_dir):
            for model_name in dirs:
                fold_dirs = glob.glob(os.path.join(root, model_name, 'fold_*'))
                try:
                    config = model_utils.load_config(os.path.join(root, model_name))
                    config = utils.cleanup_config(config)
                except:
                    continue
                for fold_dir in fold_dirs:
                    hist_filepath = os.path.join(fold_dir, 'history.csv')
                    if os.path.exists(hist_filepath):
                        df_hist = pd.read_csv(hist_filepath)
                        fold = int(fold_dir.split('_')[-1])

                        df_hist = df_hist.rename(columns={'val_roc_auc_score': 'val_auc_score'})
                        best_step = df_hist['val_auc_score'].idxmax()
                        best_val_auc_step = best_step + 1
                        best_val_loss_step = df_hist['val_loss'].idxmin() + 1

                        df_best = df_hist.loc[best_step].to_frame().T
                        df_best = df_best[['epoch', 'loss', 'auc_score', 'val_loss', 'val_auc_score', 'elapsed_time', 'lr']]

                        df_best['best_val_auc_step'] = best_val_auc_step
                        df_best['best_val_loss_step'] = best_val_loss_step
                        df_best['best_val_loss_step_val_auc'] = df_hist.loc[df_hist['epoch'] == best_val_loss_step, 'val_auc_score'].iloc[0]
                        df_best['best_val_auc_step_val_loss'] = df_hist.loc[df_hist['epoch'] == best_val_auc_step, 'val_loss'].iloc[0]
                        df_best['best_val_auc_step_auc'] = df_hist.loc[df_hist['epoch'] == best_val_auc_step, 'auc_score'].iloc[0]
                        df_best['best_val_auc_step_loss'] = df_hist.loc[df_hist['epoch'] == best_val_auc_step, 'loss'].iloc[0]
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
                            ckpt = ckpt_filepath.split('/')[-1]
                        except:
                            ckpt = None
                        df_best['ckpt'] = ckpt
                        df_best = df_best.reset_index(drop=True)
                        df_best = pd.concat((df_best,
                                             pd.DataFrame(get_session_attr(config))), axis=1)
                        df_val_scores = compute_summary_scores(fold_dir, group='val')
                        df_train_scores = compute_summary_scores(fold_dir, group='train')
                        df_holdout_scores = compute_summary_scores(fold_dir, group='holdout')
                        df_best = pd.concat((df_best, df_val_scores, df_train_scores, df_holdout_scores), axis=1)
                        df_scores = pd.concat((df_scores, df_best), ignore_index=True)

    df_scores['dt'] = df_scores['model_name'].apply(get_ckpt_dt)
    df_scores = df_scores.loc[df_scores['val_auc_score'].sort_values(ascending=False).index]
    df_scores = df_scores.reset_index(drop=True)
    return df_scores
