import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler
warnings.simplefilter("ignore", UserWarning)

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel, convert_syncbn_model
except:
    pass

from melanoma.utils import generic_utils as utils
from melanoma.utils import train_utils, model_utils, data_utils
from melanoma.data.dataset import MelanomaDataset
from melanoma.core.trainer import Trainer
from melanoma.evaluation.postprocess import generate_df_pred, log_model_summary
import melanoma.config as melanoma_config


def evaluate(config):
    """Run a melanoma training session."""
    model_dir = config['input']['model']
    output_dir = config['output']['predictions']
    train_config = model_utils.load_config(model_dir)

    img_version = train_config['input']['images'].strip('/').split('/')[-1]
    model_name = model_dir.strip('/').split('/')[-1]
    output_dir = os.path.join(output_dir, model_name)
    if not os.path.exists(output_dir):
        print('Generating output directory {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    # log activity from the training session to a logfile
    # sys.stdout = utils.Tee(os.path.join(model_dir, 'eval_history.log'))
    utils.set_state(config['random_state'])
    device_ids = config.get('device_ids', [0])
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

	# read in the training table and generate cv folds
    df_mela = data_utils.load_data(train_config['input']['train'],
                                   duplicate_path=train_config['input']['duplicates'],
                                   cv_folds_dir=train_config['input']['cv_folds'],
                                   keep_prob=config.get('keep_prob', 1.),
                                   random_state=config['random_state'])


    fold_ids = config.get(
        'fold_ids',
        [int(fn.split('_')[1]) for fn in os.listdir(model_dir) if fn.startswith('fold')]
    )
    fold_ids = fold_ids if isinstance(fold_ids, list) else [fold_ids]

    if config['eval_test'] and config['input'].get('test'):
        df_test = data_utils.load_data(config['input']['test'],
                                       keep_prob=config.get('keep_prob', 1.),
                                       random_state=config['random_state'])
    else:
        df_test = None

    # begin training session
    for fold_id in fold_ids:
        if isinstance(fold_id, str) and fold_id.isdigit():
            fold_id = int(fold_id)
        ckpt_dir = os.path.join(model_dir, f'fold_{fold_id}')
        output_fold_dir = os.path.join(output_dir, f'fold_{fold_id}')
        if not os.path.exists(output_fold_dir):
            print(f'Generating fold directory {output_fold_dir}')
            os.makedirs(output_fold_dir)

        if isinstance(device_ids, list):
            visible_devices = ','.join([str(x) for x in device_ids])
        else:
            visible_devices = device_ids
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
        num_workers = config['num_workers']

        # instantiate model
        model = model_utils.load_model(ckpt_dir, step='best')
        model = model.cuda()
        model = nn.DataParallel(model, device_ids).to(device)

        trainer = Trainer(model,
                          device_ids=device_ids,
                          **train_config['trainer'])

        # augmentors
        if 'normalize' in config['augmentations'].get('transforms', {}):
            img_stats = data_utils.load_img_stats(os.path.join(train_config['input']['cv_folds'], img_version),
                                                  fold_id)
            config['augmentations']['transforms']['normalize'] = img_stats

        _, val_aug, test_aug = train_utils.get_augmentors(
            config['augmentations'].get('transforms'),
            norm_cols=train_config['data'].get('norm_cols'),
            tta_val=config['augmentations'].get('tta_val', False),
            tta_test=config['augmentations'].get('tta_test', True),
        )

        if config.get('eval_val', False):
            df_val = df_mela.loc[df_mela['fold'] == fold_id].reset_index(drop=True)
            val_ds = MelanomaDataset(os.path.join(train_config['input']['images'], 'train'),
                                     df_val,
                                     augmentor=val_aug,
                                     **train_config['data'])
            val_sampler = train_utils.get_sampler(val_ds, method='sequential',)
            val_dl = DataLoader(val_ds,
                                batch_size=config['batch_size'],
                                sampler=SequentialSampler(val_ds),
                                num_workers=config['num_workers'])
        else:
            val_dl = None

        if df_test is not None:
            # use the same augmentor as the validation set
            test_ds = MelanomaDataset(os.path.join(train_config['input']['images'], 'test'),
                                      df_test,
                                      target_col=None,
                                      augmentor=test_aug,
                                      **train_config['data'])
            test_sampler = train_utils.get_sampler(test_ds, method='sequential')
            test_dl = DataLoader(test_ds,
                                 batch_size=config['batch_size'],
                                 sampler=test_sampler,
                                 num_workers=config['num_workers'])
        else:
            test_dl = None

        num_bags = config.get('num_bags', 1)
        if val_dl is not None:
            print(f'\nGenerating {num_bags} validation prediction(s).')
            df_pred_val = generate_df_pred(trainer,
                                           val_dl,
                                           y_true=val_dl.dataset.get_labels(),
                                           df_mela=df_val,
                                       num_bags=num_bags)
            df_pred_val.to_csv(os.path.join(output_fold_dir, 'val_predictions.csv'),
                               index=False)
            log_model_summary(df_pred_val, logger=trainer.logger, group='val')

        if test_dl is not None:
            print(f'\nGenerating {num_bags} test prediction(s).')
            df_pred_test = generate_df_pred(trainer,
                                            test_dl,
                                            df_mela=df_test,
                                            num_bags=num_bags)
            df_pred_test.to_csv(os.path.join(output_fold_dir, 'test_predictions.csv'),
                                index=False)

        print(f'Saved predictions to {output_fold_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath',
                        '-f',
                        type=str,
                        help='Path to configurable file.')
    parser.add_argument('--local_rank', '-r', default=0, type=int)
    parser.add_argument('--distributed', '-d', default=False, action="store_true")
    parser.add_argument('--fp_16', '-fp16', default=False, action="store_true")
    parser.add_argument('--keep_prob', '-p', default=None, type=float)
    parser.add_argument('--batch_size', '-bs', default=None, type=int)
    parser.add_argument('--experiment_name', '-e', default=None, type=str)
    parser.add_argument('--num_workers', '-w', default=None, type=int)
    parser.add_argument('--fold_ids', default=None, type=list)
    parser.add_argument('--num_gpus', default=1, type=int)
    args = parser.parse_args()

    config = utils.load_config_from_yaml(args.config_filepath)
    config = utils.prepare_config(config, args)

    evaluate(config)

