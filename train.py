import os
import sys
import glob
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import time
import argparse
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, confusion_matrix
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


def train(config):
    """Run a melanoma training session."""
    # clean up the model directory and generate a new output path for the training session
    log_dir = config['input']['models']
    if config.get('experiment_name'):
        log_dir = os.path.join(log_dir, config['experiment_name'])
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    # clean up model directory
    # for exp_dir in glob.glob(os.path.join(config['input']['models'], '*exp*')):
    #     utils.cleanup_ckpts(log_dir)
    # model output configurations
    img_version = config['input']['images'].strip('/').split('/')[-1]
    model_fname = utils.get_model_fname(config)
    model_dir = os.path.join(log_dir, model_fname)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    # log activity from the training session to a logfile
    if config['local_rank'] == 0:
        sys.stdout = utils.Tee(os.path.join(model_dir, 'train_history.log'))
    utils.set_state(config['random_state'])
    device_ids = config.get('device_ids', [0])
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

	# read in the training table and generate cv folds
    df_mela = data_utils.load_data(config['input']['train'],
                                   duplicate_path=config['input']['duplicates'],
                                   cv_folds_dir=config['input']['cv_folds'],
                                   keep_prob=config.get('keep_prob', 1.))


    fold_ids = config.get('fold_ids', list(set(df_mela['fold'].tolist())))
    fold_ids = fold_ids if isinstance(fold_ids, list) else [fold_ids]

    if config['input'].get('test'):
        df_test = data_utils.load_data(config['input']['test'])
    else:
        df_test = None

    # begin training session
    for fold_id in fold_ids:
        fold_id = int(fold_id) if fold_id.isdigit() else fold_id
        # generate a checkpoint directory to save weights for a given fold
        print(f"{'-'*80}\nFold {fold_id}\n{'-'*80}")
        ckpt_dir = os.path.join(model_dir, f'fold_{fold_id}')
        print(f'Checkpoint path {ckpt_dir}\n')
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        df_train = df_mela.loc[(df_mela['fold'] != fold_id) &
                                (df_mela['fold'] != 'test')].reset_index(drop=True)
        df_val = df_mela.loc[df_mela['fold'] == fold_id].reset_index(drop=True)

        if config['distributed']:
            torch.cuda.set_device(config['local_rank'])
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://'
            )
            num_workers = config['num_workers'] // config['num_gpus']
        else:
            if isinstance(device_ids, list):
                visible_devices = ','.join([str(x) for x in device_ids])
            else:
                visible_devices = device_ids
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
            num_workers = config['num_workers']

        # instantiate model
        model = model_utils.get_model(**config['model'])
        if config['input'].get('pretrained'):
            if config['local_rank'] == 0:
                print(f"Loading pretrained weights from {config['input']['pretrained']}")
                state_dict = torch.load(config['input']['pretrained'])
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                model.load_state_dict(state_dict)
        model = model.cuda()

        if config['distributed']:
            model = convert_syncbn_model(model)
            model = DistributedDataParallel(model, delay_allreduce=True)
        else:
            model = nn.DataParallel(model, device_ids).to(device)

        # data utils
        if 'normalize' in config['augmentations'].get('transforms', {}):
            img_stats = data_utils.load_img_stats(os.path.join(config['input']['cv_folds'], img_version),
                                                  fold_id)
            config['augmentations']['transforms']['normalize'] = img_stats

        print(img_stats)
        # tile & image augmentors
        train_aug, val_aug, test_aug = train_utils.get_augmentors(
            config['augmentations'].get('transforms'),
            norm_cols=config['data'].get('norm_cols'),
            tta_val=config['augmentations'].get('tta_val', False),
            tta_test=config['augmentations'].get('tta_test', False),
            fp_16=False
        )
            #fp_16=config.get('fp_16'))

        train_ds = MelanomaDataset(os.path.join(config['input']['images'], 'train'),
                                   df_train,
                                   augmentor=train_aug,
                                   fp_16=False,
                                   #fp_16=config.get('fp_16'),
                                   **config['data'])
        val_ds = MelanomaDataset(os.path.join(config['input']['images'], 'train'),
                              df_val,
                              augmentor=val_aug,
                              fp_16=False)
                              #fp_16=config.get('fp_16'),

        train_sampler = train_utils.get_sampler(train_ds,
                                                distributed=config['distributed'],
                                                batch_size=config['batch_size'] * config['num_gpus'],
                                                random_state=config['random_state'],
                                                method=config['sampler']['method'],
                                                params=config['sampler'].get('params', {}))
        val_sampler = train_utils.get_sampler(val_ds,
                                              method='sequential',
                                              distributed=config['distributed'])

        train_dl = DataLoader(train_ds,
                              batch_size=config['batch_size'],
                              sampler=train_sampler,
                              num_workers=num_workers,
                              drop_last=True)
        val_dl = DataLoader(val_ds,
                            batch_size=config.get('val_batch_size', config['batch_size']),
                            sampler=SequentialSampler(val_ds),
                            num_workers=config['num_workers'])

        if config.get('test_batch_size'):
            eval_ds = MelanomaDataset(os.path.join(config['input']['images'], 'tiles'),
                                   df_val,
                                   augmentor=test_aug)
            eval_dl = DataLoader(eval_ds,
                                 batch_size=config['test_batch_size'],
                                 sampler=SequentialSampler(eval_ds),
                                 num_workers=config['num_workers'])
        else:
            eval_dl = val_dl

        if df_test is not None:
            # use the same augmentor as the validation set
            test_ds = MelanomaDataset(os.path.join(config['input']['images'], 'test'),
                                   df_test,
                                   augmentor=test_aug,
                                   **config['data'])
            test_sampler = train_utils.get_sampler(test_ds, method='sequential')
            test_dl = DataLoader(test_ds,
                                 batch_size=config.get('test_batch_size', config['val_batch_size']),
                                 sampler=test_sampler,
                                 num_workers=config['num_workers'])
        else:
            test_dl = None

        optim = train_utils.get_optimizer(model=model, **config['optimizer'])
        sched = train_utils.get_scheduler(config, optim,
                                          steps_per_epoch=len(train_dl))
        criterion = train_utils.get_criterion(config)

        if config.get('fp_16'):
            model, optim = amp.initialize(model,
                                          optim,
                                          opt_level=config.get('opt_level', 'O2'),
                                          loss_scale='dynamic',
                                          keep_batchnorm_fp32=config.get('keep_batchnorm_fp32', True))

        # training session for a given fold
        trainer = Trainer(model,
                          optim,
                          criterion=criterion,
                          scheduler=sched,
                          ckpt_dir=ckpt_dir,
                          device_ids=device_ids,
                          fp_16=config.get('fp_16', False),
                          rank=config.get('local_rank', 0),
                          **config['trainer'])
        trainer.fit(train_dl, config['steps'], val_dl)

        # generate predictions table from the best step model
        if config['local_rank'] == 0:
            df_pred_val = generate_df_pred(trainer,
                                           eval_dl,
                                           df_val,
                                           num_bags=config.get('num_bags'))
            df_pred_val.to_csv(os.path.join(ckpt_dir, 'val_predictions.csv'), index=False)
            if postprocessor is not None:
                print('Updating postprocessor val predictions')
                postprocessor.fit(df_pred_val['prediction_raw'], df_pred_val[config['target_col']])
            log_model_summary(df_pred_val, logger=trainer.logger, group='val')

            if test_dl is not None:
                df_pred_test = generate_df_pred(trainer,
                                                test_dl,
                                                df_test,
                                                num_bags=config.get('num_bags'))
            df_pred_test.to_csv(os.path.join(ckpt_dir, 'test_predictions.csv'), index=False)
            log_model_summary(df_pred_test, logger=trainer.logger, group='test')

            if postprocessor is not None:
                np.save(os.path.join(ckpt_dir, 'coef.npy'), postprocessor.get_coef())
            print(f'Saved output to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath',
                        '-f',
                        type=str,
                        help='Path to configurable file.')
    parser.add_argument('--local_rank', '-r', default=0, type=int)
    parser.add_argument('--distributed', '-d', default=False, action="store_true")
    parser.add_argument('--fp_16', '-fp16', default=False, action="store_true")
    parser.add_argument('--steps', '-s', default=None, type=int)
    parser.add_argument('--keep_prob', '-p', default=None, type=float)
    parser.add_argument('--batch_size', '-bs', default=None, type=int)
    parser.add_argument('--experiment_name', '-e', default=None, type=str)
    parser.add_argument('--num_workers', '-w', default=None, type=int)
    parser.add_argument('--fold_ids', default=None, type=list)
    parser.add_argument('--num_gpus', default=1, type=int)
    args = parser.parse_args()

    config = utils.load_config_from_yaml(args.config_filepath)
    config = utils.prepare_config(config, args)

    train(config)

