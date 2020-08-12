import os
import sys
import time
import glob
import yaml
import datetime
import shutil
import random
import pprint
import operator
import numpy as np
import torch
import multiprocessing as mp
from functools import reduce
from itertools import product


# generic utils
class Logger(object):
    """Basic logger to record stdout activity"""
    def __init__(self, filename):
        self._filename = filename
        self.terminal = sys.stdout

    def write(self, msg):
        msg += '\n'
        self.terminal.write(msg)
        with open(self._filename, 'a+') as f:
            f.write(msg)

    def carriage(self, msg):
        msg += '\r'
        self.terminal.write(msg)
        with open(self._filename, 'w') as f:
            f.write(msg)
        sys.stdout.flush()


class Tee(object):

    def __init__(self, name, mode='a+'):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
         sys.stdout = self.stdout
         self.file.close()

    def write(self, data):
         self.file.write(data)
         self.stdout.write(data)

    def flush(self):
         self.file.flush()


def set_state(seed=42069):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_config_from_yaml(filepath):
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    print(f'Loaded configurable file from {filepath}')
    pprint.pprint(config)
    return config


def prepare_config(config, args):
    for arg in vars(args):
        v = getattr(args, arg)
        if arg == 'batch_size':
            v = v if v != 0 else None
        if v is not None:
            print(f'Updating parameter `{arg}` to {v}')
            # hotfix
            if arg == 'model':
                config['input']['model'] = v
            else:
                config[arg] = v
    config['distributed'] = config.get('distributed', False)
    return config

def cleanup_config(config):
    # hotfix for old configs
    if not config['model'].get('method'):
        config['model'] = {'method': 'melanoma', 'params': config['model']}
    if not config['data'].get('method'):
        config['data'] = {'method': 'melanoma', 'params': config['data']}
    return config

def get_model_fname(config):
    """
    scheme
    ------
    arch
    img_size
    batch_size
    pool_method
    optimizer
    scheduler
    description
    datetime
    """
    dt_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    img_size = config['input']['images'].strip('/').split('/')[-1].split('x')[0]
    model_fname = config['model']['params']['backbone']
    model_fname += f"_{config['criterion']['method']}"
    model_fname += f'_{img_size}'
    model_fname += f"_{config['batch_size']}"
    if config['model']['params'].get('pool_params'):
        model_fname += f"_{config['model']['params']['pool_params']['method']}"
    model_fname += f"_{config['optimizer']['method']}"
    if config.get('scheduler'):
        model_fname += f"_{config['scheduler']['method']}"
    if config.get('subset'):
        for k, v in config['subset'].items():
            model_fname += f'_{v}'
    if config.get('tags'):
        model_fname += f"_{'_'.join(config['tags'])}"
    model_fname += f"_{dt_str}"
    return model_fname


def cleanup_log_dir(root, min_steps=5, keep_n=5):
    for model_name in os.listdir(root):
        model_dir = os.path.join(root, model_name)
        cleanup_ckpts(model_dir, min_steps, keep_n)


def cleanup_ckpts(model_dir, min_steps=5, keep_n=5):
    if not os.path.exists(os.path.join(model_dir, 'train_history.log')):
        return
    fold_dirs = glob.glob(os.path.join(model_dir, 'fold_*'))
    if len(fold_dirs) > 0:
        for fold_dir in fold_dirs:
            ckpt_files = []
            max_step = 0
            for fn in os.listdir(fold_dir):
                if fn.startswith('ckpt_'):
                    step = int(fn.split('_')[1])
                    if step > max_step:
                        max_step = step
                    ckpt_files.append(fn)
            if max_step < min_steps:
                print(f'removing fold directory {fold_dir}')
                shutil.rmtree(fold_dir)
            else:
                ckpt_files = sorted(ckpt_files, key=lambda x: int(x.split('_')[1]))
                if len(ckpt_files) > keep_n:
                    for fn in ckpt_files[:-keep_n]:
                        print(f'removing checkpoint {os.path.join(fold_dir, fn)}')
                        step = int(fn.split('_')[1])
                        os.remove(os.path.join(fold_dir, fn))
                        if os.path.exists(os.path.join(fold_dir, f'coef_{step:04d}.npy')):
                            os.remove(os.path.join(fold_dir, f'coef_{step:04d}.npy'))
        if not any([os.path.exists(f) for f in fold_dirs]):
            print(f'removing model directory {model_dir}')
            shutil.rmtree(model_dir)
    else:
        print(f'removing model directory {model_dir}')
        shutil.rmtree(model_dir)



def deep_get(dictionary, *keys):
    return reduce(lambda d, key: d.get(key) if d else None, keys, dictionary)


def subset_df(df, conditions):
    return df.loc[reduce(operator.and_, conditions(df), True)]
