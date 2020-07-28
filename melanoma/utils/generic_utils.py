import os
import sys
import time
import glob
import yaml
import datetime
import shutil
import random
import pprint
import numpy as np
import torch
import multiprocessing as mp


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
        if v is not None:
            print(f'Updating parameter `{arg}` to {v}')
            config[arg] = v
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
    model_fname = config['model']['backbone']
    model_fname += f"_{config['criterion']['method']}"
    model_fname += f'_{img_size}'
    model_fname += f"_{config['batch_size']}"
    if config['model'].get('pool_params'):
        model_fname += f"_{config['model']['pool_params']['method']}"
    model_fname += f"_{config['optimizer']['method']}"
    if config.get('scheduler'):
        model_fname += f"_{config['scheduler']['method']}"
    if config.get('subset'):
        for k, v in config['subset'].items():
            model_fname += f'_{v}'
    if config.get('tags'):
        model_fname += f"_{'_'.join(tags)}"
    model_fname += f"_{dt_str}"
    return model_fname

