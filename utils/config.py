import os
import socket
import torch
import random
import numpy as np
from datetime import datetime
from pathlib import Path

from .filesystem import (
    load_pickle, save_pickle, commit, save_json, load_json)


PROJ_DIR = Path('.')
DATA_DIR = PROJ_DIR / 'DATA'
RUNS_DIR = PROJ_DIR / 'RUNS'


DEFAULTS = {
    # general
    'title': 'Molecule Generator',
    'description': 'An RNN-based Molecule Generator',
    'log_dir': RUNS_DIR.as_posix(),
    'random_seed': 42,
    'use_gpu': False,
    # data
    'batch_size': 32,
    'shuffle': True,
    'use_mask': True,
    # model
    'embed_size': 256,
    'embed_window': 3,
    'mask_freq': 10,
    'num_clusters': 30,
    'hidden_size': 64,
    'hidden_layers': 2,
    'dropout': 0.3,
    'latent_size': 100,
    # learning
    'num_epochs': 10,
    'optim_lr': 0.001,
    'use_scheduler': True,
    'sched_step_size': 2,
    'sched_gamma': 0.9,
    'clip_norm': 5.0,
    # sampler
    'load_last': False,
    'validate_after': 0.3,
    'validation_samples': 300,
    'num_samples': 100,
    'max_length': 10,
    'temperature': 0.8,
    'reproduce': False,
    'sampling_seed': None
}


def set_random_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 2**32-1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return seed


def get_run_info(name):
    start_time = datetime.now().strftime('%Y-%m-%d@%X')
    host_name = socket.gethostname()
    run_name = f'{start_time}-{host_name}-{name}'
    return run_name, host_name, start_time


def get_dataset_info(name):
    path = PROJ_DIR / 'utils/data' / f'{name}.json'
    return load_json(path)


def get_text_summary(params):
    start_time = params.get('start_time')
    tag = (f"Experiment params: {params.get('title')}\n")

    text = f"<h3>{tag}</h3>\n"
    text += f"{params.get('description')}\n"

    text += '<pre>'
    text += f"Start Time: {start_time}\n"
    text += f"Host Name: {params.get('host_name')}\n"
    text += f'CWD: {os.getcwd()}\n'
    text += f'PID: {os.getpid()}\n'
    text += f"Commit Hash: {params.get('commit_hash')}\n"
    text += f"Random Seed: {params.get('random_seed')}\n"
    text += '</pre>\n<pre>'

    skip_keys = ['title', 'description', 'random_seed', 'run_name']
    for key, val in params.items():
        if key in skip_keys:
            continue
        text += f'{key}: {val}\n'
    text += '</pre>'

    return tag, text


def create_folder_structure(root, run_name, data_path):
    paths = {'data': data_path}

    paths['run'] = root / run_name
    if not os.path.exists(paths['run']):
        os.makedirs(paths['run'])

    paths['ckpt'] = paths['run'] / 'ckpt'
    if not os.path.exists(paths['ckpt']):
        os.makedirs(paths['ckpt'])

    paths['config'] = paths['run'] / 'config'
    if not os.path.exists(paths['config']):
        os.makedirs(paths['config'])

    paths['tb'] = paths['run'] / 'tb'
    if not os.path.exists(paths['tb']):
        os.makedirs(paths['tb'])

    paths['results'] = paths['run'] / 'results'
    if not os.path.exists(paths['results']):
        os.makedirs(paths['results'])

    paths['samples'] = paths['results'] / 'samples'
    if not os.path.exists(paths['samples']):
        os.makedirs(paths['samples'])

    paths['performance'] = paths['results'] / 'performance'
    if not os.path.exists(paths['performance']):
        os.makedirs(paths['performance'])

    return paths


class Config:
    FILENAME = 'config.pkl'
    JSON_FILENAME = 'params.json'

    @classmethod
    def load(cls, run_dir, **opts):
        path = Path(run_dir) / 'config' / cls.FILENAME
        config = load_pickle(path)
        config.update(**opts)
        return config

    def __init__(self, dataset, **opts):
        run_dir, host_name, start_time = get_run_info(dataset)
        data_path = DATA_DIR / dataset / 'PROCESSED'
        params = DEFAULTS.copy()
        params.update({
            'dataset': dataset,
            'data_path': data_path.as_posix(),
            'run_dir': run_dir,
            'host_name': host_name,
            'start_time': start_time
        })
        paths = create_folder_structure(RUNS_DIR, run_dir, data_path)

        for opt in opts:
            if opt not in params:
                continue
            params[opt] = opts[opt]

        _ = set_random_seed(params['random_seed'])

        self._PARAMS = params
        self._PATHS = paths

        self.save()

    def get(self, attr):
        if attr in self._PARAMS:
            return self._PARAMS[attr]
        raise ValueError(f'{self} does not contain attribute {attr}.')

    def set(self, attr, value):
        if attr in self._PARAMS:
            self._PARAMS[attr] = value
        else:
            raise ValueError(f'{self} does not contain attribute {attr}.')

    def params(self):
        return self._PARAMS

    def path(self, name):
        return self._PATHS[name]

    def save(self):
        # commit if you can
        try:
            commit_hash = commit(self.get('title'), self.get('start_time'))
        except Exception:
            commit_hash = "<automatic commit disabled>"
        self._PARAMS['commit_hash'] = commit_hash

        path = self.path('config') / self.JSON_FILENAME
        save_json(self.params(), path)

        path = self.path('config') / self.FILENAME
        save_pickle(self, path)

    def update(self, **params):
        for param in params:
            if param not in self._PARAMS:
                continue
            self._PARAMS[param] = params[param]

    def write_summary(self, writer):
        tag, text = get_text_summary(self.params())
        writer.add_text(tag, text, 0)

    def __repr__(self):
        return str(self._PARAMS)
