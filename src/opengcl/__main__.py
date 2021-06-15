import time

t_start = time.time()

import datetime
import os, sys
import copy
import ast

import numpy as np
import random
from argparse import ArgumentParser
import torch

from .utils import set_device, debug, ModuleParameter, HyperParameter, ModelInput
from . import datasets, framework

cur_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.normpath(os.path.join(cur_dir, '..', '..'))
log_dir = os.path.join(root_dir, 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


def parse_args(*args, **kwargs):
    parser = ArgumentParser()
    devicegroup = parser.add_mutually_exclusive_group()
    devicegroup.add_argument('--cpu', action='store_true')
    devicegroup.add_argument('--device', type=int, default=0)

    parser.add_argument('--task', required=True)
    parser.add_argument('--dataset', required=True)

    for k, v in ModuleParameter().params().items():
        parser.add_argument('--' + k.replace('_', '-'), default=v)

    for k, v in HyperParameter().params().items():
        parser.add_argument('--' + k.replace('_', '-'), default=v, type=type(v))

    return parser.parse_args(*args, **kwargs)


def main(args):
    # parsing
    while True:
        log_name = os.path.join(log_dir, "_".join((
            datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            str(int(time.time() * 1000))[-1:], args.dataset, args.task)) + ".txt" )
        if not os.path.exists(log_name):
            break

    args = {x: y for x, y in args.__dict__.items() if y is not None}

    logs = []

    def log(*args, display=False, **kwargs):
        args = copy.deepcopy(args)
        if display:
            print(*args, **kwargs)
        logs.append((args, kwargs))

    log('python3 -m opengcl', ' '.join(sys.argv[1:]), flush=True)

    # set device
    if not torch.cuda.is_available() or args.get('cpu', False):
        set_device(torch.device('cpu'))
    else:
        set_device(torch.device('cuda', args['device']))

    debug(f'[main] ({time.time() - t_start}s) Loading dataset...')

    dataset = datasets.datasetdict[args['dataset']]()
    debug(f'[main] ({time.time() - t_start}s) Splitting dataset...')
    dataset.get_split_data(args['clf_ratio'])
    debug(f'[main] ({time.time() - t_start}s) Dataset loaded and split.')
    debug(f'[main] ({time.time() - t_start}s) Start building GCL framework...')
    model = framework.Trainer(dataset, ModuleParameter(**args), HyperParameter(**args))
    debug(f'[main] ({time.time() - t_start}s) GCL framework built and trained.')
    debug(f'[main] ({time.time() - t_start}s) Start classification...')
    if args['task'] == 'node':
        res = model.classify(ModelInput.NODES)
    else:
        res = model.classify(ModelInput.GRAPHS)
    debug(f'[main] ({time.time() - t_start}s) Classification finished.')
    debug(f'[main] F1-score: micro = {res}.')

    log(res, flush=True)

    # write to log
    with open(log_name, "w") as f:
        for args, kwargs in logs:
            print(*args, file=f, **kwargs)


if __name__ == "__main__":
    main(parse_args())
