import os
import argparse
import pickle
import numpy as np
import random
import torch
import torch.optim

"""
    Utility functions for handling parsed arguments

"""
def get_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser('Train a model')
    parser.add_argument(
        '--caps_dir',
        help='Data using CAPS structure.',
        default='/path/to/CAPS'
    )

    parser.add_argument(
        '--tsv_path',
        help='TSV path with subjects/sessions to use for data generation.',
        default='/path/to/TSV'
    )

    parser.add_argument(
        '--output_dir',
        help='Folder containing the synthetic dataset.',
        default='/tmp/output'
    )

    parser.add_argument(
        '-np', '--nproc',
        help='Number of cores used for processing (2 by default)',
        type=int, default=2
    )

    parser.add_argument(
        '--preprocessing',
        help='Defines the type of preprocessing of CAPS data.',
        type=str, default='t1-linear')

    parser.add_argument(
        "--diagnoses",
        help="Labels that must be extracted from merged_tsv.",
        nargs="+", type=str, choices=['AD', 'CN', 'MCI', 'sMCI', 'pMCI'], default=['AD', 'CN'])

    parser.add_argument(
        '--hippocampus_roi',
        help='Defines the type of preprocessing of CAPS data.',
        default=True)

    parser.add_argument(
        '--n_splits',
        help='If a value is given will load data of a k-fold CV. Else will load a single split.',
        type=int, default=1)

    parser.add_argument('--batch_size',
                        help='Batch size used in DataLoader (default=1).',
                        default=2, type=int)

    parser.add_argument(
        "--baseline",
        help="Performs the analysis based on <label>_baseline.tsv files",
        default=True, action="store_true")

    parser.add_argument(
        '--accumulation_steps', '-asteps',
        help='Accumulates gradients during the given number of iterations before performing the weight update '
             'in order to virtually increase the size of the batch. (default=1)',
        default=1, type=int)

    parser.add_argument(
        '--evaluation_steps', '-esteps',
        default=20, type=int,
        help='Fix the number of iterations to perform before computing an evaluations. Default will only '
             'perform one evaluation at the end of each epoch. (default=0)')

    parser.add_argument(
        '--sigma',
        type=float,
        default=0,
        help="Standard deviation of the noise added for the random dataset."
    )

    parser.add_argument(
        '--unnormalize',
        help='Disable default MinMaxNormalization.',
        action="store_true",
        default=False)

    parser.add_argument(
        '--patience',
        help='Number of epochs for early stopping patience. (default=10)',
        type=int, default=10)

    parser.add_argument(
        '--mode',
        help='Choose which dataset is generated (random, trivial).',
        default='roi'  #patch
    )

    parser.add_argument(
        '--mode_task',
        help='''****** Choose a type of network ******''',
        default='single')   # single for roi  cnn for patch

    parser.add_argument(
        '--model',
        help='CNN Model to be used during the training.',
        default='MAXNet')

    parser.add_argument(
        '--dropout',
        help='rate of dropout that will be applied to dropout layers in CNN. (default=None)',
        default=None)

    parser.add_argument(
        '--split',
        help='Train the list of given folds. By default train all folds.',
        type=int, default=1)

    parser.add_argument(
        '--tolerance',
        help='Value for the early stopping tolerance. (default=0.0)',
        type=float, default=0.0)

    parser.add_argument('--use_cpu',
                        help='If provided, will use CPU instead of GPU.',
                        default=False)
    parser.add_argument(
        '--n_subjects',
        type=int,
        default=1,
        help="Number of subjects in each class of the synthetic dataset."
    )
    parser.add_argument(
        '--mean',
        type=float,
        default=0,
        help="Mean value of the noise added for the random dataset."
    )
    parser.add_argument('--epochs',
                        type=int,
                        default=100,
                        help='The number of epochs should be trained')
    parser.add_argument('--optimizer',
                        type=str,
                        default='AdamW',
                        help='The optimizer that should be used when training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001, 
                        help='The learning rate for training the model')
    parser.add_argument('--lr_block',
                        type=float,
                        default=0.001, 
                        help='The learning rate for training the conv layer')
    parser.add_argument('--lr_net',
                        type=float,
                        default=1e-5, 
                        help='The learning rate for the latent neural network')
    parser.add_argument('--momentum',
                        type=float,
                        default=0.9,
                        help='The optimizer momentum parameter')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help='Weight decay used in the optimizer')
    parser.add_argument('--disable_cuda',
                        action='store_true',
                        help='Flag that disables GPU usage if set')
    parser.add_argument('--log_dir',
                        type=str,
                        default='tensorboard_logs',
                        help='The directory in which train progress should be logged')
    parser.add_argument('--W1',
                        type=int,
                        default = 1)
    parser.add_argument('--H1',
                        type=int,
                        default = 1)
    parser.add_argument('--L1',
                        type=int,
                        default = 1)
    parser.add_argument('--num_channels',
                        type=int,
                        default = 256)
    parser.add_argument('--num_features',
                        type=int,
                        default = 511)
    parser.add_argument('--gamma',
                        type=float,
                        default=0.5)
    parser.add_argument('--resume',
                        type=str,
                        help='resume from checkpoint',
                        default='')
    parser.add_argument('--eval_sample',
                        type=str,
                        help='evaluate one sample',
                        default='')
    parser.add_argument('--sample_session',
                        type=str,
                        default='')
    parser.add_argument('--intermediate_layers', type=str, default='conv3,conv4,conv5',
                        help='Intermediate layers numbers in features of the model, use comma to split')

    args = parser.parse_args()
    return args


def save_args(args: argparse.Namespace, directory_path: str) -> None:
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    # Save the args in a text file
    with open(directory_path + '/args.txt', 'w') as f:
        for arg in vars(args):
            val = getattr(args, arg)
            if isinstance(val, str):  # Add quotation marks to indicate that the argument is of string type
                val = f"'{val}'"
            f.write('{}: {}\n'.format(arg, val))
    # Pickle the args for possible reuse
    with open(directory_path + '/args.pickle', 'wb') as f:
        pickle.dump(args, f)                                                                               
    

def load_args(directory_path: str) -> argparse.Namespace:
    with open(directory_path + '/args.pickle', 'rb') as f:
        args = pickle.load(f)
    return args

def get_optimizer(model, args: argparse.Namespace) -> torch.optim.Optimizer:
    optim_type = args.optimizer
    #create parameter groups
    params_to_freeze = []
    params_to_train = []

    for name,param in model.named_parameters():
        if 'feature_' not in name and '_add_on' not in name:
            params_to_train.append(param)

    # set up optimizer
    if optim_type == 'SGD':
        paramlist = [
            {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay,
             "momentum": args.momentum},
            {"params": model._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay,
             "momentum": args.momentum},
            {"params": model.feature_layer.parameters(), "lr": args.lr, "weight_decay_rate": 0, "momentum": 0}]
    else:
        paramlist = [
            {"params": params_to_train, "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": model._add_on.parameters(), "lr": args.lr_block, "weight_decay_rate": args.weight_decay},
            {"params": model.feature_layer.parameters(), "lr": args.lr, "weight_decay_rate": 0}]



    if optim_type == 'SGD':
        return torch.optim.SGD(paramlist,
                               lr=args.lr,
                               momentum=args.momentum), params_to_freeze, params_to_train
    if optim_type == 'Adam':
        return torch.optim.Adam(paramlist,lr=args.lr,eps=1e-07), params_to_freeze, params_to_train
    if optim_type == 'AdamW':
        return torch.optim.AdamW(paramlist,lr=args.lr,eps=1e-07, weight_decay=args.weight_decay), params_to_freeze, params_to_train

    raise Exception('Unknown optimizer argument given!')


