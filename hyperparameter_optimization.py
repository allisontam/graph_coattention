"""Optimizes hyperparameters using Bayesian optimization."""

from argparse import ArgumentParser, Namespace
from copy import deepcopy
import json
from typing import Dict, Union
import os
import logging

from hyperopt import fmin, hp, tpe
import numpy as np

from e2e import all_folds, add_args


SPACE = {
    'n_attention_head': hp.quniform('n_attention_head', low=1, high=16, q=1),
    'n_prop_step': hp.quniform('depth', low=2, high=6, q=1),
    'dropout': hp.quniform('dropout', low=0.0, high=0.4, q=0.05),
    # 'margin': hp.quniform('margin', low=1, high=10, q=1),
    'd_hid': hp.quniform('d_hid', low=6, high=64, q=4),
}
INT_KEYS = ['n_attention_head', 'n_prop_step', 'd_hid']

## UTILS
def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def create_logger(name: str, save_dir: str = None, quiet: bool = False) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Set logger depending on desired verbosity
    ch = logging.StreamHandler()
    if quiet:
        ch.setLevel(logging.INFO)
    else:
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    if save_dir is not None:
        makedirs(save_dir)

        fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'))
        fh_v.setLevel(logging.DEBUG)
        fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'))
        fh_q.setLevel(logging.INFO)

        logger.addHandler(fh_v)
        logger.addHandler(fh_q)

    return logger


def grid_search(args: Namespace):
    # Create loggers
    logger = create_logger(name='hyperparameter_optimization', save_dir=args.log_dir, quiet=True)

    # Run grid search
    results = []

    # Define hyperparameter optimization
    def objective(hyperparams: Dict[str, Union[int, float]]) -> float:
        # Convert hyperparams from float to int when necessary
        for key in INT_KEYS:
            hyperparams[key] = int(hyperparams[key])

        # Update args with hyperparams
        hyper_args = deepcopy(args)
        hyper_args.memo = '_'.join(f'{key}-{value}' for key, value in hyperparams.items())
        for key, value in hyperparams.items():
            setattr(hyper_args, key, value)
        # Record hyperparameters
        logger.info(hyperparams)

        # Cross validate
        mean_score, std_score = all_folds(hyper_args)

        # Record results
        logger.info(f'{mean_score} +/- {std_score} {hyper_args.metric}')

        results.append({
            'mean_score': mean_score,
            'std_score': std_score,
            'hyperparams': hyperparams,
        })

        # Deal with nan
        if np.isnan(mean_score):
            if hyper_args.dataset_type == 'classification':
                mean_score = 0
            else:
                raise ValueError('Can\'t handle nan score for non-classification dataset.')

        return -1 * mean_score

    fmin(objective, SPACE, algo=tpe.suggest, max_evals=args.num_iters)

    # Report best result
    results = [result for result in results if not np.isnan(result['mean_score'])]
    best_result = min(results, key=lambda result: -1 * result['mean_score'])
    logger.info('best')
    logger.info(best_result['hyperparams'])
    logger.info(f'{best_result["mean_score"]} +/- {best_result["std_score"]} {args.metric}')

    # Save best hyperparameter settings as JSON config file
    makedirs(args.config_save_path, isfile=True)

    with open(args.config_save_path, 'w') as f:
        json.dump(best_result['hyperparams'], f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    add_args(parser)
    parser.add_argument('--num_iters', type=int, default=20,
                        help='Number of hyperparameter choices to try')
    parser.add_argument('--config_save_path', type=str,
                        help='Path to .json file where best hyperparameter settings will be written')
    parser.add_argument('--log_dir', type=str,
                        help='(Optional) Path to a directory where all results of the hyperparameter optimization will be written')
    args = parser.parse_args()
    assert args.config_save_path  # check if supplied

    grid_search(args)
