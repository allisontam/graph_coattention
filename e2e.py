"""
File used to train the networks.
"""
import os
import csv
import pprint
import random
import logging
import argparse
import pickle as pkl
from tqdm import tqdm
from sklearn import metrics

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data

import gen_dataset as gen
from utils.file_utils import setup_running_directories, save_experiment_settings
from utils.functional_utils import combine

from model import DrugDrugInteractionNetwork
from test import load_trained_model, run_evaluation, prepare_test_dataloader
from utils.ddi_utils import ddi_train_epoch, ddi_valid_epoch


def post_parse_args(opt):
    # Set the random seed manually for reproducibility.
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    data = os.path.basename(os.path.normpath(opt.input_data_path))
    opt.model_dir = './{}_trained'.format(data)
    opt.result_dir = './{}_results'.format(data)
    opt.setting_dir = './{}_settings'.format(data)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)

    if not hasattr(opt, 'exp_prefix'):
        opt.exp_prefix = opt.memo + '-cv_{}_{}'.format(opt.fold_i, opt.n_fold)
    if opt.debug:
        opt.exp_prefix = 'dbg-{}'.format(opt.exp_prefix)
    if not hasattr(opt, 'global_step'):
        opt.global_step = 0

    opt.setting_pkl = os.path.join(opt.setting_dir, opt.exp_prefix + '.npy')
    opt.best_model_pkl = os.path.join(opt.model_dir, opt.exp_prefix + '.pth')
    opt.result_csv_file = os.path.join(opt.result_dir, opt.exp_prefix + '.csv')

    print(opt.best_model_pkl)
    print(opt.result_csv_file)
    print(opt.model_dir)
    return opt


def build_model(opt, device):
    return DrugDrugInteractionNetwork(
        n_side_effect=1,
        n_atom_type=100,
        n_bond_type=20,
        d_node=opt.d_hid,
        d_edge=opt.d_hid,
        d_atom_feat=3,
        d_hid=opt.d_hid,
        d_readout=opt.d_readout,
        n_head=opt.n_attention_head,
        n_prop_step=opt.n_prop_step,
        dropout=opt.dropout,
        score_fn='trans').to(device)


def prepare_dataloaders(opt):
    data = pd.read_csv(os.path.join(opt.input_data_path, 'data.csv'))
    splits = pkl.load(open(opt.split_path, 'rb'))
    train_split, val_split = data.iloc[splits[0]], data.iloc[splits[1]]

    train_loader = torch.utils.data.DataLoader(
        gen.GenericDataset(train_split, opt.graph_input),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=gen.train_collate_fn)

    valid_loader = torch.utils.data.DataLoader(
        gen.GenericDataset(val_split, opt.graph_input),
        num_workers=2,
        batch_size=opt.batch_size,
        collate_fn=gen.test_collate_fn)
    return train_loader, valid_loader


def valid_epoch(model, data_valid, device, opt, threshold=None):
    return ddi_valid_epoch(model, data_valid, device, opt, threshold)


def train(model, datasets, device, opt):

    data_train, data_valid = datasets
    optimizer = optim.Adam(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_lambda)

    with open(opt.result_csv_file, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['train_loss', 'auroc_valid'])

    ddi_best_valid_perf = 0
    waited_epoch = 0
    averaged_model = model.state_dict()
    for epoch_i in range(opt.n_epochs):
        data_train.dataset.shuffle()
        data_valid.dataset.shuffle()
        print('\nEpoch ', epoch_i)

        # ============= Training Phase =============
        train_loss, elapse, averaged_model = \
            ddi_train_epoch(model, data_train, optimizer, averaged_model, device, opt)
        logging.info(' Loss:    %5f, used time: %f min', train_loss, elapse)

        # ============= Validation Phase =============

        # Load the averaged model weight for validation
        updated_model = model.state_dict() # validation start
        model.load_state_dict(averaged_model)

        valid_perf, elapse = valid_epoch(model, data_valid, device, opt)
        valid_auroc = valid_perf['auroc']
        logging.info(' Validation: %5f, used time: %f min', valid_auroc, elapse)

        # Load back the trained weight
        model.load_state_dict(updated_model) # validation end

        # early stopping
        if valid_auroc > ddi_best_valid_perf:
            logging.info(' --> Better validation result!')
            waited_epoch = 0
            torch.save(
                {'global_step': opt.global_step,
                 'model':averaged_model,
                 'threshold': valid_perf['threshold']},
                opt.best_model_pkl)
        else:
            waited_epoch += 1
            logging.info(' --> Observing ... (%d/%d)', waited_epoch, opt.n_epochs)

        # ============= Bookkeeping Phase =============
        # Keep the validation record
        ddi_best_valid_perf = max(valid_auroc, ddi_best_valid_perf)

        # Keep all metrics in file
        with open(opt.result_csv_file, 'a') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([train_loss, valid_auroc])


def predict(test_data, device, test_opt, out_log):
    model, threshold = load_trained_model(test_opt, device)
    print("loaded model")
    test_perf, _ = run_evaluation(model, test_data, device, test_opt, threshold=threshold)

    with open(out_log, 'a') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["TEST PERFORMANCE", "METRICS"])
        for k,v in test_perf.items():
            csv_writer.writerow([k, v])


# HYPERPARAMS
# d_hid, d_readout, n_attention_head, n_prop_step, dropout, margin

def main():
    parser = argparse.ArgumentParser()

    # Directory containing precomputed training data split.
    parser.add_argument('input_data_path', default=None,
                        help="Input data path, e.g. ./data/decagon/")

    parser.add_argument('-f', '--fold', default='0/10', type=str,
              help="Which fold to test on, format x/total")
    parser.add_argument('-mm', '--memo', help='Memo for experiment', default='default')

    # How to train
    parser.add_argument('-e', '--n_epochs', type=int, default=30)
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Bc of uneven class sizes')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-l2', '--l2_lambda', type=float, default=0)

    # Hyperparams
    parser.add_argument('-d_h', '--d_hid', type=int, default=32)
    parser.add_argument('-d_readout', '--d_readout', type=int, default=32)
    parser.add_argument('-n_p', '--n_prop_step', type=int, default=3)
    parser.add_argument('-n_h', '--n_attention_head', type=int, default=8)
    parser.add_argument('-drop', '--dropout', type=float, default=0.1)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-dbg', '--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=1337)

    # If mpnn is true, then first pairing happens with itself
    # for repetitions=1, this is mpnn-like.
    parser.add_argument('--mpnn', action='store_true')

    opt = parser.parse_args()

    opt.fold_i, opt.n_fold = map(int, opt.fold.split('/'))
    assert 0 <= opt.fold_i < opt.n_fold

    opt = post_parse_args(opt)
    setup_running_directories(opt)

    # save the dataset split in setting dictionary
    pprint.pprint(vars(opt))
    # save the setting
    logging.info('Related data will be saved with prefix: %s', opt.exp_prefix)

    opt.split_path = os.path.join(opt.input_data_path, "folds/fold_{}.pkl".format(opt.fold_i))
    opt.graph_input = os.path.join(opt.input_data_path, 'drug.feat.wo_h.self_loop.idx.jsonl')
    assert os.path.exists(opt.split_path)
    assert os.path.exists(opt.graph_input)

    save_experiment_settings(opt)

    # build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("using cuda",opt.gpu)
        torch.cuda.set_device(opt.gpu)
    else:
        print("on cpu")

    dataloaders = prepare_dataloaders(opt)
    train(build_model(opt), dataloaders, device, opt)

    test_opt = np.load(opt.setting_pkl, allow_pickle=True).item()
    test_opt.batch_size = 6
    test_opt.split_path = os.path.join(opt.input_data_path, 'folds/fold_{}.pkl'.format(eval_opt.fold_i))

    test_data = prepare_test_dataloader(test_opt)
    predict(test_data, device, test_opt, opt.result_csv_file)


if __name__ == "__main__":
    main()
