import pickle as pkl
import logging
import argparse
import random
import pprint
import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data

import gen_dataset as gen
from model import DrugDrugInteractionNetwork
from model import DrugDrugInteractionNetwork
from train import valid_epoch as run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


def prepare_test_dataloader(train_opt):
    data = pd.read_csv(os.path.join(train_opt.input_data_path, 'data.csv'))
    splits = pkl.load(open(train_opt.split_path, 'rb'))
    test_split = data.iloc[splits[2]]
    print("batch_size", train_opt.batch_size)

    return torch.utils.data.DataLoader(
        gen.GenericDataset(test_split, train_opt.graph_input),
        num_workers=2,
        batch_size=train_opt.batch_size,
        collate_fn=gen.test_collate_fn)


def load_trained_model(train_opt, device):
    model = DrugDrugInteractionNetwork(
        n_side_effect=1,
        n_atom_type=100,
        n_bond_type=20,
        d_node=train_opt.d_hid,
        d_edge=train_opt.d_hid,
        d_atom_feat=3,
        d_hid=train_opt.d_hid,
        d_readout=train_opt.d_readout,
        n_head=train_opt.n_attention_head,
        n_prop_step=train_opt.n_prop_step).to(device)

    trained_state = torch.load(train_opt.best_model_pkl)
    model.load_state_dict(trained_state['model'])
    threshold = trained_state['threshold']
    return model, threshold


def main():
    parser = argparse.ArgumentParser()

    # Dirs
    # parser.add_argument('--settings', help='Setting, ends in .npy', default=None)
    # parser.add_argument('-mm', '--memo', help='Trained model, ends in .pth', default='default')
    parser.add_argument('--model_dir', default='./exp_trained')
    parser.add_argument('--settings_dir', default='./exp_settings')
    parser.add_argument('--fold_i', type=int, required=True)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)

    eval_opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("using cuda",eval_opt.gpu)
        torch.cuda.set_device(eval_opt.gpu)

    eval_opt.setting_pkl = os.path.join(eval_opt.settings_dir, 'default-cv_{}_10.npy'.format(eval_opt.fold_i))
    eval_opt.best_model_pkl = os.path.join(eval_opt.model_dir, 'default-cv_{}_10.pth'.format(eval_opt.fold_i))

    test_opt = np.load(eval_opt.setting_pkl, allow_pickle=True).item()
    test_opt.best_model_pkl = eval_opt.best_model_pkl
    test_opt.batch_size = eval_opt.batch_size
    test_opt.split_path = os.path.join(test_opt.input_data_path, 'folds/fold_{}.pkl'.format(eval_opt.fold_i))

    # EVAL OPT NOT REALLY USED ANYMORE AFTER THIS

    # create data loader
    test_data = prepare_test_dataloader(test_opt)

    # build model
    model, threshold = load_trained_model(test_opt, device)
    print("Threshold", threshold, 'for fold', eval_opt.fold_i)

    # start testing
    test_perf, _ = run_evaluation(model, test_data, device, test_opt, threshold=threshold)
    for k,v in test_perf.items():
        if k != 'threshold':
            print(k, v, 'for fold', eval_opt.fold_i)


if __name__ == "__main__":
    main()
