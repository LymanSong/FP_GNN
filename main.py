# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 20:56:17 2020

@author: user
"""

# import dgl
# import dgl.nn as dglnn
import torch
# import torch_geometric as tg
# from torch_geometric import data
import argparse
import os
import numpy as np
import geopandas as gpd

# from torch.utils import data as udata
from utils import *
from sklearn import model_selection
from models import *
from vectorize import *
from dataset_module import FP_dataset
from train_test import train
from test_code import test

parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for floor plans objects classification')
parser.add_argument('--dataset', type=str, default= "cubicasa", choices=["cubicasa", "UOS", "UOS_aug" ],
                    help='name of dataset (default: cubicasa)')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--random_split', type=bool, default=False,
                    help='if True, suffle files when split train/test dataset. If False, get list from local directory')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--fold_idx', type=int, default=0,
                    help='the index of the fold index in k-fold validation.')
parser.add_argument('--seed', type=int, default=np.random.permutation(1).item(),
                    help='random seed for splitting the dataset into 10 (default: 0)')
parser.add_argument('--gnn_model', type=str, default='dwgnn', choices = ["sage", "gin", "gcn", 'dwgnn'],
                    help='model to be used to analyze floor plans (sage, gin, gcn, dwgnn)')
parser.add_argument('--aggregator_type', type=str, default='sum', choices=["pool", "lstm", "mean", "max", "sum"],
                help='an aggregator function for GNN model')
parser.add_argument('--num_layers', type=int, default=6,
                    help='number of layers INCLUDING the input one. has to be greater than 1.')
parser.add_argument('--num_mlp_layers', type=int, default=3,
                    help='number of layers for MLP EXCLUDING the input one (default: 2). 1 for just simple linear model.')
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='number of hidden units (default: 64)')
parser.add_argument('--final_dropout', type=float, default=0.5,
                    help='final layer dropout (default: 0.5)')
parser.add_argument('--feature_normalize', type=str, default='standard', choices = ['standard', 'minmax', 'none'],
                    help='normalize feature matrix (default: standard).')
parser.add_argument('--train_percen', type=float, default=0.5,
                help='percentage of training data in whole dataset.')
# parser.add_argument('--val_percen', type=float, default=0.2,
#                 help='percentage of training data in whole dataset.')
parser.add_argument('--summary_dir', type=str, default='./summary',
                help='local dircetory to save the summary of the session.')
parser.add_argument('--return_output', type=bool, default=False,
                help='if true, export predicted classes with test files')
parser.add_argument('--checkpoint', type=str, default='./checkpoint',
                help='local dircetory to save the model')
parser.add_argument('--continue_train', type=bool, default=False,
                help='continue training: load the pretrained model')
parser.add_argument('--load_epoch', type=str, default='latest',
                help='load the pretrained model at epoch n, if n is \'latest\', load the latest epoch model')

#general setting
args = parser.parse_args()
device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
root_path = os.getcwd()
arguments_str = print_options(args)
dataset = FP_dataset(os.path.join(os.getcwd(), 'dataset'), args.dataset, device, args)    

os.chdir(root_path)
# write_data(dataset, args.dataset + '_' + args.feature_normalize)
# dataset = load_data(args.dataset + '_' + args.feature_normalize)


if args.fold_idx == 0:
    
    if args.random_split == True:
        train_files, test_files = model_selection.train_test_split(dataset, test_size = (1 - args.train_percen), shuffle = True)
    else:
        os.chdir(dataset.dataset_path)
        train_files = [dataset.__getitem_by_name__(i.replace("\n", "")) for i in open("train.txt", "r").readlines()]
        test_files = [dataset.__getitem_by_name__(i.replace("\n", "")) for i in open("test.txt", "r").readlines()]
        
    train_loader = torch.utils.data.DataLoader(train_files, batch_size = 10, shuffle = True, collate_fn = collate)
    test_loader = torch.utils.data.DataLoader(test_files, batch_size = 10, collate_fn = collate)
    train_filenames = [i[0] for i in train_files]
    test_filenames = [i[0] for i in test_files]
    
    legend = ['class 0(objects)', 'class 1(wall)', 'class 2(window)', 'class 3(door)', 'class 4(stair)', 'class 5(room)', 'class 6(porch)', 'class 7(outer space)']   

    model = train(args, dataset, train_loader, test_loader, device, train_files, test_files, legend, arguments_str, root_path)
    
    if args.return_output == True:
        test(args, test_files, device, dataset, root_path, False)

else: # if the k-fold index argment is not 0, use k-fold cross validation. k is the size of a batch.
    records = []
    print("Cross Validation")
    filenames = dataset.filename
    kf = model_selection.KFold(args.fold_idx, shuffle = True, random_state = args.seed)
    for train_idx, test_idx in kf.split(dataset):
        
        train_files = [dataset.__getitem_by_name__(dataset.filename[i]) for i in train_idx]
        test_files = [dataset.__getitem_by_name__(dataset.filename[i]) for i in test_idx]
        train_loader = torch.utils.data.DataLoader(train_files, batch_size = 1, shuffle = True, collate_fn = collate)
        test_loader = torch.utils.data.DataLoader(test_files, batch_size = 1, collate_fn = collate)
        train_filenames = [i[0] for i in train_files]
        test_filenames = [i[0] for i in test_files]
        # if test_files[0][0] in ['fl4', 'fl3', 'fl2', 'fl1_aug1']:#, 'fl4_aug1', 'fl1', 'fl3_aug1']:
        #     continue
        print(test_filenames)
        legend = ['class 0(objects)', 'class 1(wall)', 'class 2(window)', 'class 3(door)', 'class 4(stair)', 'class 5(lift)', 'class 6(room)', 'class 7(hallway)', 'class 8(X-room)']   
        model, record, report, filename = train(args, dataset, train_loader, test_loader, device, train_files, test_files, legend, arguments_str, root_path)
        records.append((filename, record, report))
        args.continue_train = False
        if args.return_output == True:
            test(args, test_files, device, dataset, root_path, False)
            
    # output summary(cross validation)
    os.chdir(root_path)
    os.chdir(args.summary_dir)
    if args.dataset not in os.listdir():
        os.mkdir(args.dataset)
    os.chdir(args.dataset)
    
    s_filepath = os.path.join(time.ctime().replace(':', ';') + '_' + args.dataset + '_' + args.gnn_model + '_' + args.aggregator_type + '_' + '.txt')
    if os.path.exists(s_filepath) == False:
        os.mkdir(s_filepath)
    os.chdir(s_filepath)
    with open('arguments_str.txt', 'w') as f:
        f.write(arguments_str)
    for rec in records:
        with open(rec[0] + '.txt', "w") as f:
            f.write(rec[1])
            f.write(rec[2])