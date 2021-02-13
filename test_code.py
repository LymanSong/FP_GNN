# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 09:46:07 2020

@author: user
"""

import torch
import os
import shutil
import natsort
import dgl
import time
import numpy as np
from tqdm import tqdm
import argparse
# import dgl.nn as dglnn

from utils import *
from models import *
from train_test import model_load
from dataset_module import FP_dataset

parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for floor plans objects classification')
parser.add_argument('--dataset', type=str, default= "cubicasa_test", choices=["cubicasa_test", "UOS_test", "UOS_aug_test" ],
                    help='name of dataset (default: cubicasa_test)')
parser.add_argument('--device', type=int, default=0,
                    help='which gpu to use if any (default: 0)')
parser.add_argument('--gnn_model', type=str, default='dwgnn', choices = ["sage", "gin", "gcn", "mpnn", 'dwgnn'],
                    help='model to be used to analyze floor plans (sage, gin, gcn, mpnn, dwgnn)')
parser.add_argument('--aggregator_type', type=str, default='mean', choices=["pool", "lstm", "mean", "max", "sum"],
                    help='an aggregator function for GNN model')
parser.add_argument('--feature_normalize', type=str, default='standard', choices = ['standard', 'minmax', 'none'],
                    help='normalize feature matrix (default: standard).')
parser.add_argument('--num_layers', type=int, default=6,
                    help='number of layers INCLUDING the input one. has to be greater than 1.')
parser.add_argument('--num_mlp_layers', type=int, default=3,
                    help='number of layers for MLP EXCLUDING the input one (default: 2). 1 for just simple linear model.')
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='number of hidden units (default: 64)')
parser.add_argument('--checkpoint', type=str, default='./checkpoint',
                    help='local dircetory to save the model')
parser.add_argument('--load_epoch', type=str, default='latest',
                    help='load the pretrained model at epoch n, if n is \'latest\', load the latest epoch model')
args = parser.parse_args()

def test(args, test_files, device, dataset, root_path, test = True):
    os.chdir(root_path)
    n_features = dataset.FP_set[0][2].shape[1]
    if test == True:
        n_labels = dataset.__get_class_num__()
    else:
        n_labels = torch.max(torch.cat([i[3] for i in dataset.FP_set])).item() + 1
    #model setting
    if args.gnn_model == 'sage':
        model = SAGE(in_feats=n_features, hid_feats=args.hidden_dim, out_feats=n_labels, num_layers = args.num_layers, aggregator_type = args.aggregator_type).to(device)
    elif args.gnn_model == 'gcn':
        model = GCN(in_feats=n_features, hid_feats=args.hidden_dim, out_feats=n_labels, num_layers = args.num_layers).to(device)
    elif args.gnn_model == 'dwgnn':
        model = DWGNN(in_feats=n_features, hid_feats=args.hidden_dim, out_feats=n_labels, edge_feats = 1, num_layers = args.num_layers, aggregator_type = args.aggregator_type).to(device)
    elif args.gnn_model == 'gin':
        model = GIN(in_feats=n_features, hid_feats=args.hidden_dim, out_feats=n_labels, num_layers = args.num_layers, num_mlp_layers = args.num_mlp_layers, aggregator_type = args.aggregator_type).to(device)
    
    
    if args.gnn_model != 'gcn':
        PATH = os.path.join(args.checkpoint, args.dataset.split(sep = '_')[0], args.gnn_model + '_' + args.aggregator_type + '_' + args.load_epoch + '_net.pth')
    else:
        PATH = os.path.join(args.checkpoint, args.dataset.split(sep = '_')[0], args.gnn_model + '_' + args.load_epoch + '_net.pth')
    model, opt, cur_epoch, loss, record, session_record, e_time = model_load(model, PATH, n_features, n_labels, args, device)
    model.eval()
    filenames = [fp[0] for fp in test_files]
    
    for filename in filenames:
        # get testfile from the dataset and predict with trained model
        FP = dataset.__getitem_by_name__(filename)
        filename = FP[0]
        x = FP[4]
        
        node_features = torch.stack([FP[4].ndata[j] for j in [i for i in FP[4].ndata]]).T # shape : (n_node, n_features)
        node_features = node_features.type(torch.FloatTensor).to(device)
        if args.gnn_model in ['mpnn', 'dwgnn']:
            edge_features = FP[4].edata['edge_dists']
        else:
            edge_features = None
        if len(filenames) == 1:
            logit = model(x, node_features, edge_features, batch_norm = False)
        else:
            logit = model(x, node_features, edge_features)
        class_pred = logit.max(1)[1].type_as(torch.zeros(size = logit.max(1)[1].shape)).numpy()
        os.chdir(dataset.fps_path)
        if test == True:
            df = gpd.read_file(os.path.join(filename, filename + '_polys.shp'))
        else:
            df = gpd.read_file(os.path.join(filename, filename + '_labeled.shp'))
        df['obj_class'] = class_pred
        
        # make prediction directory
        output_path = os.path.join(root_path, 'output')
        os.chdir(output_path)
        if args.dataset not in os.listdir(output_path):
            os.mkdir(args.dataset)
        cur_output_path = os.path.join(output_path, args.dataset)
        os.chdir(cur_output_path)
        cur_model_path = os.path.join(cur_output_path, args.gnn_model)
        if args.gnn_model not in os.listdir(cur_output_path):
            os.mkdir(cur_model_path)
        os.chdir(cur_model_path)
        if filename + '_' + args.aggregator_type in os.listdir(cur_model_path):
            shutil.rmtree(filename + '_' + args.aggregator_type)
        
        # save prediction
        os.mkdir(os.path.join(cur_model_path, filename + '_' + args.aggregator_type))
        df.to_file(os.path.join(cur_model_path, filename + '_' + args.aggregator_type, str(filename) + '_pred.shp'))
    
    os.chdir(root_path)

if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    root_path = os.getcwd()
    arguments_str = print_options(args)
    
    dataset = FP_dataset(os.path.join(os.getcwd(), 'dataset'), args.dataset, device, args, False)
    test_files = [dataset[i] for i in range(len(dataset))]
    test(args, test_files, device, dataset, root_path)