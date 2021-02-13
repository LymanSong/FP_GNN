# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:15:15 2020

@author: user
"""
import torch
import os
import shutil
import natsort
import dgl
# import dgl.nn as dglnn
from utils import *
from models import *
import time
import numpy as np
from tqdm import tqdm
torch.backends.cudnn.enabled = False
def model_save(model, epoch, loss, opt, args, record, session_record, e_time, latest = False):
    if os.path.exists(os.path.join(args.checkpoint, args.dataset)) == False:
        os.mkdir(os.path.join(args.checkpoint, args.dataset))
    if latest == True:
        PATH = os.path.join(args.checkpoint, args.dataset, args.gnn_model + '_' + args.aggregator_type + '_latest_net.pth')
        if os.path.exists(PATH):
            os.remove(PATH)
    else:
        PATH = os.path.join(args.checkpoint, args.dataset, args.gnn_model + '_' + args.aggregator_type + '_' + str(epoch) + '_net.pth')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
            'record' : record,
            'session_record' : session_record,
            'e_time' : e_time
            }, PATH)
    
def model_load(model, PATH, n_features, n_labels, args, device):
    opt = torch.optim.Adam(model.parameters())
    checkpoint = torch.load(PATH)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        record = checkpoint['record']
        session_record = checkpoint['session_record']
        e_time = checkpoint['e_time']
        return model, opt, epoch, loss, record, session_record, e_time
    except:
        raise RuntimeError('you must match the hyperparameters with loaded model.')
    
def train(args, dataset, train_loader, test_loader, device, train_files, test_files, legend, arguments_str, root_path):
    os.chdir(root_path)
    n_features = dataset.FP_set[1][2].shape[1]
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
    if args.continue_train == True:
        PATH = os.path.join(args.checkpoint, args.dataset, args.gnn_model + '_' + args.aggregator_type + '_' + args.load_epoch + '_net.pth')
        model, opt, cur_epoch, loss, record, session_record, e_time = model_load(model, PATH, n_features, n_labels, args, device)
    
    else:
        opt = torch.optim.Adam(model.parameters())
        cur_epoch = 0
        record = []
        session_record = ''
        e_time = 0
    
    
    start = time.time()
    print(session_record)
    pbar = tqdm(range(cur_epoch + 1, args.epochs + 1), desc = 'training...')
    for epoch in pbar:
        model.train()
        acc_accum = loss_accum = 0
        for batch in train_loader:
            # filenames = batch[2]
            x = batch[0]
            node_labels = batch[1]
            node_features = torch.stack([batch[0].ndata[j] for j in [i for i in batch[0].ndata]]).T #stack all the columns of node attributes [shape : (n_node, n_features)]
            node_features = node_features.type(torch.FloatTensor).to(device)
            if args.gnn_model == 'dwgnn':
                edge_features = batch[0].edata['edge_dists']
            else:
                edge_features = None
            if train_loader.batch_size == 1:
                logit = model(x, node_features, edge_features, batch_norm = False)
            else:
                logit = model(x, node_features, edge_features)
            logit.cuda()
            
            loss_train = F.cross_entropy(logit.cpu(), node_labels)
            acc_train = accuracy(logit, node_labels)
            if opt is not None:
                opt.zero_grad()
                loss_train.backward()
                opt.step()
            loss_accum += loss_train.detach().numpy()
            acc_accum += acc_train.detach().numpy()
        iteration = len(train_files)/train_loader.batch_size
        average_acc = acc_accum/iteration
        average_loss = loss_accum/iteration
        
        if epoch % 50 == 0 or epoch == args.epochs:
            session_record += 'Epoch: {:04d} '.format(epoch + 1)
            session_record += 'loss_train: {:.4f} '.format(average_loss)
            session_record += 'acc_train: {:.4f}\n'.format(average_acc)
            print('\nEpoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(average_loss),
                  'acc_train: {:.4f}'.format(average_acc))
            
            model.eval()
            acc_accum = loss_accum = 0
            labels = []
            preds = []
            for batch in test_loader:
                x = batch[0]
                node_features = torch.stack([batch[0].ndata[j] for j in [i for i in batch[0].ndata]]).T 
                node_features = node_features.type(torch.FloatTensor).to(device)
                if args.gnn_model == 'dwgnn':
                    edge_features = batch[0].edata['edge_dists']
                else:
                    edge_features = None
                logit = model(x, node_features, edge_features)
                node_labels = batch[1]
    
                loss_test = F.cross_entropy(logit.cpu(), node_labels)
                acc_test = accuracy(logit, node_labels)
                loss_accum += loss_test.detach().numpy()
                acc_accum += acc_test.detach().numpy()
                
                labels.append(node_labels.numpy())
                pred = logit.max(1)[1].type_as(node_labels)
                preds.append(pred.numpy())
            iteration = len(test_files)/test_loader.batch_size
            average_acc = acc_accum/iteration
            average_loss = loss_accum/iteration  
            session_record += '\t    '
            session_record += 'loss_test: {:.4f} '.format(average_loss)
            session_record += 'acc_test: {:.4f}\n'.format(average_acc)
            print(
            'loss_test: {:.4f}'.format(average_loss),
            'acc_test: {:.4f}'.format(average_acc))
            record.append((average_loss, average_acc))
            print((time.time() - start) + e_time)
           
            model_save(model, epoch, loss_train, opt, args, record, session_record, time.time() - start)
            model_save(model, epoch, loss_train, opt, args, record, session_record, time.time() - start, latest = True)
            
            if epoch == args.epochs:
                report = classification_repo(labels, preds, legend)
                iteration = len(test_files)/test_loader.batch_size
                average_acc = acc_accum/iteration
                average_loss = loss_accum/iteration
            
    stop = time.time()
    print(stop - start)     
    best = np.max([i[1] for i in record])
    
    os.chdir(args.summary_dir)
    
    # output summary
    if train_loader.batch_size >  1:
        if args.dataset not in os.listdir():
            os.mkdir(args.dataset)
        os.chdir(args.dataset)
        s_filename = os.path.join(time.ctime().replace(':', ';') + '_' + args.dataset + '_' + args.gnn_model + '_' + args.aggregator_type + '_' + '.txt')
        with open(s_filename, 'w') as f:
            f.write(arguments_str)
            f.write(session_record)
            f.write(report)
            f.write('\n best acc : ' + str(best))
        return model
    else:
        if args.dataset not in os.listdir():
            os.mkdir(args.dataset)
        os.chdir(args.dataset)
        filenames = [test_files[i][0] for i in range(len(test_files))]
        s_filename = os.path.join(time.ctime().replace(':', ';') + '_' + args.dataset + '_' + args.gnn_model + '_' + args.aggregator_type + '_' + str(filenames) + '.txt')
        with open(s_filename, 'w') as f:
            f.write(arguments_str)
            f.write(session_record)
            f.write(report)
            f.write('\n best acc : ' + str(best))
        return model, session_record, report, str(filenames)
    
    
    
def test(args, test_files, device, dataset, root_path):
    os.chdir(root_path)
    n_features = dataset.FP_set[1][2].shape[1]
    n_labels = torch.max(torch.cat([i[3] for i in dataset.FP_set])).item() + 1
    #model setting
    if args.gnn_model == 'sage':
        model = SAGE(in_feats=n_features, hid_feats=args.hidden_dim, out_feats=n_labels, num_layers = args.num_layers, aggregator_type = args.aggregator_type).to(device)
    
    PATH = os.path.join(args.checkpoint, args.dataset, args.gnn_model + '_' + args.aggregator_type + '_' + args.load_epoch + '_net.pth')
    model, opt, cur_epoch, loss, record, session_record, e_time = model_load(model, PATH, n_features, n_labels, args, device)
    model.eval()
    filenames = [fp[0] for fp in test_files]
    
    for filename in filenames:
        # get testfile from the dataset and do predict with trained model
        FP = dataset.__getitem_by_name__(filename)
        filename = FP[0]
        x = FP[4]
        node_labels = FP[3]
        node_features = torch.stack([FP[4].ndata[j] for j in [i for i in FP[4].ndata]]).T # shape : (n_node, n_features)
        node_features = node_features.type(torch.FloatTensor).to(device)
        if len(filenames) == 1:
            logit = model(x, node_features, batch_norm = False)
        else:
            logit = model(x, node_features)
        class_pred = logit.max(1)[1].type_as(node_labels).numpy()
        os.chdir(dataset.fps_path)
        df = gpd.read_file(os.path.join(filename, filename + '_labeled.shp'))
        df['obj_class'] = class_pred
        
        # make prediction directory
        output_path = os.path.join(root_path, 'output')
        os.chdir(output_path)
        if args.dataset not in os.listdir(output_path):
            os.mkdir(args.dataset)
        cur_output_path = os.path.join(output_path, args.dataset)
        os.chdir(cur_output_path)
        if filename in os.listdir(cur_output_path):
            shutil.rmtree(filename)
        
        # save prediction
        os.mkdir(os.path.join(cur_output_path, str(filename)))
        df.to_file(os.path.join(cur_output_path, str(filename), str(filename) + '_pred.shp'))
    
    os.chdir(root_path)