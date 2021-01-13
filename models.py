# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:08:07 2020

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import nn as dglnn
from dgl import utils as du
import dgl.function as fn

class EWConv(nn.Module):
    def __init__(self, in_feats, out_feats, edge_func, aggregator_type='mean'):
        super().__init__()
        self._in_src_feats, self._in_dst_feats = du.expand_as_pair(in_feats)
        self.out_feats = out_feats
        self.edge_func = edge_func
        self.aggregator_type = aggregator_type
        self.pool_func = nn.Linear(self._in_src_feats, self.out_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self.out_feats, self.out_feats, batch_first=True)
        self.self_func = nn.Linear(self._in_src_feats, self.out_feats)
        self.reset_parameters()
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.pool_func.weight, gain=gain)
        if self.aggregator_type == 'lstm':
            self.lstm.reset_parameters()
        # nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
    
    def udf_edge(self, edges):
        return {'edge_features': edges.data['w'], 'neighbors' : edges._src_data['h']}
    
    def udf_u_mul_e(self, nodes):
        m = self.edge_func
        weights = nodes.mailbox['edge_features']
        weights = torch.div(weights.squeeze(dim = 2), weights.sum(1)).unsqueeze(dim = 2)
        softmin_ed = m(weights)
        # num_edges = nodes.mailbox['edge_features'].shape[1]
        res = softmin_ed * nodes.mailbox['neighbors']
        if self.aggregator_type == 'sum':
            res = res.sum(axis = 1)
        elif self.aggregator_type == 'mean':
            res = res.mean(axis = 1)
        elif self.aggregator_type == 'max':
            res = res.max(axis = 1)[0]
        elif self.aggregator_type == 'lstm':
            batch_size = res.shape[0]
            hid = (res.new_zeros((1, batch_size, self.out_feats)), res.new_zeros((1, batch_size, self.out_feats)))
            _, (res, _) = self.lstm(res, hid) # only get hidden state
            res = res.permute(1, 0, 2)
        return {'h_reduced' : res}
    
    def forward(self, graph, feat, efeat):
        with graph.local_scope():
            feat_src, feat_dst = du.expand_as_pair(feat, graph)
            graph.srcdata['h'] = self.pool_func(feat_src) 
            graph.edata['w'] = efeat
            graph.update_all(self.udf_edge, self.udf_u_mul_e) 
            result = self.self_func(feat_dst) + graph.dstdata['h_reduced'].squeeze()
            
            return result

class DWGNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, edge_feats, num_layers, aggregator_type):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        # self.edge_funcs = torch.nn.ModuleList()
        if aggregator_type not in ['sum', 'mean', 'max', 'lstm']:
            raise KeyError("invalid aggregator type(sum, mean, max)")
        for layer in range(self.num_layers - 1):
            if layer == 0:
                # self.edge_funcs.append(torch.nn.Linear(edge_feats, in_feats * hid_feats))
                self.layers.append(EWConv(in_feats=in_feats, out_feats=hid_feats, edge_func = nn.Softmin(dim = 1), aggregator_type=aggregator_type))
            else:
                # self.edge_funcs.append(torch.nn.Linear(edge_feats, hid_feats*hid_feats))
                self.layers.append(EWConv(in_feats=hid_feats, out_feats=hid_feats, edge_func = nn.Softmin(dim = 1), aggregator_type=aggregator_type))
        # self.edge_funcs.append(torch.nn.Linear(edge_feats, hid_feats * out_feats))
        self.layers.append(EWConv(in_feats=hid_feats, out_feats=out_feats, edge_func = nn.Softmin(dim = 1), aggregator_type=aggregator_type))
            
        for layer in range(self.num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d((hid_feats)))
            
    def forward(self, graph, inputs, edge_features, batch_norm = True):
        h = inputs
        e = edge_features
        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                h = F.relu(self.layers[i](graph, h, e))
                if batch_norm == True:
                    h = self.batch_norms[i](h)
            else:
                h = self.layers[i](graph, h, e)

        h = F.log_softmax(h, dim=1)
       
        return h
    
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False #linear model
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            for layer in range(num_layers - 1):
                if layer == 0:
                    self.linears.append(nn.Linear(input_dim, hidden_dim))
                else:
                    self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x, batch_norm):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                if batch_norm == True:
                    h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                else:
                    h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)



class GCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.layers.append(dglnn.GraphConv(in_feats=in_feats, out_feats=hid_feats, norm = 'both'))
            else:
                self.layers.append(dglnn.GraphConv(in_feats=hid_feats, out_feats=hid_feats, norm = 'both'))
        self.layers.append(dglnn.GraphConv(in_feats=hid_feats, out_feats=out_feats, norm = 'both'))
        
        for layer in range(self.num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d((hid_feats)))
            
    def forward(self, graph, inputs, edge_feautes = None, batch_norm = True):
        h = inputs
        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                h = F.relu(self.layers[i](graph, h))
                if batch_norm == True:
                    h = self.batch_norms[i](h)
            else:
                h = self.layers[i](graph, h)

        h = F.log_softmax(h, dim=1)
       
        return h
    

class MPNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, edge_feats, num_layers, aggregator_type, residual = False):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.edge_funcs = torch.nn.ModuleList()
        if aggregator_type not in ['sum', 'mean', 'max']:
            raise KeyError("invalid aggregator type(sum, mean, max)")
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.edge_funcs.append(torch.nn.Linear(edge_feats, in_feats * hid_feats))
                self.layers.append(dglnn.NNConv(in_feats=in_feats, out_feats=hid_feats, edge_func = self.edge_funcs[layer], aggregator_type=aggregator_type, residual = residual))
            else:
                self.edge_funcs.append(torch.nn.Linear(edge_feats, hid_feats*hid_feats))
                self.layers.append(dglnn.NNConv(in_feats=hid_feats, out_feats=hid_feats, edge_func = self.edge_funcs[layer], aggregator_type=aggregator_type, residual = residual))
        self.edge_funcs.append(torch.nn.Linear(edge_feats, hid_feats * out_feats))
        self.layers.append(dglnn.NNConv(in_feats=hid_feats, out_feats=out_feats, edge_func = self.edge_funcs[self.num_layers - 1], aggregator_type=aggregator_type, residual = residual))
            
        for layer in range(self.num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d((hid_feats)))
            
    def forward(self, graph, inputs, edge_features, batch_norm = True):
        h = inputs
        e = edge_features
        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                h = F.relu(self.layers[i](graph, h, e))
                if batch_norm == True:
                    h = self.batch_norms[i](h)
            else:
                h = self.layers[i](graph, h, e)

        h = F.log_softmax(h, dim=1)
       
        return h
    
class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_layers, aggregator_type):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        # self.sumpool = dglnn.pytorch.glob.SumPooling()
        
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.layers.append(dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type=aggregator_type))
            else:
                self.layers.append(dglnn.SAGEConv(in_feats=hid_feats, out_feats=hid_feats, aggregator_type=aggregator_type))
        self.layers.append(dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type=aggregator_type))
            
        for layer in range(self.num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d((hid_feats)))
    
    def forward(self, graph, inputs, edge_feautes = None, batch_norm = True):
        h = inputs
        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                h = F.relu(self.layers[i](graph, h))
                if batch_norm == True:
                    h = self.batch_norms[i](h)
            else:
                h = self.layers[i](graph, h)

        h = F.log_softmax(h, dim=1)
       
        return h

class GIN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, num_layers, num_mlp_layers, aggregator_type):
        super().__init__()
        self.num_layers = num_layers
        self.layers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        for layer in range(self.num_layers - 1):
            if layer == 0:
                cur_func = MLP(num_mlp_layers, in_feats, hid_feats, hid_feats)
                self.layers.append(dglnn.GINConv(cur_func, aggregator_type))
            else:
                cur_func = MLP(num_mlp_layers, hid_feats, hid_feats, hid_feats)
                self.layers.append(dglnn.GINConv(cur_func, aggregator_type))
        cur_func = MLP(num_mlp_layers, hid_feats, hid_feats, out_feats)
        self.layers.append(dglnn.GINConv(cur_func, aggregator_type))
        
        for layer in range(self.num_layers - 1):
            self.batch_norms.append(nn.BatchNorm1d((hid_feats)))
        
    def forward(self, graph, inputs, edge_feautes = None, batch_norm = True):
        h = inputs
        for i in range(self.num_layers):
            if i != self.num_layers - 1:
                h = F.relu(self.layers[i](graph, h, batch_norm))
                if batch_norm == True:
                    h = self.batch_norms[i](h)
            else:
                h = self.layers[i](graph, h, batch_norm)

        h = F.log_softmax(h, dim=1)
       
        return h    
        
    