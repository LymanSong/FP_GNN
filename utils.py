# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 12:58:42 2020

@author: user
"""

import os
import fiona
import rasterio
from affine import Affine
from rasterio import features
from PIL.ImageDraw import Draw
from PIL import Image
import math
from math import factorial
import numpy as np
import cv2
import networkx as nx
import shapely
from shapely.geometry import Polygon as sPolygon
from shapely.geometry import shape
from shapely.geometry import LineString as sLine
from shapely.geometry import Point as sPoint
from shapely.strtree import STRtree
from shapely.affinity import affine_transform as af
import geopandas as gpd
import pandas as pd
import time
import pickle
import torch
import scipy.sparse as sp
import geojson as gj
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
import random as rd
import dgl
from tqdm import tqdm

def vis(img):
    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def inters(i, j, df):
    return df.iloc[i]['geometry'].intersects(df.iloc[j]['geometry'])

def touch(i, j, df):
    return df.iloc[i]['geometry'].touches(df.iloc[j]['geometry'])

def dist(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def distance(n1, n2, nodes):
    n1 = nodes[n1]
    n2 = nodes[n2]
    distance = np.sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)
    return distance

def write_data(data, name):
    with open(name + '.bin', 'wb') as f:
        pickle.dump(data, f)
        
def load_data(name):
    try:
        with open(name + '.bin', 'rb') as f:
            data = pickle.load(f)
    except:
        with open(name + '.txt', 'rb') as f:
            data = pickle.load(f)
    return data        


# functions to use in feature extraction
def get_adj_dist(G, nodes, nodes_, edges, edges_):
    nodes_id = list(G.nodes)
    dist_vector = np.zeros(len(nodes_id))
    for i in range(len(nodes_id)):
        cur_node = nodes_id[i]
        neighs = [k for k in nx.neighbors(G, cur_node)]
        distsum = 0
        for j in neighs:
            distsum += distance(cur_node, j, nodes)
        dist_vector[i] = distsum
    return dist_vector

def rasterize(polys, pidx, affinity = True):
    affine = [1, 0, 0, -1, 0, 0]
    poly = af(polys[pidx], affine)
    bbox = poly.bounds
    ras = features.rasterize(shapes = [poly], out_shape = (int(bbox[3] + 1), int(bbox[2]) + 1))# indices start with 0, add 1s to contain edges
    ras = np.where(ras == 1, 255, ras)
    test = np.where(ras == 0, 0, ras)
    test = ~test[int(bbox[1]):, int(bbox[0]):]
    test = np.pad(test, pad_width = (4), mode='constant', constant_values = 255)
    return test   

def Zernikemoment(src, n, m):
    def radialpoly(r, n, m):
        rad = np.zeros(r.shape, r.dtype)
        P = int((n - abs(m)) / 2)
        Q = int((n + abs(m)) / 2)
        for s in range(P + 1):
            c = (-1) ** s * factorial(n - s)
            c /= factorial(s) * factorial(Q - s) * factorial(P - s)
            rad += c * r ** (n - 2 * s)
        return rad
    
    if src.dtype != np.float32:
        src = np.where(src > 0, 0, 1).astype(np.float32)
    if len(src.shape) == 3:
        print('the input image src should be in gray scale')
        return

    H, W = src.shape
    if H > W:
        src = src[int((H - W) / 2): int((H + W) / 2), :]
    elif H < W:
        src = src[:, int((W - H) / 2): int((H + W) / 2)]

    N = src.shape[0]
    if N % 2:
        src = src[:-1, :-1]
        N -= 1
    x = range(N)
    y = x
    X, Y = np.meshgrid(x, y)
    R = np.sqrt((2 * X - N + 1) ** 2 + (2 * Y - N + 1) ** 2) / N
    Theta = np.arctan2(N - 1 - 2 * Y, 2 * X - N + 1)
    R = np.where(R <= 1, 1, 0) * R

    # get the radial polynomial
    Rad = radialpoly(R, n, m)

    Product = src * Rad * np.exp(-1j * m * Theta)
    # calculate the moments
    Z = Product.sum()

    # count the number of pixels inside the unit circle
    cnt = np.count_nonzero(R) + 1
    # normalize the amplitude of moments
    Z = (n + 1) * Z / cnt
    # calculate the amplitude of the moment
    A = abs(Z)
    # calculate the phase of the mement (in degrees)
    Phi = np.angle(Z) * 180 / np.pi

    return Z, A, Phi    

def normalizing(datalist, columns, mode = 'standard'):
    # normalize the input if its data type is pandas series.
    # convert it to a numpy array and process. returns pandas dataframe with single column.
    data_ary = datalist.values #returns 2-dimentional numpy array
    if mode == 'standard':
        scaler = preprocessing.StandardScaler()
        result = scaler.fit_transform(data_ary)
    elif mode == 'minmax':
        min_max_scaler = preprocessing.MinMaxScaler()
        result = min_max_scaler.fit_transform(data_ary)
    result_df = pd.DataFrame(result, columns = columns)
    return np.squeeze(result), result_df

def normalizing2(data_ary, mode = 'standard'):
    # normalize the input array(numpy array). normalization mode is selectable.
    # the input array has to be 2-dimensional. if it is 1-dimensional, expand its dimension automatically.
    # returns 1-dimensional array.
    if len(data_ary.shape) == 1:
        data_ary = np.expand_dims(data_ary, 1)
    if mode == 'standard':
        scaler = preprocessing.StandardScaler()
        result = scaler.fit_transform(data_ary)
    elif mode == 'minmax':
        min_max_scaler = preprocessing.MinMaxScaler()
        result = min_max_scaler.fit_transform(data_ary)
    return np.squeeze(result)

def get_moments(poly_objs, G, zm = 'zm42', mm = 'nu11'):    
    moments = pd.DataFrame(columns = [zm, mm])
    # pbar = tqdm(range(len(poly_objs)), desc = 'extracting moments...')
    for i in range(len(poly_objs)):
        img = rasterize(poly_objs, i, affinity = True)
        moments = moments.append(({zm:Zernikemoment(img, n = 4, m = 2)[1], mm:abs(cv2.moments(img)[mm])}), ignore_index =True)

    return moments

# functions for training and testing neural net model

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def class_trim(obj_class, change = {7:5, 8:6, 9:7}):
    try:
        obj_class = obj_class.to_list()
        if type(obj_class[0]) == str:
            obj_class = [int(i) for i in obj_class]
    except:
        print('here')
    prev = 0
    for i in np.unique(obj_class):
        if prev != i and (i - 1) != prev:
            start = True
            break
        else:
            prev = i
            start = False
            
    if start == True:
        new_cls = []
        for i in obj_class:
            # if type(num) != str:
            #     i = str(num)
            if i in change.keys():
                new_cls.append(change[i])
            else:
                new_cls.append(i)
        return new_cls
    else:
        return obj_class

def missing_node(G,nodedict = dict(), obj_class = dict()):
    nodelist = list(G.nodes)
    for idx in range(1, nodelist[-1]):
        if idx not in nodelist:
            G.add_node(idx, obj_class = 0, coordinate = (0,0))
            nodedict[idx] = (0,0)
            obj_class[idx] = 0
            print(idx)
    return G, nodedict, obj_class


def geojson_to_graph(filename, edgefilename):
    G = nx.Graph()
    nodes = dict()
    edges = dict()
    obj_class = dict()
    with open(filename + '.geojson', 'r') as f:
        ndata = gj.load(f)
        for feature in ndata['features']:
            curpoint = tuple(feature['geometry']['coordinates'])
            if curpoint not in nodes.values():
                nodes[feature['properties']['v_id']] = curpoint
                G.add_node(feature['properties']['v_id'], obj_class= feature['properties']['obj_class'], coordinate = curpoint)
                obj_class[feature['properties']['v_id']] = feature['properties']['obj_class']
    
    nodes_ = dict()
    for k, v in nodes.items():
        nodes_[v] = k
    with open(edgefilename + '.geojson', 'r') as f2:   
        edata= gj.load(f2)
        for feature in edata['features']:
            points = tuple(feature['geometry']['coordinates'])
            if len(points) >= 3:
                print('halt')
            p1, p2 = tuple(points[0]), tuple(points[1])
            curedge = (nodes_[p1], nodes_[p2])
            if curedge  not in edges.values():
                G.add_edge(curedge[0], curedge[1])
                edges[feature['properties']['e_id']] = curedge
            
    edges_ = dict()
    for k, v in edges.items():
        edges_[v] = k
    return G, nodes, nodes_, edges, edges_, obj_class    
         
def check_adj(gdata, framework):
    from torch_geometric import transforms
    from matplotlib import pyplot as plt
    if framework == 'torch_geometric':
        s2d = transforms.ToDense(1000)
        s2d(gdata)
        mat = gdata.adj.numpy()
    else:
        mat = gdata.adjacency_matrix_scipy().todense()
    _, ax = plt.subplots(figsize=(30, 30))
    ax.imshow(mat, cmap='Blues')
    plt.show()
    
def collate(samples):
    # The input `samples` is a list of tuples
    #  (filename, graph, label, ...).
    filenames, G, features, labels, graphs, edge_features = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.cat(labels), filenames

def get_edge_features(nodes, edges):
    dists = dict()
    length = len(edges)
    for k, e in edges.items():
        dists[e] = dist(nodes[e[0]], nodes[e[1]])
        dists[(e[1], e[0])] = dist(nodes[e[1]], nodes[e[0]])
    return dists

def extract_attrs(input_path, output_path, label = False,):
    os.chdir(input_path)
    if os.path.exists(output_path) == False:
        os.mkdir(output_path)
    data = load_data('data')
    FPset = []
    pbar = tqdm(range(len(data)), desc = 'extracting node attributes...')
    for d in pbar:
        filename, df, G, node_features, edges = data[d]
        poly_objs = dict()
        nodes = dict()
        areas = dict()
        for i in range(len(node_features)):
            poly_objs[i] = node_features[i]['polygon']
            nodes[i] = node_features[i]['point']
            areas[i] = node_features[i]['area']
        nodes_ = dict()
        for k, v in nodes.items():
            nodes_[v] = k
        edges_ = dict()
        for k, v in edges.items():
            edges_[v] = k
        if label == True:
            labels = class_trim(df['obj_class'])
        else:
            labels = None
        adj = nx.linalg.adjacency_matrix(G, weight = None)
        adj = adj + sp.eye(adj.shape[0])#+ adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
        adj_dist = get_adj_dist(G, nodes, nodes_, edges, edges_)
        edge_dists = get_edge_features(nodes, edges)
        moments = get_moments(poly_objs, G)
        os.chdir(output_path)
        if os.path.exists(filename) == False:
            os.mkdir(os.path.join(output_path, filename))
        
        write_data(labels, filename + '/labels')
        write_data(moments, filename + '/moments')
        write_data(G, filename + '/G')
        write_data(nodes, filename + '/nodes')
        write_data(poly_objs, filename + '/polygons')
        write_data(areas, filename + '/areas')
        write_data(edges, filename + '/edges')
        write_data(adj_dist, filename + '/adj_dist')
        write_data(edge_dists, filename + '/edge_dists')
        # print('{} has been saved'.format(filename))
        
        FP = dict()
        for j in os.listdir(filename):
            FP[j[:-4]] = load_data(filename +'/'+ j[:-4])
        FP['filename'] = filename
        FPset.append(FP)
        os.chdir('../')
    return FPset
    
def evaluate(model, graph, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        if args.gnn_model == 'gat':
                logits = torch.squeeze(logits, 1)
        # logits = logits
        labels = labels
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
def classification_repo(labels, preds, legend):
    report = ''
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)
    unique_class, class_predict = np.unique(preds, return_counts = True)
    unique_class_true, classes = np.unique(labels, return_counts = True)
    out = dict()
    j = 0
    for i in unique_class:
        out[i] = class_predict[j]
        j += 1
    for i in range(len(unique_class_true)):
        try:
            report += "class {}[predict]: {:>4}, percentage: {}\n".format(i, out[i], round(out[i]/len(preds), 2))
            print("class {}[predict]: {:>4}, percentage: {}".format(i, out[i], round(out[i]/len(preds), 2)))
        except(KeyError):
            print("class {}[predict]: {}, percentage: {}".format(i, None, None))
        report += "class {}[labeled]: {:>4}, percentage: {}\n".format(i, classes[i], round(classes[i]/len(preds), 2))
        print("class {}[labeled]: {:>4}, percentage: {}".format(i, classes[i], round(classes[i]/len(preds), 2)))
    
    from sklearn.metrics import classification_report as cr
    
    metric_res = cr(labels, preds, target_names = legend, digits = 4)
    print(metric_res)
    report += metric_res
    
    return report

def print_options(opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        message += '{:>25}: {:<30}\n'.format(str(k), str(v))
    message += '----------------- End -------------------'
    print(message)
    return message

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)  