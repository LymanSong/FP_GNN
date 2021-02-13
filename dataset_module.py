# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 18:04:13 2020

@author: user
"""
import torch
from torch.utils.data import Dataset
import dgl
# import dgl.nn as dglnn
import os
import natsort
import numpy as np
import geopandas as gpd
import cv2
import shutil
from vectorize import *
from tqdm import tqdm

class FP_dataset(Dataset):
    '''
    [ Data processing step ]
        step 1: need to get preprocessed(svg2png, binarization) floor plan images vetorized.
        step 2: make adjacency matrix using vectorized floor plans
        step 3: extracts attribute matrix from polygons of vector files
        step 3-1: if we want to train the model, need to check whether labeled vector files exist
        step 4: all things done, ready to train/test the model
    '''
    def __init__(self, root_path, dataset_name, device, args, train = True):
        self.root_path = root_path #selectd datatset folder under the dataset folder[dataset/selected_dataset]
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(root_path, dataset_name)
        self.fps_path = os.path.join(root_path, dataset_name, 'fps')
        self.preprocessed = os.path.join(root_path, dataset_name, 'preprocessed')
        self.filename = os.listdir(self.fps_path)
        if os.path.exists(self.fps_path) == False:
            raise RuntimeError("Dataset not found.")
        
        if train == True:
            self.train = train
            self.label_path = os.path.join(self.dataset_path, 'obj_class')
        else:
            self.train = False
        step = self._check_step()
        if step < 4:
            self.data = self.process(step)
        self.FP_set = self._build_FP_set(device, args)
    
    def __len__(self):
        return len(os.listdir(self.fps_path))
    
    def _check_step(self):
        '''
        Check which step the model is. Return the step number indicating the next process.
        '''
        filename = os.listdir(self.fps_path)[0]
        if os.path.exists(self.preprocessed):
            if self.train == True:
                if not os.path.exists(os.path.join(self.fps_path, filename, filename + '_labeled.shp')):
                    raise RuntimeError('labeled vector files don\'t exist')
            if len(os.listdir(self.preprocessed)) == self.__len__():
                print('All the FP images are vectorized and has feature table. Ready to train/test!\n # of fps: {}'.format(self.__len__()))
                return 4
            else:
                print("The number of preprocessed attribute data does not match the number of floorplans. \n Gonna remove preprocessed folder and extract atrributes again...")
                shutil.rmtree(self.preprocessed)
                return 3
        else: #no preprocessed data -> need to extract attribute tables from the vector files
            os.chdir(self.fps_path)
            os.chdir(filename)
            if os.path.exists(filename + '_adjacency.shp'): #vectorized fp and adjacency matrix exist.
                print("step 3, extract node and edge attributes")
                return 3
            else:
                if self.train:
                    if os.path.exists(filename + '_labeled.shp'):  # need to get adjacency matrices.
                        return 2
                    else:
                        raise RuntimeError('no label data.')
                else:
                    if os.path.exists(filename + '_polys.shp'):
                        return 2
                    else: #the floorplan images are not vectorized yet.
                        return 1
    
    def process(self, step):
        '''
        process data depending on the current step
        '''
        data = []
        filenames = natsort.natsorted(self.filename)
        if step == 1:
            pbar = tqdm(range(self.__len__()), desc = 'making vectors and RAGs...')
            for pos in pbar:
                filename = filenames[pos]
                file_path = os.path.join(self.fps_path, filename)
                gdf = vectorization(file_path, file_path, filename, '.png')
                G, nodes, edges = build_FPgraph_RAG(file_path, file_path, filename, '_polys')
                data.append((filename, gdf, G, nodes, edges))
            write_data(data, os.path.join(self.dataset_path, 'data'))
            FPset = extract_attrs(self.dataset_path, self.preprocessed)
        elif step == 2:
            if self.train:
                pbar = tqdm(range(self.__len__()), desc = 'loading vectors and make RAG...')
                for pos in pbar:
                    filename = filenames[pos]
                    file_path = os.path.join(self.fps_path, filename)
                    gdf = gpd.read_file(os.path.join(file_path, filename + '_labeled.shp'))
                    G, nodes, edges = build_FPgraph_RAG(file_path, file_path, filename, '_labeled')
                    data.append((filename, gdf, G, nodes, edges))
                    print(filename)
                write_data(data, os.path.join(self.dataset_path, 'data'))
                FPset = extract_attrs(self.dataset_path, self.preprocessed, label = True)
            else:
                pbar = tqdm(range(self.__len__()), desc = 'loading vectors and make RAG...')
                for pos in pbar:
                    filename = filenames[pos]
                    file_path = os.path.join(self.fps_path, filename)
                    gdf = gpd.read_file(os.path.join(file_path, filename + '_polys.shp'))
                    G, nodes, edges = build_FPgraph_RAG(file_path, file_path, filename, '_polys')
                    data.append((filename, gdf, G, nodes, edges))
                write_data(data, os.path.join(self.dataset_path, 'data'))
                FPset = extract_attrs(self.dataset_path, self.preprocessed)
        else: #step == 3:
            if self.train:
                pbar = tqdm(range(self.__len__()), desc = 'loading vectors and graphs...')
                for pos in pbar:
                    filename = filenames[pos]
                    file_path = os.path.join(self.fps_path, filename)
                    gdf = gpd.read_file(os.path.join(file_path, filename + '_labeled.shp'))
                    G, nodes, edges = build_FPgraph(file_path, file_path, filename, '_labeled')
                    data.append((filename, gdf, G, nodes, edges))
                write_data(data, os.path.join(self.dataset_path, 'data'))
                FPset = extract_attrs(self.dataset_path, self.preprocessed, label = True)
            else:
                pbar = tqdm(range(self.__len__()), desc = 'loading vectors and graphs...')
                for pos in pbar:
                    filename = filenames[pos]
                    file_path = os.path.join(self.fps_path, filename)
                    gdf = gpd.read_file(os.path.join(file_path, filename + '_polys.shp'))
                    G, nodes, edges = build_FPgraph(file_path, file_path, filename, '_polys')
                    data.append((filename, gdf, G, nodes, edges))
                write_data(data, os.path.join(self.dataset_path, 'data'))
                FPset = extract_attrs(self.dataset_path, self.preprocessed)
        return data
    
    def _build_FP_set(self, device, args):
        '''
        gather processed data and convert it to trainable object.
        '''
        FP_set = []
        pbar = tqdm(range(self.__len__()), desc = 'loading...')
        filenames = natsort.natsorted(self.filename)
        for pos in pbar:
            filename = filenames[pos]
            file_path = os.path.join(self.fps_path, filename)
            if self.train:
                df = gpd.read_file(os.path.join(file_path, filename + '_labeled.shp'))
            else:
                df = gpd.read_file(os.path.join(file_path, filename + '_polys.shp'))
            os.chdir(os.path.join(self.preprocessed, filename))
            FP = dict()
            for j in os.listdir():
                FP[j[:-4]] = load_data(j[:-4])
            adj_dist, area_, moments, G, labels, nodes, edges, edge_dists = FP['adj_dist'], FP['areas'], FP['moments'], FP['G'], FP['labels'], FP['nodes'], FP['edges'], FP['edge_dists']
            
            #construct node features
            degrees = list(i[1] for i in G.degree)
            adj_dist = normalizing2(adj_dist)
            if args.gnn_model == 'dwgnn':
                edge_array = np.array(list(edge_dists.items()), dtype = object)
                edge_array[:, 1] = normalizing2(edge_array[:, 1], mode = args.feature_normalize)
                edge_dists = dict(edge_array)
            
            areas = np.expand_dims(np.array([i for i in area_.values()]), 1)
            areas = normalizing2(areas)
            
            moments42 = normalizing2(moments['zm42'].to_numpy())
            moments11 = normalizing2(moments['nu11'].to_numpy())
        
            degrees, _ = normalizing(pd.DataFrame(degrees), ['adj_dist'])
            # degrees = np.squeeze(degrees.to_numpy())
            
            features = np.array([degrees, areas, moments11, moments42]).T
            if self.train == True:
                labels = torch.LongTensor(labels)
            
            dgl_G = dgl.from_networkx(G).to(device) # DGLGraph type
            dgl_G.ndata['degrees'] = torch.FloatTensor(degrees).to(device)
            dgl_G.ndata['areas'] = torch.FloatTensor(areas).to(device)
            dgl_G.ndata['moments11'] = torch.FloatTensor(moments11).to(device)
            dgl_G.ndata['moments42'] = torch.FloatTensor(moments42).to(device)

            if args.gnn_model == 'dwgnn':
                for i in edge_dists.keys():
                    dgl_G.edges[i[0], i[1]].data['edge_dists'] = torch.Tensor([[edge_dists[i]]]).to(device)
            FP_set.append((filename, G, features, labels, dgl_G, edge_dists))
            os.chdir('../')
        return FP_set
    
    def __getitem__(self, idx):
        return self.FP_set[idx]
    
    def __getitem_by_name__(self, filename):
        for fp in self.FP_set:
            if fp[0].isnumeric() == False:
                if fp[0] == filename:
                    return fp
            else:
                if fp[0] == filename:
                    return fp
        return None
    def __get_class_num__(self):
        if 'cubicasa' in self.dataset_name:
            return 8
        else:
            return 9
