# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:17:58 2020

@author: user
"""
import cv2
from PIL import Image
import os
import fiona
from affine import Affine
from rasterio import features
import numpy as np
import networkx as nx
import shapely
# from shapely.geometry import Polygon as sPolygon
# from shapely.geometry import shape
from shapely.geometry import LineString as sLine
from shapely.geometry import Point as sPoint
from shapely.strtree import STRtree
import geopandas as gpd
from utils import *

def vectorization(input_path, output_path, filename, extension, bbox_on = False, min_area = 20):
    '''
    # Parameters: input_path, output_path, filename, extension, bbox_on = False, min_area = 5
    - input_path: A directory where the image to be vectorized is located
    - output_path: A directory to store pre-processed image and output polygon files
    - filename: The name of the target image without extension
    - extension: which extension the targe image use
    - bbox_on: get the bounding box of the input image (default: False)
    - min_area: a threshold parameter to remove small polygons
    
    # Usage
    Input the location and name of the target image to open and vectorize by using Rasterio lib.
    This function has several steps inbetween, like bufferizing polygons, filling the bubbles \
    (inner holes in/ or between polygons) by making them into polygon.
    
    # output
    A geodataframe extracted from the image
    '''
    
    os.chdir(input_path)
    img = cv2.imread(filename + extension)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, cur_bin_img = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)#binarization
    
    if bbox_on:
        #get the bounding box of image and split
        img = Image.fromarray(cur_bin_img)
        l, u, r, d = img.getbbox()#bbox of input image
        cur_bin_img = cur_bin_img[:d, :r]
        cur_bin_img = cur_bin_img[u:, l:]
        cur_bin_img = np.pad(cur_bin_img, pad_width = (4), mode='constant', constant_values = 255)
    os.chdir(output_path)
    cv2.imwrite('bin_' + filename + extension, cur_bin_img) # save processed image
    
    #get vectorized shapes using rasterio
    image = cur_bin_img
    aff = Affine(1, 0, 1, 0, -1, 0) # Adjust flip and symmetry
    mask = image == 255
    
    shapes = features.shapes(image, mask=mask, transform = aff)
    results = ({'properties': {'raster_val': i}, 'geometry': s} for i, (s, v) in enumerate(shapes))
    
    #save polygon in shape/GeoJSON format
    with fiona.open(
            filename + '_polys.shp', 'w',
            driver='ESRI Shapefile',
            schema={'properties': [('raster_val', 'int')],
                    'geometry': 'Polygon'}) as dst:
        dst.writerecords(results)
    
    #buffered polygons(shapely)
    df = gpd.read_file(filename + '_polys.shp')
    df = df.buffer(0.5, 10, cap_style = 2, join_style = 2)
    
    #remove useless and small polygons whose area is less than threshold
    to_pop = [i for i in range(len(df)) if df.iloc[i].area < min_area]
    # to_pop.append(-1)
    
    
    
    df = df.drop(df.index[to_pop])
    df = df.reset_index(drop = True)
    
    #fill the black area in the image(refered as bubbles)
    uunion = df.unary_union
    canvas = shapely.geometry.box(uunion.bounds[0], uunion.bounds[1], uunion.bounds[2], uunion.bounds[3])
    bubbles_ = canvas - uunion
    bubbles = []
    for i in bubbles_.geoms:
        if i.area > min_area:
            bubbles.append(i)
    if len(bubbles) > 0:
        gpd.GeoSeries(bubbles).to_file('bubbles.shp')
        df = df.append(gpd.GeoSeries(bubbles))
    
    #export buffered polygon geodataframe
    df2 = gpd.GeoDataFrame({'obj_id' : range(len(df)), 'obj_class' : np.zeros(len(df), np.int64), 'geometry':df})#initialize obj class features with 0s
    df2 = df2.reset_index()
    df2.to_file(filename + "_polys.shp")
    
    return df2

def build_FPgraph(input_path, output_path, filename, tail):
    '''
    Make floor plan graphs based on vectorized floor plans and adjacency files. 
    '''
    os.chdir(input_path)
    df = gpd.read_file(filename + tail + '.shp')
    adj = gpd.read_file(filename + '_adjacency.shp')
    nodes = dict()
    for i in range(len(df)):
        curpoly = df.iloc[i]['geometry']
        nodes[i] = {'polygon':curpoly, 'point' : (curpoly.centroid.x, curpoly.centroid.y), 'area' : curpoly.area}
    
    edges = dict()
    for i, row in adj.iterrows():
        p1, p2 = int(row['polygon1']), int(row['polygon2'])
        if (p1, p2) not in edges.values():
            edges[i] = (p1, p2)
        
    G = nx.Graph()              
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.values())

    return G, nodes, edges
    
def build_FPgraph_RAG(input_path, output_path, filename, tail, ext = '.shp'):
    # make a region adjacency graph for vector data which only has polygons
    os.chdir(input_path)
    df = gpd.read_file(filename + tail + ext)
    nodes = dict()
    for i in range(len(df)):
        curpoly = df.iloc[i]['geometry']
        nodes[i] = {'polygon':curpoly, 'point' : (curpoly.centroid.x, curpoly.centroid.y), 'area' : curpoly.area}
    
    # make RAG
    tree = STRtree(df['geometry'].buffer(1, 10, cap_style = 2, join_style = 2))
    n_sum = 0
    n_list = []
    p_dict = dict()
    pidx = 0
    for i in df['geometry'].buffer(1, 10, cap_style = 2, join_style = 2):
        p_dict[i.bounds] = pidx
        pidx += 1
    for i in range(len(df)):
        q_poly = df['geometry'][i].buffer(1, 10, cap_style = 2, join_style = 2)
        
        try:
            curnei = [p_poly for p_poly in tree.query(q_poly) if p_poly.intersects(q_poly) and p_poly != q_poly]
            cur_list = [p_dict[i.bounds] for i in curnei]
            n_sum += len(curnei)
            n_list.append(cur_list)
        except:
            n_list.append([])
    edges = dict()
    eidx = 0
    for i in range(len(n_list)):
        for j in n_list[i]:
            curtuple = (i, j)
            if curtuple not in edges.values() and (curtuple[1], curtuple[0]) not in edges.values() and curtuple[0] != curtuple[1]:
                edges[eidx] = curtuple
                eidx += 1
    G = nx.Graph()              
    G.add_nodes_from(nodes)
    G.add_edges_from(edges.values())
    
    # construct edge lines as shapely objs and export it in .shp format file    
    lines = gpd.GeoDataFrame(columns = ['eidx', 'polygon1', 'polygon2', 'geometry'])
    eidx = 0
    for i in edges.values():
        p1 = nodes[i[0]]
        p2 = nodes[i[1]]
        point1 = sPoint(p1['point'])
        point2 = sPoint(p2['point'])
        curline = sLine([point1, point2])
        lines = lines.append({'eidx':eidx, 'polygon1': i[0], 'polygon2':i[1], 'geometry':curline}, ignore_index = True)
        eidx += 1

    os.chdir(output_path)    
    lines.to_file(filename + "_adjacency.shp")
    return G, nodes, edges

def trim_vector(df, filename, output_path, min_area = 5):
    # trimming vectors by converting multipolygon geometric elements to polygon and remove small polygons
    
    if ('obj_id' not in df.columns) or ('geometry' not in df.columns): # df needs to have columns named 'obj_id' and 'geometry'
        return None
    df['obj_id'] = range(len(df)) # rearrange obj_id values for sure
    
    # for changing multipolygon obj to polygon obj
    # get the polygon with maximum area from multipolygon obj and swap 
    geom_dict = dict()
    for i in range(len(df)):
        geom_dict[i] = df.loc[i]['geometry']
    for k, v in geom_dict.items():
        if v.type == 'MultiPolygon':
            max_poly = v[0]
            for i in v.geoms:
                if i.area > max_poly.area:
                    max_poly = i
            geom_dict[k] = max_poly 
    df['geometry'] = geom_dict.values()
    
    # remove small polygons; if a polygon has area less than minimum area, pop it
    to_drop = []
    for i in range(len(df)):
        cur_poly = df.iloc[i]['geometry']
        if cur_poly.area < min_area:
            to_drop.append(i)
    df = df.drop(to_drop)
    # rearrange index and obj_id values
    df = df.reset_index().drop('index', axis = 1)
    df['obj_id'] = range(len(df))
    df.to_file(output_path + filename + '.shp')
    return df