# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:12:49 2020

@author: user
"""
import geopandas as gpd
import shapely
import os
import time
import cv2

root_path = os.getcwd()

os.chdir('./dataset/UOS_aug/fps')
filename = 'fl1'
for filename in os.listdir():
    image = cv2.imread(os.path.join(filename, filename + '.png'))
    
    df = gpd.read_file(os.path.join(filename, filename + '_labeled.shp'))
    
    affine = [0.7, 0, 0, -0.7, 0, 0]
    newgeo = df.affine_transform(affine)
    newgeo = newgeo.rotate(90, (0,0))
    newgeo = newgeo.affine_transform([1, 0, 0, 1, image.shape[1], -(image.shape[1])])
    newdf = gpd.GeoDataFrame(df[['obj_id', 'obj_class']], geometry = newgeo)
    if os.path.exists(filename + '_aug1') == False:
        os.mkdir(filename + '_aug1')
    newdf.to_file(os.path.join(filename + '_aug1', filename + '_aug1_labeled.shp'))
    
    # affine = [0.7, 0, 0, 0.7, 0, 0]
    # newgeo = df.affine_transform(affine)
    # newdf2 = gpd.GeoDataFrame(df[['obj_id', 'obj_class']], geometry = newgeo)
    # if os.path.exists(filename + '_aug2') == False:
    #     os.mkdir(filename + '_aug2')
    # newdf2.to_file(os.path.join(filename + '_aug2', filename + '_aug2_labeled.shp'))
    
    # affine = [1.2, 0, 0, 1.2, 0, 0]
    # newgeo = newdf.affine_transform(affine)
    # newdf3 = gpd.GeoDataFrame(df[['obj_id', 'obj_class']], geometry = newgeo)
    # if os.path.exists(filename + '_aug3') == False:
    #     os.mkdir(filename + '_aug3')
    # newdf3.to_file(os.path.join(filename + '_aug3', filename + '_aug3_labeled.shp'))