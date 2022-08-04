#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# CREATION OF THE CLUSTER DATASET (CD)
# TILES WITH SIZE 32X32 ARE CREATED FROM ORIGINAL IMAGE
# EACH IMAGE TILE RECEIVES A LABEL ACCORDINGLY TO THE QUANTITY OF PIXELS BELONGING TO A CLUSTER. 
# IMAGE TILES ARE SAVED AS X FILE.
# CLUSTER LABEL TILES ARE SAVED AS Y FILE. 

import sys
import os
from glob import glob
import rasterio
import numpy as np
from spectral import kmeans
import random

from utils import side, bands, clusters_list # NUMBER OF CLUSTER ARE DEFINED IN THE CLUSTERS_LIST

if __name__ == '__main__':
    kmeans_iters = 50
    
    for n_clusters in clusters_list:
        clusters = list(range(n_clusters))
        start_clusters = None

        source_dir = 'PATH_TO_DIRECTORY_WITH_ALL_IMAGES' # Path to location of all images
        img_files = sorted(glob(os.path.join(source_dir, 'NAME_OF_THE_IMAGES.TIF'))) # Import each image separately for clusterization; All images have the same final name to facilitate the import process

        for img_file in img_files:
            print(n_clusters, img_file)

            X_file = img_file + '_X_int16.npy'
            y_file = img_file + '_y.npy'
            
            img = rasterio.open(img_file)

            w = img.width
            h = img.height
            
            crops_h = h // side 
            crops_w = w // side

            img_band = np.zeros((h, w, bands)).astype('int16')

            for b in range(bands):
                img_band[:,:,b] = img.read(b+1)

            print('k-means')
            mask, start_clusters = kmeans(img_band, n_clusters, kmeans_iters, start_clusters=start_clusters) # Execute K-means algorithm

            X_img = np.zeros((crops_h*crops_w, side, side, bands)).astype('int16') # Dimension of the image tiles file
            y = np.zeros((crops_h*crops_w,)).astype('uint8') # Dimensions of the label file. Labels are automatically created from the cluster process

            for i in range(crops_h):
                for j in range(crops_w):
                    mask_crop = mask[i*side:(i+1)*side, j*side:(j+1)*side]
                    maj_class = np.argmax([np.sum(mask_crop == c) for c in clusters])
                    y[i*crops_w + j] = maj_class
            
                    for b in range(bands):
                        X_img[i*crops_w+j,:,:,b] = img_band[i*side:(i+1)*side, j*side:(j+1)*side, b]

            np.save(X_file, X_img)
            np.save(y_file, y)

        X_files = sorted(glob(os.path.join(source_dir, '*_X_int16.npy')))
        y_files = sorted(glob(os.path.join(source_dir, '*_y.npy')))

        Xs = []
        ys = []

        for X_file, y_file in zip(X_files, y_files):
            print(X_file)
            Xs.append(np.load(X_file))
            ys.append(np.load(y_file))
            os.remove(X_file)
            os.remove(y_file)


        X = np.vstack(Xs)
        y = np.hstack(ys)

        X = (X - X.min()) * (1.0 / (X.max() - X.min())) # NORMALIZATION OF THE IMAGE TILES
        X = X.astype('float32')


        for c in clusters:
            print('class {0}: {1}'.format(c, np.sum(y == c)))

        

        print('X.shape: ', X.shape)
        print('y.shape: ', y.shape)
        
        # START OF THE CLUSTER BALANCING PROCESS

        print(y.shape[0])
        
        
        # CREATE LISTS TO BE ITERATED AND RETAIN INDICES OF CLUSTER CLASSES
        n = np.arange(y.shape[0])
        m = np.arange(n_clusters)

        
        # CREATE A DICTIONARY TO COLLECT THE LIST WITH THE INDICES OF CLUSTER CLASSES
        classes = {}

      
        # COLLECT THE CLUSTER CLASSES INDICES
        for i in m:
            classes['classe_%s' % i] = []
            for p in n:
                if y[p] == i:
                    classes['classe_%s' % i].append(p)    

        

        
        # MEAN CALCULATION OF THE CLUSTER CLASSES
        div = len(classes) - 1   
        total = y.shape[0] - len(classes['classe_0']) 
        media = int(total / div)
        print('media =', media)  

        
        # SUBTRACTION OF CLASS_0 AND MEAN 
        dif = int(len(classes['classe_0']) - media)

        print('diferença entre a classe_0 e a media = ', dif)


        # RANDOMLY SELECTION OF CLASS_0 INDICES THAT WILL BE DELETED FOR BALANCING THE CLASSES
        random = np.random.choice(classes['classe_0'], dif, replace=False)

        
        # DELETION OF INDICES PREVIOUSLY SELECTED
        X = np.delete(X, random, 0)
        y = np.delete(y, random)

        print('X.shape balanceada: ', X.shape)
        print('y.shape balanceada: ', y.shape)

        for c in clusters:
            print('class {0}: {1}'.format(c, np.sum(y == c)))


           

        np.save('PATH_TO_DIRECTORY_TO_SAVE_X_{0}_CLUSTER.NPY'.format(n_clusters), X) # PATH TO SAVE THE BALANCED DATASET OF EACH CLUSTER FOR THE IMAGE TILES; SAVE AS 'X_2_CLUSTER', 'X_4_CLUSTER' AND SO ON
        np.save('PATH_TO_DIRECTORY_TO_SAVE_Y_{0}_CLUSTER.NPY'.format(n_clusters), y) # PATH TO SAVE THE BALANCED DATASED OF EACH CLUSTER FOR THE LABELS. SAVE AS Y_2_CLUSTER', 'Y_4_CLUSTER' AND SO ON
