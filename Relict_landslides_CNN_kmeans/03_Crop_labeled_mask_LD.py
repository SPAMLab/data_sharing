#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rasterio
from skimage import io
from utils import side, bands

# CREATION OF THE LABELED DATASET (LD)
# TILES WITH SIZE 32X32 ARE CREATED FROM ORIGINAL IMAGE
# EACH TILE RECEIVES A LABEL AS LANDSLIDE OR NOT. IF THERE IS ONE PIXEL OF LANDSLIDE, THE TILE RECEIVES THE LANDSLIDE LABEL.
# IMAGE TILES ARE SAVED AS X FILE.
# LABEL TILES ARE SAVED AS Y FILE. 

if __name__ == '__main__':
    img = rasterio.open('PATH_TO_STUDY_AREA_IMAGE.TIF') # PATH TO ORIGINAL IMAGE TIFF FILE OF THE STUDY AREA 
    mask_file = 'PATH_TO_LANDSLIDE_MASK_IMAGE.TIF' # PATH TO LANDSLIDE MASK FILE CREATED IN THE 'RASTERIZE' STEP

    mask = io.imread(mask_file)
    mask[mask == 255] = 1

    w = img.width
    h = img.height

    img_band = np.zeros((h, w, bands)).astype('int16')

    crops_h = h // side
    crops_w = w // side

    X = np.zeros((crops_h * crops_w, side, side, bands)).astype('int16') # DEFINITION OF THE DIMENSIONS OF IMAGE TILES FILE
    y = np.zeros((crops_h * crops_w,)).astype('uint8') # DEFINITION OF THE DIMENSIONS OF LABELS FILE

    y_mask = np.zeros((crops_h * crops_w, side, side)).astype('uint8')

    for b in range(bands):
        img_band[:,:,b] = img.read(b+1) # CREATE TILES FOR EACH BAND SEPARATELY

    for i in range(crops_h):
        for j in range(crops_w):
            mask_crop = mask[i*side:(i+1)*side, j*side:(j+1)*side]
            y_mask[i*crops_w+j,:,:] = mask_crop

            if mask_crop.min() != mask_crop.max():
                y[i*crops_w + j] = 1 

            for b in range(bands):
                X[i*crops_w+j,:,:,b] = img_band[i*side:(i+1)*side, j*side:(j+1)*side, b] # 


    X = (X - X.min()) * (1.0 / (X.max() - X.min())) # NORMALIZATION OF THE IMAGE TILES
    X = X.astype('float32')

    print('scar: {0}'.format(np.sum(y == 1)))
    print('no scar: {0}'.format(np.sum(y == 0)))

    print('X.shape: ', X.shape)
    print('y.shape: ', y.shape)
    print('y_mask.shape: ', y_mask.shape)

    np.save('PATH_TO_DIRECTORY_TO_SAVE_X.NPY', X) # PATH TO SAVE IMAGE TILES AS NUMPY ARRAY
    np.save('PATH_TO_DIRECTORY_TO_SAVE_Y.NPY', y) # PATH TO SAVE LABELS AS NUMPY ARRAY
    np.save('PATH_TO_DIRECTORY_TO_SAVE_Y_MASK.NPY', y_mask) # PATH TO SAVE LANDSLIDE MASK AS NUMPY ARRAY

