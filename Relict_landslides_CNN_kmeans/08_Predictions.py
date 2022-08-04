#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LANDSLIDE PREDICTIONS


import rasterio
import sys
from glob import glob
import numpy as np
import segmentation_models as sm
from segmentation_models.utils import set_trainable
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from rasterio.plot import reshape_as_image, reshape_as_raster

import keras_ft

from utils import bands, augment, set_random_seeds

def get_arch(arch):
    if arch == 'unet':
        return sm.Unet
    elif arch == 'fpn':
        return sm.FPN
    elif arch == 'linknet':
        return sm.Linknet
    else:
        raise(Exception('Invalid arch name: {0}'.format(arch)))


if __name__ == '__main__':
    model = sys.argv[1] 
    aug_factor = int(sys.argv[2])
    n_classes = int(sys.argv[3])
    arch_name = sys.argv[4]
    #sample_size = int(sys.argv[6])
    Threshold = float(sys.argv[5])
    side = 1471 # DIMENSION OF TEST AREA

   

    arch = get_arch(arch_name)

    input_shape = (side, side, bands)

    smodel =arch(model, input_shape=(input_shape), encoder_weights=None, classes=2, activation='softmax', encoder_freeze=True)

    smodel.load_weights('PATH_TO_WEIGHTS_FILES.H5'.format(model, arch_name, n_classes, aug_factor)) # PATH TO WEIGHTS FILE FOR PREDCITION

    dataset = rasterio.open(r'PATH_TO_TEST_AREA_IMAGE.TIF').read() # PATH TO TEST AREA IMAGE 

    # IMAGE NORMALIZATION
    dataset =  (dataset - dataset.min()) * (1.0 / (dataset.max() - dataset.min())) 

    # RESHAPE IMAGES TO USE AS INPUT (SIDE, SIDE, BANDS)
    dataset = reshape_as_image(dataset)

    # INSERT A NEW DIMENSION (1, SIDE. SIDE, BANDS)
    dataset = np.expand_dims(dataset, axis=0)

    # PREDICT
    preds_train = smodel.predict(dataset, verbose=1)

    # ASSIGN VALUE 'ONE' FOR PREDICTIONS WITH RESULTS > THRESHOLD
    preds_train_t = (preds_train > Threshold).astype(np.uint8)    

    # OPEN THE ORIGINAL IMAGE
    dataset_orig = rasterio.open(r'PATH_TO_TEST_AREA_IMAGE.TIF')

    # GET THE METADATA FROM THE IMAGE
    meta = dataset_orig.meta
    print(meta)

    # CHANGE THE METADATA TO 1 (RESULT IMAGE WILL HAVE JUST ONE CHANNEL)
    meta["count"] = 2

    # NODATA VALUES = 0
    meta["nodata"] = 0

    # DATA TYPE = UINT8 (8BITS)
    meta["dtype"] = "uint8"

    # AJUST ARRAY DIMENSIONS
    save = np.squeeze(preds_train_t, axis=0)
    save = reshape_as_raster(save)

    # SAVE .TIF IMAGE
    with rasterio.open("PATH_TO_SAVE_THE_PREDICTION/tile_{3}_{0}_{1}_{2}_predict_threshold_{4}.tif".format(arch_name, n_classes, aug_factor, side, Threshold), 'w', **meta) as dst:
        dst.write(save)


    

