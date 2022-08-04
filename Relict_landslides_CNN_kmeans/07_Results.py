#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# RESULTS

import sys
from glob import glob
import numpy as np
import segmentation_models as sm
from segmentation_models.utils import set_trainable
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import keras_ft
import pickle

from utils import side, bands, augment, set_random_seeds

X = np.load('PATH_TO_TRAIN_AREA_NUMPY_ARRAY_IMAGE.NPY') # PATH TO TRAIN AREA NUMPY ARRAT
y = np.load('PATH_TO_TRAIN_AREA_NUMPY_ARRAY_LABELS.NPY') # PATH TO LABELS OF TRAIN AREA NUMPY ARRAY 
y_mask = np.load('PATH_TO_TRAIN_AREA_NUMPY_ARRAY_MASK.NPY') # PATH TO LANDSLIDE MASK NUMPY ARRAY


y_mask_test = np.zeros(y_mask.shape +(2,)).astype('float32')

y_mask_test[:,:,:,0] = (y_mask == 0).astype('float32')
y_mask_test[:,:,:,1] = (y_mask == 1).astype('float32')



def get_arch(arch):
    if arch == 'unet':
        return sm.Unet
    elif arch == 'fpn':
        return sm.FPN
    elif arch == 'linknet':
        return sm.Linknet
    else:
        raise(Exception('Invalid arch name: {0}'.format(arch)))


# LOAD DICTIONARY WITH THE RESULTS
file = open('PATH_TO_DICTIONARY_RESULTS_VALUES.PKL', 'rb') # PATH TO DICTIONARY WITH RESULTS VALUES
resultados = pickle.load(file)
file.close()

if __name__ == '__main__':
    model = sys.argv[1]
    aug_factor = int(sys.argv[2])
    n_classes = int(sys.argv[3])
    arch_name = sys.argv[4]
  

    set_random_seeds()
  

    arch = get_arch(arch_name)

    input_shape = (side, side, bands)

   
    stopper = EarlyStopping(monitor='val_iou_score', min_delta=0.00001, patience=10, verbose=1, mode='max', restore_best_weights=True)
    chkpt = ModelCheckpoint('PATH_TO_CHECKPOINT.H5'.format(model, arch_name, n_classes, aug_factor), monitor='val_iou_score', verbose=1, save_best_only=True, save_weights_only=True, mode='max', save_freq='epoch') # Path to checkpoint created in the training process

    smodel = arch(model, input_shape=input_shape, encoder_weights=None, classes=2, activation='softmax', encoder_freeze=True)
    smodel.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.Precision(threshold=0.5),sm.metrics.Recall(threshold=0.5)])

    resultados['resultados_{0}_{1}_{2}_{3}'.format(model, n_classes, aug_factor, arch_name)] = [] # EMPTY LIST TO COLLECT THE RESULTS

    smodel.load_weights('PATH_TO_WEIGHTS_FROM_TRAINING_PROCESS.H5'.format(model, arch_name, n_classes, aug_factor)) # USES THE WEIGHTS LEARNED TO COMPUTE THE VALIDATION INDICES
    avalia = smodel.evaluate(X, y_mask_test)
    resultados['resultados_{0}_{1}_{2}_{3}'.format(model, n_classes, aug_factor, arch_name)].append(avalia)
    print('resultados_{0}_{1}_{2}_{3} ='.format(model, n_classes, aug_factor, arch_name), avalia)

    print(resultados)


    file = open('PATH_TO_SAVE_DICTIONARY_WITH_RESULTS_VALUES.PKL', 'wb') # SAVE THE DICITONARY WITH TE RESULTS VALUES
    pickle.dump(resultados, file)
    file.close()


    # RESULTS VISUALIZATION
    file = open('resultados.pkl', 'rb')
    teste = pickle.load(file)
    file.close()
    for k,v in teste.items():
        print(k,v)