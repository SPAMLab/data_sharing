#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# CNN TRAINING FOR SEMANTIC SEGMENTATION

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

from utils import side, bands, augment, set_random_seeds

def load_data(aug_factor):  # FUNCTION TO COLLECT ONLY THE IMAGE WITH POSITIVE CLASS (LABELED AS LANDSLIDE) FOR THE TRAINING PROCESS
    X = np.load('PATH_TO_NUMPY_ARRAY_OF_THE_IMAGE_TILES.NPY') # PATH TO NUMPY ARRAY OF IMAGE TILES FROM LD
    y = np.load('PATH_TO_NUMPY_ARRAY_OF_THE_LABEL.NPY') # PATH TO NUMPY ARRAY OF LABELS FORM LD
    y_mask = np.load('PATH_TO_NUMPY_ARRAY_OF_THE_LANDSLIDE_MASK.NPY') # PATH TO NUMPY ARRAY OF LANDSLIDE MASK

    X_one = X[y == 1,:,:,:]
    y_one_mask = y_mask[y == 1,:,:]

    X_one_aug = np.zeros((X_one.shape[0]*aug_factor, side, side, bands))
    y_one_aug_mask = np.zeros((y_one_mask.shape[0]*aug_factor, side, side))

    for i in range(X_one.shape[0]):
        X_one_aug[i*aug_factor:(i+1)*aug_factor,:,:,:] = augment(X_one[i], factor=aug_factor)
        y_one_aug_mask[i*aug_factor:(i+1)*aug_factor,:,:] = augment(y_one_mask[i], factor=aug_factor)

    y_one = np.ones((X_one_aug.shape[0],))

    X_ = X_one_aug #np.vstack((X_zero, X_one_aug))
    y_ = y_one #np.hstack((y_zero, y_one))
    y_mask_ = y_one_aug_mask #np.vstack((y_zero_mask, y_one_aug_mask))

    y_mask_ohe = np.zeros(y_mask_.shape +(2,)).astype('float32')

    y_mask_ohe[:,:,:,0] = (y_mask_ == 0).astype('float32')
    y_mask_ohe[:,:,:,1] = (y_mask_ == 1).astype('float32')

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
    sss_valid = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)

    for tmp_idx, test_idx in sss_test.split(X_, y_):
        X_tmp, X_test = X_[tmp_idx], X_[test_idx]
        y_tmp, y_test = y_[tmp_idx], y_[test_idx]
        y_mask_tmp, y_mask_test = y_mask_ohe[tmp_idx], y_mask_ohe[test_idx]

    for train_idx, valid_idx in sss_valid.split(X_tmp, y_tmp):
        X_train, X_valid = X_tmp[train_idx], X_tmp[valid_idx]
        y_train, y_valid = y_tmp[train_idx], y_tmp[valid_idx]
        y_mask_train, y_mask_valid = y_mask_tmp[train_idx], y_mask_tmp[valid_idx]

    return X_train, X_valid, X_test, y_train, y_valid, y_test, y_mask_train, y_mask_valid, y_mask_test

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
    model = sys.argv[1] # MODEL USED AS BACKBONE (DENSENET121)
    aug_factor = int(sys.argv[2]) # AUGMENTATION FACTOR (30 OR 50)
    n_classes = int(sys.argv[3]) # NUMBER OF THE CLUSTER DATASET
    arch_name = sys.argv[4] # CNN USED (FPN, UNET OR LINKNET)
    #sample_size = int(sys.argv[6])

    set_random_seeds()
    weights_file = glob('PATH_TO_WEIGHTS_FROM_PRETRAINING_STEP.H5'.format(model, n_classes))[0] # Path to weights file from the pretraining process

    arch = get_arch(arch_name)

    input_shape = (side, side, bands)

    X_train, X_valid, X_test, y_train, y_valid, y_test, y_mask_train, y_mask_valid, y_mask_test = load_data(aug_factor) # Execute the augmentation process   

    ft = keras_ft.KerasFineTuner()
    ft.init_model(model, input_shape, n_classes, opt='adam', weights=None) # Prepare a model for the fine tuning process
    ft.load_weights(weights_file)

    new_model = Model(inputs=ft.model.input, outputs=ft.model.layers[-5].output) # NEW MODEL CREATED TO EXECUTE THE FINE TUNING

    for l in new_model.layers:
        l.trainable = False

    tmp_weights = 'PATH_TO_SAVE_THE_WEIGHTS_FINE_TUNED_BY_THIS_MODEL.H5'.format(model, n_classes, aug_factor) # PATH TO SAVE THE WEIGHTS FROM THE TRAINING PROCESS
    new_model.save_weights(tmp_weights)

    stopper = EarlyStopping(monitor='val_iou_score', min_delta=0.00001, patience=10, verbose=1, mode='max', restore_best_weights=True) # DEFINE A STOPPER FOR THE TRAINING
    chkpt = ModelCheckpoint('PATH_TO_SAVE_CHECKPOINTS_FROM_THE_TRAINNG_PROCESS.H5'.format(model, arch_name, n_classes, aug_factor), monitor='val_iou_score', verbose=1, save_best_only=True, save_weights_only=True, mode='max', save_freq='epoch') # Path to save the checkpoints of the training process

    smodel = arch(model, input_shape=input_shape, encoder_weights=tmp_weights, classes=2, activation='softmax', encoder_freeze=True) # DEFINE THE METRICS OF TRAINING
    smodel.compile('Adam', loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.iou_score]) # COMPILE THE MODEL

    # EXECUTE THE TRAINING PROCESS
    smodel.fit( X_train,  
                y_mask_train, 
                batch_size=32, 
                epochs=300, 
                validation_data=(X_valid, y_mask_valid), 
                callbacks=[stopper, chkpt],
                verbose=1)

    #smodel.load_weights('F:\\Guilherme\\Dados_Deep_Learning\\Imagens_CBERS4A\\results_NDVI_sem_oceano_2\\{0}_{1}_{2}_{3}_seg_weights_chkpt.h5'.format(model, arch_name, n_classes, aug_factor))
    #avalia = smodel.evaluate(X_test, y_mask_test)
    #print('resultados_{0}_{1}_{2}_{3} ='.format(model, n_classes. aug_factor. arch_name), avalia)