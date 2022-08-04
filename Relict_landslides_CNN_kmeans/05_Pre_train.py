#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PRETRAINING USING THE DATASET CREATED FOR EACH CLUSTER
# USES DENSENET AS BACKBONE IN THIS PROCESS

import os
import random as rn
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras import optimizers, utils

import keras_ft
from utils import clusters_list, output_dir, set_random_seeds

if __name__ == '__main__':
    set_random_seeds()

    model = sys.argv[1] # MODEL IS DENSENET121
    n_clusters = int(sys.argv[2]) # NUMBER OF THE CLUSTER DATASET TO BE USED

    X = np.load('PATH_TO_NUMPY_ARRAYS_OF_THE_IMAGES.NPY'.format(n_clusters)) # PATH TO NUMPY ARRAY OF EACH CLUSTER
    y = np.load('PATH_TO_NUMPY_ARRAYS_OF_THE_LABELS.NPY'.format(n_clusters)) # PATH TO THE LABELS OF EACH CLUSTER

    y = utils.to_categorical(y, n_clusters)
    
    n_classes = y.shape[1]
    shape = X.shape[1:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, stratify=y, random_state=42) # SPLIT THE DATASET IN TRAIN AND TEST

    ft = keras_ft.KerasFineTuner()

    results_dir = os.path.join(output_dir, '%s_%s' % (model, n_clusters)) # PATH TO SAVE THE WEIGHTS FILE FROM THE PRETRAINING PROCESS
    os.makedirs(results_dir)

    ft.init_model(model, shape, n_classes, opt='adam', weights=None, results_dir=results_dir) # EXECUTION OF THE PRETRAINING
    ft.fit_top_model(   X_train,
                        y_train,
                        X_test,
                        y_test,
                        batch_size=32,
                        epochs=50)

