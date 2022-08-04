import os
import random as rn

import numpy as np
import tensorflow as tf
from skimage import transform


# File with the metrics used in all processes

side = 32 # SIZE OF THE TILES
bands = 4 # NUMBER OF BANDS OF THE INPUT IMAGE
clusters_list = [2,4,6,8,10,12] # NUMBERS OF CLUSTERS
output_dir = 'PATH_TO_OUTPUT_DIRECTORY' # OUTPUT DIRECTORY WHERE DIRECTORIES WILL BE CREATED AND THE RESULTS SAVED


def augment(img, factor=10): # FUNCTION TO PERFORM AUGMENTATION IN THE IMAGE
    imgs = []

    angles = list(range(0, 360, 10))
    
    for angle in angles:
        tmp = transform.rotate(img, angle, mode='edge', preserve_range=True)
        #NO FLIP
        imgs.append(tmp)
        #V-FLIP
        imgs.append(tmp[::-1,:])
        #H-FLIP
        imgs.append(tmp[:,::-1])
        #HV-FLIP
        imgs.append(tmp[::-1,::-1])

        if len(imgs) >= factor:
            break

    return np.array(imgs[:factor])


def set_random_seeds():
    #begin set seed
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(42)

    tf.random.set_seed(42)
    #end set seed