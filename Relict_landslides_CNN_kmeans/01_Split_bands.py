#!/usr/bin/env python3
# -*- coding: utf-8 -*-

 # CODE TO SPLIT IMAGE IN BANDS
 
import rasterio

dataset = rasterio.open('PATH_TO_STUDY_AREA_IMAGE.TIF') # PATH TO ORIGINAL TIFF IMAGE FILE 

meta = dataset.meta.copy()
meta.update(compress='lzw')
meta.update(count=1)

band_file = 'PATH_TO_SAVE_EACH_BAND_OF_STUDY_AREA_IMAGE.TIF' # PATH TO EACH BAND FILE OF THE ORIGINAL IMAGE

for b in range(1, 5): # NUMBER OF BANDS IN THE IMAGE TO BE SPLITTED
    with rasterio.open(band_file.format(b), 'w+', **meta) as out:
        out_arr = out.read(1)
        out.write_band(1, dataset.read(b))

