#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# CODE TO RASTERIZE LANDSLIDE VECTOR LAYER
# THE MAKS CREATED WILL BE USED IN THE CNN TRAINING PROCESS

import rasterio
from rasterio import features
import geopandas

if __name__ == '__main__':
    img = rasterio.open('PATH_TO_STUDY_AREA_IMAGE') # PATH TO ORIGINAL TIFF IMAGE FILE
    cic = geopandas.read_file('PATH_TO_LANDSLIDE_VECTOR') # PATH TO LANDSLIDE INVENTORY VECTOR LAYER

    meta = img.meta.copy()
    meta.update(compress='lzw')
    meta.update(count=1)

    raster_mask = '.tif' # PATH TO SAVE MASK RASTER FILE OF THE LANDSLIDES

    with rasterio.open(raster_mask, 'w+', **meta) as out:
        out_arr = out.read(1)
        shapes = ((geom, 255) for geom in cic.geometry)

        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)

