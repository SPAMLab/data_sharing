import os
import skimage.io
import tensorflow as tf
from glob import glob
import numpy
import geopandas as gpd
import pandas as pd


def calculate_pixels_of_each_class(maskTilesPath):
    result = {"tile_name":[], "negative_pixels":[], "positive_pixels":[]}
    imagePaths = glob(f"{maskTilesPath}/*.tif")
    for path in imagePaths:
        image = open_tif_image(path)
        unique, counts = numpy.unique(image, return_counts=True)
        imagePixelsCount = dict(zip(unique, counts))
        result["tile_name"] .append(path.split("/")[-1])
        try:
            result["negative_pixels"].append(imagePixelsCount[0])
        except:
            result["negative_pixels"].append(0)

        try:
            result["positive_pixels"].append(imagePixelsCount[255])
        except:
            result["positive_pixels"].append(0)
    df = pd.DataFrame(result)
    df.to_csv("result.csv")


def calculate_expected_pixels(shapefilePath, pixelSize):
    gdf = gpd.read_file(shapefilePath)
    gdf["area"] = gdf.area
    gdf["pixels"] = gdf["area"]//(pixelSize*pixelSize)
    print("hey")
    gdf.to_file("test.geojson", driver="GeoJSON")



def create_new_folder_if_it_doesnt_exist(savePath):
    if not os.path.exists(savePath):
        os.makedirs(savePath)


def normalize_by_a_fixed_value(imagesArray, masksArray, imageNormalizingValue, maskNormalizingValue):
    return imagesArray/imageNormalizingValue, masksArray/maskNormalizingValue


def normalize_each_band(self, imagesArray, masksArray):
    for i in range(imagesArray.shape[3]):
        print("Max Value Band {} - Before Normalization: {}".format(i, imagesArray[:, :, :, i].max()))
        imagesArray[:, :, :, i] = imagesArray[:, :, :, i] / imagesArray[:, :, :, i].max()
        print("Max Value Band {} - After Normalization: {}".format(i, imagesArray[:, :, :, i].max()))
    print("#" * 15 + "Normalizing Masks" + "#" * 15)
    for i in range(masksArray.shape[3]):
        print("Max Value Band {} - Before Normalization: {}".format(i, masksArray[:, :, :, i].max()))
        masksArray[:, :, :, i] = masksArray[:, :, :, i] / masksArray[:, :, :, i].max()
        print("Max Value Band {} - After Normalization: {}".format(i, masksArray[:, :, :, i].max()))
    return imagesArray, masksArray

def open_tif_image(path):
    # type: (function) -> np.array
    """Function to open tif images.
        Parameters:
        input_path (string) = path where the image file is located;
        return

        np.array of the tif image"""

    # get the image_path.
    # read image
    im = skimage.io.imread(path, plugin="tifffile")
    return im


def set_gpu():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB * 2 of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 6)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)