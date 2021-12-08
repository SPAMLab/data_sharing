import random

from networks import *
from train import *
from utils import normalize_by_a_fixed_value, set_gpu
import shutil, os



saveFolder = "enterSaveFolderName"
sizes = "Enter size (int)"

dataPaths = {"train": {"image":"imagePath",
                       "shapefile":"shapePath"},
              "test1": {"image":"testImagePath",
                        "shapefile": "shapePath"},
              "test2": {"image":"testImagePath",
                        "shapefile": "shapePath"},
             "test3": {"image":"testImagePath",
                        "shapefile": "shapePath"}
             }




def generate_dataset_with_n_samples(numberOfSamples=390, path="", dest=""):
    image_names = glob(f"{path}/*.tif")
    x = random.sample(image_names, numberOfSamples)
    for file in x:
        shutil.copy(file,"/media/lucas/hd_sata/mestrado/trabalhos/mestrado/eval_patch_size_sampling/same_size_data/random/32/images")
        mask_name = file.replace("images", "masks")
        shutil.copy(mask_name,"/media/lucas/hd_sata/mestrado/trabalhos/mestrado/eval_patch_size_sampling/same_size_data/random/32/masks")



def generate_binary_mask(imagesPath = dataPaths, augment = False):
    for key in imagesPath.keys():
        image = imagesPath[key]["image"]
        shapefile = imagesPath[key]["shapefile"]
        BinaryMasks(imageRaster=Raster(image),
                    shapefile=Shapefile(shapefile),
                    savePath=f"../{saveFolder}/{key}").generate_binary_mask(save=True)
        if key == "train":
            generate_sampling_grid(image, sizes, "train", shapefile)
            # generate_random_sampling(image, sizes, "train", shapefile, 20)

            patch_data(image, key, sizes, augment)

def generate_sampling_grid(rasterPath, size, key, shapefile):
    samples = Sampling(Raster(rasterPath), size-1, size-1, savePath=f"../{saveFolder}/{key}/sampling"
             )
    # samples.generate_regular_sampling_polygons(0.3,save=True)
    samples.generate_random_sampling_polygons(5000,save=True)
    samples.select_only_the_tiles_that_intersect_polygons(Shapefile(f"../{saveFolder}/{key}/sampling/random_grid_size_{size-1}.geojson"),
                                                          Shapefile(shapefile), save=True)

def generate_random_sampling(rasterPath, size, key, shapefile, numOfPolygons):
    print("Random")
    samples = Sampling(Raster(rasterPath), size - 1, size - 1, savePath=f"../{saveFolder}/{key}/sampling")
    samples.generate_random_sampling_polygons(numOfPolygons, save=True)
    samples.select_only_the_tiles_that_intersect_polygons(
        Shapefile(f"../{saveFolder}/{key}/sampling/random_grid_size_{size - 1}.geojson"),
        Shapefile(shapefile), save=True)



def patch_data(imagePath, key, size, augment=False):
    PatchImage(imagePath=imagePath,
               savePath=f"../{saveFolder}/{key}/tiles/{size}/images",
               samplingShapefile=Shapefile(f"../{saveFolder}/{key}/sampling/s_join_random_grid_size_{size-1}.geojson")).patch_images(size, save=True)
    maskPath = imagePath.split("/")[-1].split(".")[0] + "_mask.tif"
    maskPath = f"../{saveFolder}/{key}/{maskPath}"
    PatchImage(imagePath=maskPath,
               savePath=f"../{saveFolder}/{key}/tiles/{size}/masks",
               samplingShapefile=Shapefile(f"../{saveFolder}/{key}/sampling/s_join_random_grid_size_{size-1}.geojson")).patch_images(size, save=True)
    if augment:
        augment_data(key,size)


def augment_data(key,size):
    print("Augmenting")
    Augmentation(imagesPath=f"../{saveFolder}/{key}/tiles/{size}/images",
                 masksPath=f"../{saveFolder}/{key}/tiles/{size}/masks",
                 size=sizes).augment_images()

def train_dl_model(size=sizes):
    HYPERPARAMETERS = {"learningRate": [0.00001, 0.0001],
                       "loss": "binary_crossentropy",
                       "epochs": 100,
                       "shape": (size, size, 6),
                       "batchSize": [8,16,32],
                       }

    models = {"unet": unet}

    pathsAndShapes = {
        "test1": [f"pathTest1",
                  f"maskTest1Path",
                  1024], #test image size
        "test2": [f"pathTest2",
                  f"maskTest2Path",
                  1024],
        "test3": [f".pathTest3",
                  f"maskTest3Path",
                  1024]
    }

    pipe = TrainingPipeline(
        imagePath=f"../{saveFolder}/train/tiles/{size}/images",
        masksPath=f"../{saveFolder}/train/tiles/{size}/masks",
        savePath=f"../art_results/{saveFolder}/{size}",
        hyperparameters=HYPERPARAMETERS, models=models, pathsAndShapes=pathsAndShapes,
        normalizingFunction=normalize_by_a_fixed_value)
    pipe.train_from_generator()

def preprocess():
    generate_binary_mask(augment=True)



if __name__ == '__main__':
    preprocess()
    set_gpu()
    train_dl_model(sizes)