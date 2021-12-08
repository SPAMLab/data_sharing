from glob import glob

from sklearn.model_selection import train_test_split


import numpy as np
import pandas as pd
import rasterio
import tensorflow as tf

from utils import open_tif_image, create_new_folder_if_it_doesnt_exist

import segmentation_models as sm

from preprocessing import *



class TrainingPipeline:
    def __init__(self, imagePath, masksPath, savePath, hyperparameters, models, pathsAndShapes,
                 normalizingFunction):
        self.imagesPath = imagePath
        self.masksPath = masksPath
        self.savePath = savePath
        self.hyperparameters = hyperparameters
        self.models = models
        self.pathsAndShapes = pathsAndShapes
        self.normalizingFunction = normalizingFunction

    def train_from_generator(self):
        TrainModelsFromGenerator(generator=Generator(imagesPath=f"{self.imagesPath}",
                                                     masksPath=f"{self.masksPath}",
                                                     normalizingFunction=self.normalizingFunction,
                                                     batchSize=32),
                                 models=self.models,
                                 hyperparameters=self.hyperparameters,
                                 pathsAndShapes=self.pathsAndShapes,
                                 normalizingFunction=self.normalizingFunction,
                                 savePath=self.savePath
                                 ).train_model()





class Generator:
    def __init__(self, imagesPath, masksPath, normalizingFunction, batchSize=32, valSize=0.3):
        self.valSize = valSize
        self.imagesPath = imagesPath
        self.maskPath = masksPath
        self.trainImageNames, self.valImageNames = self.train_val_split()
        self.valLength = len(self.valImageNames)
        self.trainLength = len(self.trainImageNames)
        self.batchSize = batchSize
        self.normalizingFunction = normalizingFunction

    def __len__(self):
        return len(self.trainImageNames)

    def get_tif_images_names_from_a_folder(self):
        imagesPath = glob(f"{self.imagesPath}/*.tif")
        imageNames = [x.split("/")[-1] for x in imagesPath]
        return imageNames

    def train_val_split(self):
        trainImageNames, valImageNames = train_test_split(self.get_tif_images_names_from_a_folder(),
                                                          test_size=self.valSize, random_state=42)
        return trainImageNames, valImageNames

    def generator(self, val=False):
        """
        Yields the next training batch.
        Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
        """
        if val:
            amountOfImages = len(self.valImageNames)
            imageNames = self.valImageNames
        else:
            amountOfImages = len(self.trainImageNames)
            imageNames = self.trainImageNames
        while True:  # Loop forever so the generator never terminates

            # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size <= num_samples]
            for offset in range(0, amountOfImages, self.batchSize):
                # Get the samples you'll use in this batch

                batch_samples = imageNames[offset:offset + self.batchSize]

                # Initialise X_train and y_train arrays for this batch
                X_train = []
                y_train = []

                # For each example
                for batch_sample in batch_samples:
                    # Load image (X) and label (y)
                    img_name = batch_sample
                    img = open_tif_image(
                        f"{self.imagesPath}/{img_name}")
                    mask = open_tif_image(
                        f"{self.maskPath}/{img_name}")

                    # apply any kind of preprocessing                img = cv2.resize(img,(resize,resize))
                    # Add example to arrays
                    X_train.append(img)
                    y_train.append(mask)

                # Make sure they're numpy arrays (as opposed to lists)
                xTrain = np.array(X_train)
                yTrain = np.array(y_train)
                if self.normalizingFunction:
                    xTrain, yTrain = self.normalizingFunction(xTrain, yTrain, 2**16, 255)


                # The generator-y part: yield the next training batch
                yield xTrain, yTrain



class TrainModelsFromGenerator:
    def __init__(self, generator, pathsAndShapes, models, hyperparameters, normalizingFunction, savePath):
        self.hyperparameters = hyperparameters
        self.generator = generator
        self.trainGenerator = generator.generator()
        self.valGenerator = generator.generator(val=True)
        self.metrics = [sm.metrics.Precision(threshold=0.5), sm.metrics.Recall(threshold=0.5),
                        sm.metrics.FScore(threshold=0.5, beta=1), sm.metrics.IOUScore(threshold=0.5),]
        self.pathsAndShapes = pathsAndShapes
        self.models = models
        self.normalizingFunction = normalizingFunction
        self.savePath = savePath

    def train_model(self):
        for modelName, model in self.models.items():
            modelPointer = model
            model = model(self.hyperparameters["shape"])
            for lr in self.hyperparameters["learningRate"]:
                for batchSize in self.hyperparameters["batchSize"]:
                    self.generator.batchSize = batchSize
                    print(modelName, lr, batchSize, sep="#######################" )
                    create_new_folder_if_it_doesnt_exist(f"{self.savePath}/result/weights/{modelName}")
                    model.compile(tf.keras.optimizers.Adam(lr),
                                  loss=self.hyperparameters["loss"], metrics=self.metrics)
                    history = model.fit(self.trainGenerator, validation_data=self.valGenerator,
                                        validation_steps=self.generator.valLength // batchSize,
                                        steps_per_epoch=self.generator.trainLength // batchSize,
                                        epochs=self.hyperparameters["epochs"],
                                        batch_size=batchSize,
                                        callbacks=[tf.keras.callbacks.ModelCheckpoint(f"{self.savePath}/result/weights/{modelName}/{modelName}_{lr}_{batchSize}.hdf5",
                                                                                      monitor="val_loss",
                                                                                      save_weights_only=True,
                                                                                      save_best_only=True, verbose=1),
                                                   tf.keras.callbacks.EarlyStopping(patience=10)],
                                        verbose=1)
                    evaluate = EvaluateModels(currentModel=modelName, lr=lr, loss=self.hyperparameters["loss"],
                                              size=self.hyperparameters["shape"][0], model=modelPointer, metrics=self.metrics,
                                              pathsAndShapes=self.pathsAndShapes, batchSize=batchSize, normalizingFunction=self.normalizingFunction,
                                              channels=self.hyperparameters["shape"][-1], savePath=self.savePath)
                    evaluate.evaluate()
                    evaluate.predict()





class EvaluateModels:
    def __init__(self, currentModel, pathsAndShapes, lr, loss,size, model, metrics, batchSize, savePath, normalizingFunction=None, channels=6):
        self.results = {"model": [], "area": [], "precision": [], "recall": [], "f1_score": [], "mIoU": []}
        self.modelName = currentModel
        self.currentModel = f"{currentModel}_{lr}_{batchSize}"
        self.pathsAndShapes = pathsAndShapes
        self.lr = lr
        self.loss = loss
        self.size = size
        self.model = model
        self.metrics = metrics
        self.batchSize = batchSize
        self.normalizingFunction = normalizingFunction
        self.channels = channels
        self.savePath = savePath

    def load_data(self, name):
        img = open_tif_image(self.pathsAndShapes[name][0])
        X_test_area_1 = np.expand_dims(img, axis=0)
        Y_test_area_1 = np.expand_dims(open_tif_image(self.pathsAndShapes[name][1]), axis=0)
        if self.normalizingFunction:
            X_test_area_1, Y_test_area_1 = self.normalizingFunction(X_test_area_1, Y_test_area_1, 2**16, 255)
        return X_test_area_1, Y_test_area_1

    def load_model(self, name):
        size = self.pathsAndShapes[name][-1]
        model = self.model((size,size, self.channels))
        model.compile(tf.keras.optimizers.Adam(self.lr), loss=self.loss,
                      metrics=self.metrics)
        model.load_weights(f"{self.savePath}/result/weights/{self.modelName}/{self.currentModel}.hdf5")
        return model

    def evaluate(self):
        for name in list(self.pathsAndShapes.keys()):
            print(name)
            model = self.load_model(name)
            image, mask = self.load_data(name)
            with tf.device("CPU:0"):
                result = model.evaluate(image, mask)
            self.unserialize_result(result)
            self.results["area"].append(name)
        self.generate_table(self.results)

    def unserialize_result(self, result):
        self.results["model"].append(self.currentModel)
        self.results["precision"].append(result[1])
        self.results["recall"].append(result[2])
        self.results["f1_score"].append(result[3])
        self.results["mIoU"].append(result[4])

    def predict(self):
        for name in list(self.pathsAndShapes.keys()):
            print(name)
            if name == self.currentModel:
                pass
            else:
                model = self.load_model(name)
                image, _ = self.load_data(name)
                with tf.device("CPU:0"):
                    preds_train = model.predict(image, verbose=1)
                    preds_train_t = (preds_train > 0.5).astype(np.uint8)
                self.save_predicted_raster(name, preds_train_t)

    def generate_table(self, dataFrame):
        df = pd.DataFrame(dataFrame)
        create_new_folder_if_it_doesnt_exist(f"{self.savePath}/result/tables/{self.modelName}")
        df.to_csv(f"{self.savePath}/result/tables/{self.modelName}/{self.currentModel}_result.csv", index=False)

    def save_predicted_raster(self, name, predictionArray):
        dataset = rasterio.open(self.pathsAndShapes[name][1])
        # Get the metadata from the image
        meta = dataset.meta
        # Change the metadata to 1 (Result image will have just one channel)
        meta["count"] = 1
        # Nodata values = 0
        meta["nodata"] = 0
        # data type = uint8 (8bits)
        meta["dtype"] = "uint8"
        # Ajust array dimensions
        save = np.squeeze(predictionArray, axis=(0, 3))
        save = np.expand_dims(save, axis=0)
        # Save .tif image
        create_new_folder_if_it_doesnt_exist(f"{self.savePath}/result/predictions/{self.modelName}")
        with rasterio.open(f"{self.savePath}/result/predictions/{self.modelName}/{self.currentModel}_{name}.tif", 'w', **meta) as dst:
            dst.write(save)