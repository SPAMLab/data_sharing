from glob import glob
import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from utils import create_new_folder_if_it_doesnt_exist
import seaborn as sns
import rasterio as rio

sizes = "32"

class ImageMetrics:
    def __init__(self, imagePath, maskPath, savePath):
        self.imagePath = imagePath
        self.maskPath = maskPath
        self.savePath = savePath

    def calculate_TP_FP_FN(self, area):
        image = rio.open(self.imagePath).read()
        meta = rio.open(self.imagePath).meta
        mask = rio.open(self.maskPath).read()


        TP = image + mask
        TP = TP == 256
        FP = TP - image
        FP = FP == 255
        FN = mask - TP
        FN = FN == 255
        result = {"TP": TP, "FP": FP, "FN": FN}

        for key, value in result.items():

            create_new_folder_if_it_doesnt_exist(f"{self.savePath}/area_{area}")
            with rio.open(f"{self.savePath}/area_{area}/{key}_area_{area}.tif", 'w', **meta) as dst:
                dst.write(value.astype("uint8"))
        data = self.calculate_pixels(area)
        self.plot_hist(data,area)

    def calculate_pixels(self,area):
        result = {}
        for i in ["TP", "FP", "FN"]:
            image = rio.open(f"{self.savePath}/area_{area}/{i}_area_{area}.tif").read()
            (unique, counts) = np.unique(image, return_counts=True)
            result[i] = counts[1]
        return result

    def plot_hist(self, data,area):
        print(list(data.values()))
        colors = ["#04d253", "#ed0e0e", "#f7ff0f"]
        sns.barplot(list(data.keys()), list(data.values()), palette=colors)
        plt.savefig(f"{self.savePath}/area_{area}/hist_area_{area}.png",dpi=90)


class BestOverallResult:
    def __init__(self, area1ResultPath, area2ResultPath, area3ResultPath, sizes=[32,64,128,256]):
        self.sizes = sizes
        self.model1Result = {}
        self.better_results = {"size":[],"path":[], "f1":[], "area":[]}
        for size in sizes:
            self.model1Result[size] = self._get_data_path(path=area1ResultPath, _size=size)
            self.model1Result[size].extend(self._get_data_path(path=area1ResultPath, _size=size, network="unet"))
            self.model1Result[size].extend(self._get_data_path(path=area2ResultPath, _size=size))
            self.model1Result[size].extend(self._get_data_path(path=area2ResultPath, _size=size, network="unet"))
            self.model1Result[size].extend(self._get_data_path(path=area3ResultPath, _size=size))
            self.model1Result[size].extend(self._get_data_path(path=area3ResultPath, _size=size, network="unet"))


    def get_best_overall_result_of_each_test_area(self):
        bestTestPath = {32:{},64:{}, 128:{},256:{}}
        # loop over keys
        for size in self.sizes:
            bestResults = {"test1": 0, "test2": 0, "test3": 0}
            for i in range(len(self.model1Result[size])):
                path = self.model1Result[size][i]
                # open tables of model1
                df1 = pd.read_csv(path)
                area1 = [df1["f1_score"][0]]
                area2 = [df1["f1_score"][1]]
                area3 = [df1["f1_score"][2]]
                if max(area1) > bestResults["test1"]:
                    bestTestPath[size]["test1"] = path
                    bestResults["test1"] = max(area1)
                    self.better_results["size"].append(size)
                    self.better_results["path"].append(path)
                    self.better_results["f1"].append(max(area1))
                    self.better_results["area"].append("test1")
                if max(area2) > bestResults["test2"]:
                    bestTestPath[size]["test2"] = path
                    bestResults["test2"] = max(area2)
                    self.better_results["size"].append(size)
                    self.better_results["path"].append(path)
                    self.better_results["f1"].append(max(area2))
                    self.better_results["area"].append("test2")
                if max(area3) > bestResults["test3"]:
                    bestTestPath[size]["test3"] = path
                    bestResults["test3"] = max(area3)
                    self.better_results["size"].append(size)
                    self.better_results["path"].append(path)
                    self.better_results["f1"].append(max(area3))
                    self.better_results["area"].append("test3")
        df = pd.DataFrame(self.better_results)
        df.to_csv("better.csv")
        return bestTestPath

    @staticmethod
    def _get_data_path(path, _size, network="unet"):
        return glob(f"{path}/{_size}/result/tables/{network}/*.csv")


class Results:
    def __init__(self, resultPath, resultPath1, resultPath2, sizes=[32, 64, 128, 256]):
        self.model1 = {}
        self.model2 = {}
        self.model3 = {}
        self.i = 0
        self.best_results = {"path":[], "f1":[], "f2":[], "f3":[], "sum":[]}
        self.sizes = sizes
        for size in sizes:
            self.model1[size] = self._get_data_path(path=resultPath, _size=size)
            self.model1[size].extend(self._get_data_path(path=resultPath, _size=size, network="unet"))
            self.model2[size] = self._get_data_path(path=resultPath1, _size=size)
            self.model2[size].extend(self._get_data_path(path=resultPath1, _size=size, network="unet"))
            self.model3[size] = self._get_data_path(path=resultPath2, _size=size)
            self.model3[size].extend(self._get_data_path(path=resultPath2, _size=size, network="unet"))

    @staticmethod
    def _get_data_path(path, _size, network="unet"):
        return glob(f"{path}/{_size}/result/tables/{network}/*.csv")

    def find_best_result(self, tablePathString):
        bestModel = {}
        for size in self.sizes:
            best_f1_sum = 0
            bestModel[size] = {}
            for table in tablePathString[size]:
                df = self.open_table(table)
                if df["f1_score"].sum() > best_f1_sum:
                    best_f1_sum = df["f1_score"].sum()
                    bestModel[size] = table
                    self.best_results["path"].append(table)
                    self.best_results["f1"].append(df["f1_score"][0])
                    self.best_results["f2"].append(df["f1_score"][1])
                    self.best_results["f3"].append(df["f1_score"][2])
                    self.best_results["sum"].append(df["f1_score"].sum())
        return bestModel

    def get_best_results(self):
        a = self.find_best_result(self.model1)
        self.i += 1
        self.best_results = {"path": [], "f1": [], "f2": [], "f3": [], "sum": []}
        b = self.find_best_result(self.model2)
        self.i += 1
        self.best_results = {"path": [], "f1": [], "f2": [], "f3": [], "sum": []}
        c = self.find_best_result(self.model3)

        return a,b,c


    def copy_best_results_to_folder(self):
        a, b, c = self.get_best_results()
        for size in self.sizes:
            self._get_best_files(size,a)
            self._get_best_files(size, b)
            self._get_best_files(size, c)
    def copy_file_to_a_folder(self, source, destination):
        copyfile(source, destination)

    def _get_best_files(self, size, filePath):
        src = filePath[size].replace(".csv", ".tif").replace("tables", "predictions")
        folder = f"../art_results/best_results/{filePath[size].split('/')[2]}/{size}"
        create_new_folder_if_it_doesnt_exist(folder)
        test1Name = src.replace("result.tif", "test1.tif")
        test2Name = src.replace("result.tif", "test2.tif")
        test3Name = src.replace("result.tif", "test3.tif")

        self.copy_file_to_a_folder(test1Name, f"{folder}/{test1Name.split('/')[-1]}")
        self.copy_file_to_a_folder(test2Name, f"{folder}/{test2Name.split('/')[-1]}")
        self.copy_file_to_a_folder(test3Name, f"{folder}/{test3Name.split('/')[-1]}")



    @staticmethod
    def open_table(tablePath):
        return pd.read_csv(tablePath)








def read_data(tablePath1):
    df = pd.read_csv(tablePath1)

    result = [df["recall"][0], df["mIoU"][0], df["precision"][0],df["f1_score"][0], df["recall"][0]]
    result1 = [df["recall"][1], df["mIoU"][1], df["precision"][1], df["f1_score"][1], df["recall"][1]]
    result2 = [df["recall"][2], df["mIoU"][2], df["precision"][2], df["f1_score"][2], df["recall"][2]]

    return result, result1, result2


def plot_results(label, result1,result2,result3):
    linewidth = 3
    plt.plot(label, result1, label='Regular', color="black", linewidth=linewidth)
    plt.plot(label, result2, linestyle="dashed", label='Random', color="red", alpha=0.8,linewidth=linewidth)
    plt.plot(label, result3, linestyle="dotted", label='Random', color="blue", alpha=0.8,linewidth=linewidth)
    # plt.plot(label, resultTest1, label='Regular', color="green", alpha=0.7)


def plot_best_result(tablePath1, tablePath2=None, tablePath3=None, pathOfEachTest=None):
    result, result1, result2 = read_data(tablePath1)
    result6, result7, result8 = read_data(tablePath2)
    result9, result10, result11 = read_data(tablePath3)
    resultTest1,_,_ = read_data(pathOfEachTest["test1"])
    resultTest2,_,_ = read_data(pathOfEachTest["test2"])
    resultTest3,_,_ = read_data(pathOfEachTest["test3"])

    categories = ['Recall', 'IoU', 'Precision', 'F1', '']
    
    # if sizes != str(2):
    #     categories = ['', '', '', '', '']


    label_loc = np.linspace(start=0, stop=2 * np.pi, num=5)
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(311, polar=True)
    plot_results(label_loc,result, result6, result9)

    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories, fontsize=12)
    lines, labels = plt.rgrids((0.2, 0.4, 0.6, 0.8, 1))
    ax.tick_params(pad=20)
    # plt.legend()

    ax = plt.subplot(312, polar=True)
    plot_results(label_loc, result1, result7, result10)

    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories, fontsize=12)
    lines, labels = plt.rgrids((0.2, 0.4, 0.6, 0.8, 1))
    ax.tick_params(pad=20)
    # plt.legend()

    ax = plt.subplot(313, polar=True)
    plot_results(label_loc, result2, result8, result11)



    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories, fontsize=12)
    lines, labels = plt.rgrids((0.2, 0.4, 0.6, 0.8, 1))
    ax.tick_params(pad=20)
    # plt.legend()
    plt.subplots_adjust(bottom=4, top=5)
    plt.savefig(f'../art_results/graphs/test/{sizes}_1regular.png', bbox_inches='tight')
    # plt.show()

# a, b, c = Results("../art_results/5_bands_regular", "../art_results/5_bands_terrain_regular",
#                      "../art_results/5_bands_ndvi_regular").get_best_results()
# Results("../art_results/5_bands_regular", "../art_results/5_bands_terrain_regular",
#                      "../art_results/5_bands_ndvi_regular").copy_best_results_to_folder()
# d = BestOverallResult("../art_results/5_bands_regular", "../art_results/5_bands_terrain_regular",
#                 "../art_results/5_bands_ndvi_regular").get_best_overall_result_of_each_test_area()

ImageMetrics("../art_results/best_results/5_bands_regular/128/unet_1e-05_32_test1.tif",
             "../5_bands_regular/test1/test_area_1_mask.tif",
             "../art_results/best_results/metrics").calculate_TP_FP_FN(1)

ImageMetrics("../art_results/best_results/5_bands_ndvi_regular/64/unet_1e-05_8_test2.tif",
             "../5_bands_regular/test2/test_area_2_mask.tif",
             "../art_results/best_results/metrics").calculate_TP_FP_FN(2)

ImageMetrics("../art_results/best_results/5_bands_ndvi_regular/32/unet_0.0001_8_test3.tif",
             "../5_bands_regular/test3/test_area_3_mask.tif",
             "../art_results/best_results/metrics").calculate_TP_FP_FN(3)

# for size in [32,64,128,256]:
#     sizes = str(size)
#     plot_best_result( a[size],
#                       b[size],
#                       c[size],
#                       d[size],
#                   )
