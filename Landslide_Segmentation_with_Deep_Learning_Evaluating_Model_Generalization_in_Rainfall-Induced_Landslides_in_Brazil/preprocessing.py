import rasterio as rio
import geopandas as gpd
from shapely.ops import cascaded_union
from shapely.geometry import  Polygon
from rasterio.features import rasterize
import numpy as np
import os
from glob import glob
import earthpy.spatial as es
from rasterio.mask import mask
from rasterio.plot import reshape_as_image, reshape_as_raster
from shapely.geometry import box
import pandas as pd
from utils import open_tif_image, create_new_folder_if_it_doesnt_exist
from rasterio.warp import calculate_default_transform, reproject, Resampling
import albumentations as A


class Augmentation:

    def __init__(self, imagesPath, masksPath, size):


        self.imagePath = imagesPath
        self.maskPath = masksPath
        self.imageNames = self.get_tif_images_names_from_a_folder()
        self.size = size
    def get_tif_images_names_from_a_folder(self):
        imagesPath = glob(f"{self.imagePath}/*.tif")
        imageNames = [x.split("/")[-1] for x in imagesPath]
        return imageNames

    @staticmethod
    def save_raster(data, meta, savePath, saveName):
        with rio.open(f"{savePath}/{saveName}.tif", 'w', **meta) as dst:
            dst.write(data)

    def augment_images(self):
        for image in self.imageNames:
            # open image
            img = rio.open(f"{self.imagePath}/{image}").read()
            img_meta = rio.open(f"{self.imagePath}/{image}").meta
            img = reshape_as_image(img)
            mask = rio.open(f"{self.maskPath}/{image}").read()
            mask_meta = rio.open(f"{self.maskPath}/{image}").meta
            mask = reshape_as_image(mask)

            transform = A.Compose([
                A.HorizontalFlip(p=1),
            ])

            transformed = transform(image=img, mask=mask)
            self.save_raster(reshape_as_raster(transformed["image"]), img_meta, self.imagePath, saveName=f"{image}_hflip")
            self.save_raster(reshape_as_raster(transformed["mask"]), mask_meta, self.maskPath, saveName=f"{image}_hflip")

            transform = A.Compose([
                A.VerticalFlip(p=1),
            ])

            transformed = transform(image=img, mask=mask)
            self.save_raster(reshape_as_raster(transformed["image"]), img_meta, self.imagePath,
                             saveName=f"{image}_Vflip")
            self.save_raster(reshape_as_raster(transformed["mask"]), mask_meta, self.maskPath,
                             saveName=f"{image}_Vflip")

            # transform = A.Compose([
            #     A.RandomBrightnessContrast(p=1),
            # ])
            #
            # transformed = transform(image=img, mask=mask)
            # self.save_raster(reshape_as_raster(transformed["image"]), img_meta, self.imagePath,
            #                  saveName=f"{image}_rb")
            # self.save_raster(reshape_as_raster(transformed["mask"]), mask_meta, self.maskPath,
            #                  saveName=f"{image}_rb")

            transform = A.Compose([A.RandomSizedCrop(min_max_height=(15, 30), height=self.size, width=self.size, p=1)])

            transformed = transform(image=img, mask=mask)

            self.save_raster(reshape_as_raster(transformed["image"]), img_meta, self.imagePath,
                             saveName=f"{image}_crop")
            self.save_raster(reshape_as_raster(transformed["mask"]), mask_meta, self.maskPath,
                             saveName=f"{image}_crop")



            transform = A.Compose([
                A.ChannelShuffle(p=1),
            ])

            transformed = transform(image=img, mask=mask)
            self.save_raster(reshape_as_raster(transformed["image"]), img_meta, self.imagePath,
                             saveName=f"{image}_cs")
            self.save_raster(reshape_as_raster(transformed["mask"]), mask_meta, self.maskPath,
                             saveName=f"{image}_cs")


class Shapefile:
    def __init__(self, shapefilePath):
        self.shapefilePath = shapefilePath

    def open_shapefile(self):
        return gpd.read_file(self.shapefilePath)

    def reproject_Shapefile(self, epsg):
        shapefile = self.open_shapefile()
        shapefile.crs = epsg
        return shapefile

    def buffer(self, distance):
        shapefile = self.open_shapefile()
        buffer = shapefile.buffer(distance=distance)
        return buffer




class Raster:
    def __init__(self, rasterPath):
        self.rasterPath = rasterPath

    def open_raster(self):
        with rio.open(self.rasterPath) as image:
            return image

    @staticmethod
    def save_raster(data, meta, savePath, saveName):
        create_new_folder_if_it_doesnt_exist(savePath)
        with rio.open(f"{savePath}/{saveName}.tif", 'w', **meta) as dst:
            dst.write(np.expand_dims(data, axis=0))

    def calculate_ndvi(self, savePath, saveName):
        with rio.open(self.rasterPath) as image:
            imageArray = image.read()
            meta = image.meta
            meta.update(count=6)
            meta.update(dtype="float64")
            tImage = imageArray.transpose((1, 2, 0))
            ndvi = (tImage[:, :, 4].astype(float) - tImage[:, :, 2].astype(float)) / (tImage[:, :, 4] + tImage[:, :, 2])
            print(imageArray.shape, ndvi.shape)
            print(ndvi.max)
            ndvi = ndvi * (2**16)
            print(ndvi.max())
            ndvi = np.expand_dims(ndvi, axis=0)
            stackImage = np.vstack([imageArray, ndvi])
        create_new_folder_if_it_doesnt_exist(savePath)
        with rio.open(f"{savePath}/{saveName}.tif", 'w', **meta) as dst:
            dst.write(stackImage)

    def get_rgb(self, savePath):
        with rio.open(self.rasterPath) as image:
            imageArray = image.read()
            meta = image.meta
            meta.update(count=3)
            create_new_folder_if_it_doesnt_exist(savePath)

        with rio.open(f"{savePath}/tile_0.tif", "w", **meta) as dst:
            dst.write(imageArray[0:3, :, :])

    def stack_new_band(self, newBandPath, savePath, name):
        with rio.open(self.rasterPath) as image:
            imageArray = image.read()
            print(imageArray.shape)
            meta = image.meta
            meta.update(count=meta["count"] + 1)
            meta.update(dtype="float64")
        with rio.open(newBandPath) as newBand:
            terrain = (newBand.read())
            print(terrain.shape)
            terrain = terrain / terrain.max()
            terrain = np.where(terrain == terrain.min(), 0, terrain)
            terrain = terrain * 2 ** 16

        stackImage = np.vstack([imageArray, terrain])

        # Read each layer and write it to stack
        with rio.open(f"{savePath}/{name}_terrain.tif", 'w', **meta) as dst:
            dst.write(stackImage)





    def reproject_raster(self, epsg):
        saveName = self.rasterPath.split("/")[-1].split(".")[0]
        with rio.open(self.rasterPath) as src:
            transform, width, height = calculate_default_transform(src.crs, epsg, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': epsg,
                'transform': transform,
                'width': width,
                'height': height})
            with rio.open(f'{saveName}_epsg_{epsg.split(":")[-1]}.tif', 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                            source=rio.band(src, i),
                            destination=rio.band(dst, i),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=epsg,
                            resampling=Resampling.nearest)
        print("image reprojected!")


    def get_meta_crs_and_bounds(self):
        image = self.open_raster()
        meta = image.meta
        crs = image.crs
        xmax = image.bounds[2]
        xmin = image.bounds[0]
        ymax = image.bounds[3]
        ymin = image.bounds[1]

        return meta, crs, xmax, xmin, ymax, ymin

    def get_transform(self):
        image = self.open_raster()
        transform = image.transform
        return transform

    def get_metadata(self):
        image = self.open_raster()
        metadata = image.meta
        return metadata

    #TODO save raster
    def save_raster(self):
        pass


class Sampling:
    def __init__(self, raster: Raster, xSize, ySize, savePath):

        self.savePath = savePath
        self.ySize = ySize
        self.xSize = xSize
        self.meta, self.crs, self.xmax, self.xmin, self.ymax, self.ymin = raster.get_meta_crs_and_bounds()
        self.xPixelSize = self.meta["transform"][0]
        self.yPixelSize = self.meta["transform"][4]
        self.width = self.xPixelSize * self.xSize
        self.height = self.yPixelSize * self.ySize * -1



    def generate_regular_sampling_polygons(self, overlap=0, save=False):

        """
        # Code adapted from Keras-spatial library - https://pypi.org/project/keras-spatial/
        Generate regular grid over extent.
        Args:
          overlap (float): percentage of patch overlap (optional)
          save (bool) : True if want to save geojson in a folder.
        Returns:
                geopandas.GeoDataFrame
        """

        # Define the pixel size and multiply by the xsize to get xsize in number of pixels

        numx = int((self.xmax - self.xmin) // (self.width - self.width * overlap))
        numy = int((self.ymax - self.ymin) // (self.height - self.height * overlap))

        x = np.linspace(self.xmin, self.xmax - self.width, num=numx)
        y = np.linspace(self.ymin, self.ymax - self.height, num=numy)
        X, Y = np.meshgrid(x, y)
        polys = [box(x, y, x + self.width, y + self.height) for x, y in np.nditer([X, Y])]

        gdf = gpd.GeoDataFrame({'geometry': polys})
        gdf.crs = self.crs
        if save:
            if not os.path.exists(self.savePath):
                os.makedirs(self.savePath)
            gdf.to_file(f"{self.savePath}/regular_grid_size_{self.xSize}.geojson", driver="GeoJSON")
        else:
            return gdf

    def generate_random_sampling_polygons(self, numberOfPolygons, save=False):

        """
        Generate random grid over extent.
        Args:
          numberOfPolygons (int): number of patches
          save (bool): if true save the dataframe
        Returns:
          :obj:`geopandas.GeoDataFrame`:
        """

        x = np.random.rand(numberOfPolygons) * (self.xmax - self.xmin - self.width) + self.xmin
        y = np.random.rand(numberOfPolygons) * (self.ymax - self.ymin - self.height) + self.ymin
        polys = [box(x, y, x + self.width, y + self.height) for x, y in np.nditer([x, y])]

        gdf = gpd.GeoDataFrame({'geometry': polys})
        gdf.crs = self.crs
        gdf = self.remove_duplicated_polygons(gdf)
        if save:
            create_new_folder_if_it_doesnt_exist(savePath=self.savePath)
            gdf.to_file(f"{self.savePath}/random_grid_size_{self.xSize}.geojson", driver="GeoJSON")
        else:
            return gdf


    def generate_tiles_from_points(self, shapefile, save=False, size=None):

        """Function that converts the points to a specified size square.

        Parameters

        imagePath (string) = Path to the .tif image.

        shapefilePath (string) = path to the shapefile or geojson.

        outputPath (string) = path to output resulting geojson.

        outputName (string) = path to output resulting geojson.

        size (int) = size (in pixels) of the output.

        :return if save == False, return GeoDataFrame

        """

        # Get the pixel size
        print(f"Pixel size - X : {self.xPixelSize}, Y: {-self.yPixelSize}")
        pixelSizeX = self.xPixelSize
        # load the shapefile
        if type(shapefile) == Shapefile:
            df = shapefile.open_shapefile()
        else:
            df = shapefile
        if size:
            bufferSize = (size / 2 - 0.5) * pixelSizeX
        else:
            bufferSize = (self.xSize / 2 - 0.5) * pixelSizeX
        buffer = df.buffer(bufferSize, cap_style=3)
        if save:
            create_new_folder_if_it_doesnt_exist(self.savePath)
            buffer.to_file(f"{self.savePath}/{self.xSize}_tiles_created_from_points.geojson", driver="GeoJSON")
            print("Data Saved!")
        else:
            return buffer

    def select_only_the_tiles_that_intersect_polygons(self, samplingPolygons: Shapefile, featurePolygons: Shapefile, save=False):
        saveName = samplingPolygons.shapefilePath.split("/")[-1].split(".")[0]
        samplingPolygons = samplingPolygons.open_shapefile()
        featurePolygon = featurePolygons.open_shapefile()
        featurePolygon = self.remove_duplicated_polygons(featurePolygon)
        s_join = gpd.sjoin(samplingPolygons, featurePolygon, how="inner")
        s_join = self.remove_duplicated_polygons(s_join)

        if save:
            create_new_folder_if_it_doesnt_exist(self.savePath)
            s_join.to_file(f"{self.savePath}/s_join_{saveName}.geojson", driver="GeoJSON")
        else:
            return s_join

    @staticmethod
    def remove_duplicated_polygons(geopandasDataFrame):
        try:
            if "index_right" and "id" in geopandasDataFrame.keys():
                geopandasDataFrame = geopandasDataFrame.drop(["index_right", "id"], axis=1)
            else:
                geopandasDataFrame = geopandasDataFrame.drop(["index_left", "index_right", "id"], axis=1)
        except:
            geopandasDataFrame = geopandasDataFrame
        return geopandasDataFrame.drop_duplicates()

    def generate_random_points(self):
        xLongitude = np.random.uniform(self.xmin, self.xmax, 1)
        yLatitude = np.random.uniform(self.ymin, self.ymax, 1)
        return xLongitude, yLatitude

    @staticmethod
    def geodataframe_is_empty(geoDataFrame):
        return len(geoDataFrame) == 0

    def create_geodataframe_from_x_y_and_add_crs(self,x,y):
        gdf = gpd.GeoSeries(gpd.points_from_xy(x, y))
        gdf.crs = self.crs
        return gdf

    def buffer_point(self,geoSeries):
        gdfBuffer = geoSeries.buffer(self.width)
        return gdfBuffer

    @staticmethod
    def points_are_inside_buffer(geoSeries, buffer):
        return len(geoSeries[geoSeries.within(buffer)]) > 0

    def generate_random_tiles_without_overlap(self, numberOfPoints, save=False):
        points = gpd.GeoSeries()
        while len(points) != numberOfPoints:
            x, y = self.generate_random_points()
            if self.geodataframe_is_empty(points):
                points = self.create_geodataframe_from_x_y_and_add_crs(x,y)
            else:
                newPoint = self.create_geodataframe_from_x_y_and_add_crs(x,y)
                pointBuffer = self.buffer_point(newPoint)
                if self.points_are_inside_buffer(points, pointBuffer):
                    pass
                else:
                    points = pd.concat([points, newPoint])

        bufferSize = (self.xSize / 2 - 0.5) * self.xPixelSize
        buffer = points.buffer(bufferSize, cap_style=3)
        if save:
            create_new_folder_if_it_doesnt_exist(self.savePath)
            buffer.to_file(f"{self.savePath}/{self.xSize}_tiles_from_random_points_without_overlap.geojson",
                           driver="GeoJSON")
            print("Data Saved!")

    @staticmethod
    def __numberOfPointIsInsidePolygon(geoSeries, polygon):
        numberofPointsInsidePolygon = len(geoSeries[geoSeries.within(polygon)])
        return numberofPointsInsidePolygon

    @staticmethod
    def __generate_n_points_within_bounds(numberOfPoints, xMax, xMin, yMax, yMin):
        numberOfPoints = numberOfPoints
        xLongitute = np.random.uniform(xMin, xMax, numberOfPoints)
        yLatitude = np.random.uniform(yMin, yMax, numberOfPoints)
        return xLongitute, yLatitude

    def generate_points_inside_a_polygon(self,shapefilePath, numberOfPoints):
        """
        Function to generate n number of points inside a polygon.

        :param numberOfPoints: (int) - Number of points that will be created inside the polygon.
        :return: (geopandas.GeoSeries) - Geoseries with the requested number of points inside each polygon.
        """
        # Empty geoseries to concatenate the results
        classification = []
        samplingPointsInsidePolygon = gpd.GeoSeries()
        # open the data
        try:
            shapefiles = gpd.read_file(shapefilePath)
        except:
            shapefiles = shapefilePath
        # loop over each polygon and generate n points inside
        for i, polygon in enumerate(shapefiles["geometry"]):
            classification.append(shapefiles["classification"][i])
            # get the bounds of the polygon
            xMin, yMin, xMax, yMax = polygon.bounds
            # generate numberOfPoints inside a polygon
            xLongitude, yLatitude = self.__generate_n_points_within_bounds(numberOfPoints, xMax, xMin, yMax, yMin)
            # Create a geoseries from x,y
            gdf_points = gpd.GeoSeries(gpd.points_from_xy(xLongitude, yLatitude))
            # Evaluate if number of points inside a Polygon is greater than the desired number of Points (may improve)
            while self.__numberOfPointIsInsidePolygon(gdf_points, polygon) < numberOfPoints:
                # Generate new points
                xLongitude, yLatitude = self.__generate_n_points_within_bounds(numberOfPoints, xMax, xMin, yMax, yMin)
                # geoseries from x,y
                gdf_points = gpd.GeoSeries(gpd.points_from_xy(xLongitude, yLatitude))
            # subset the desired number of points
            gdf_points = gdf_points[gdf_points.within(polygon)][0:numberOfPoints + 1]
            # add the cordnate reference system
            gdf_points.crs = shapefiles.crs
            # concatenate the results)
            samplingPointsInsidePolygon = pd.concat([samplingPointsInsidePolygon, gdf_points])
        samplingPointsInsidePolygon = gpd.GeoDataFrame(geometry=samplingPointsInsidePolygon)
        samplingPointsInsidePolygon["classification"] = classification

        return samplingPointsInsidePolygon

    def generate_tiles_from_classified_points(self, shapefilePath, numberOfPoints=None, centroid=False, sizes=[32, 64, 128]):
        gdf = self.__classify_polygons_area(shapefilePath)
        gdf = gdf.reset_index()
        if centroid:
            samplingPoints = gpd.GeoDataFrame(geometry=gdf.centroid)
            samplingPoints["classification"] = gdf["classification"]

        else:
            samplingPoints = self.generate_points_inside_a_polygon(gdf, numberOfPoints)
        samplingPoints32 = samplingPoints[samplingPoints["classification"] == 0]
        samplingPoints64 = samplingPoints[samplingPoints["classification"] == 1]
        samplingPoints128 = samplingPoints[samplingPoints["classification"] == 2]
        tiles32 = self.generate_tiles_from_points(samplingPoints32, size=sizes[0])
        tiles64 = self.generate_tiles_from_points(samplingPoints64, size=sizes[1])
        tiles128 = self.generate_tiles_from_points(samplingPoints128, size=sizes[2])

        return tiles32, tiles64, tiles128

    @ staticmethod
    def __classify_polygons_area(shapefilePath):
        def classify_areas(row):
            if row["area"] < 2700:
                return 0
            elif (row["area"] > 2700) & (row["area"] < 6000):
                return 1
            else:
                return 2

        gdf = gpd.read_file(shapefilePath)
        gdf["area"] = gdf.area
        gdf = gdf[gdf["area"] != 0]
        gdf["classification"] = gdf.apply(lambda row: classify_areas(row), axis=1)
        return gdf



class BinaryMasks:
    def __init__(self, imageRaster: Raster, shapefile: Shapefile, savePath):
        self.savePath = savePath
        self.shapefile = shapefile
        self.imageRaster = imageRaster

    def generate_binary_mask(self, save=False):
        # load raster
        image = self.imageRaster.open_raster()
        meta = self.imageRaster.get_metadata()
        saveName = self.imageRaster.rasterPath.split("/")[-1].split(".")[0]
        # load shapefile
        labelShapefile = self.shapefile.open_shapefile()
        # Verify if the crs are the same

        if self.check_same_crs(imageCrs=meta["crs"], shapeCrs=labelShapefile.crs):
            polygonsToRasterize = self.get_polygons_to_rasterize(meta, labelShapefile)
            outputImageSize = (meta['height'], meta['width'])
            mask = rasterize(shapes=polygonsToRasterize,
                             out_shape=outputImageSize, all_touched=False)
            mask = mask.astype("uint16")

        if save:
            self.save_raster(mask, meta, self.savePath, saveName)
            return saveName

        else:
            return mask, saveName

    @staticmethod
    def check_same_crs(imageCrs, shapeCrs):
        if imageCrs != shapeCrs:
            print(f" Raster CRS : {imageCrs}  Vetor CRS : {shapeCrs}.\n Convert to the same CRS!")
            return False
        else:
            return True

    @staticmethod
    def get_polygons_to_rasterize(meta, shapefiles):
        def poly_from_utm(polygon, transform):
            poly_pts = []

            poly = cascaded_union(polygon)
            for i in np.array(poly.exterior.coords):
                poly_pts.append(~transform * tuple(i)[0:2])

            new_poly = Polygon(poly_pts)
            return new_poly

        poly_shp = []
        for num, row in shapefiles.iterrows():
            if row['geometry'].geom_type == 'Polygon':
                poly = poly_from_utm(row['geometry'], meta['transform'])
                poly_shp.append(poly)
            else:
                for p in row['geometry']:
                    poly = poly_from_utm(p, meta['transform'])
                    poly_shp.append(poly)
        return poly_shp

    @staticmethod
    def save_raster(maskRaster, meta, savePath, saveName):
        # Salvar
        mask = maskRaster.astype("uint16")
        bin_mask_meta = meta.copy()
        bin_mask_meta.update({'count': 1,
                              "dtype": mask.dtype})
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        with rio.open(f"{savePath}/{saveName}_mask.tif", 'w', **bin_mask_meta) as dst:
            dst.write(mask * 255, 1)


class PatchImage:
    def __init__(self, imagePath, savePath,samplingShapefile: Shapefile):
        self.imagePath = imagePath
        self.savePath = savePath
        self.samplingShapefile = samplingShapefile

    def patch_images(self, size, save=False):
        imagesWithCorrectDimension = []
        wrongDimensionImages = []
        # open shapefile
        shapefile = self.samplingShapefile.open_shapefile()["geometry"]

        print(f"There are {len(shapefile)} polygons to patch the image.")

        with rio.open(self.imagePath) as src:
            for i in range(len(shapefile)):
                try:
                    out_image, out_transform = mask(src, [shapefile[i]], crop=True, filled=False)
                    out_meta = src.meta
                    if out_image.shape[1] == size and out_image.shape[2] == size:
                        imagesWithCorrectDimension.append(reshape_as_image(out_image))
                        if save:
                            out_meta.update({"driver": "GTiff", "height": out_image.shape[1], "width": out_image.shape[2],
                                             "transform": out_transform})
                            create_new_folder_if_it_doesnt_exist(self.savePath)
                            with rio.open(f"{self.savePath}/tiles_{str(i)}.tif", "w", **out_meta) as dest:
                                dest.write(out_image)
                    else:
                        wrongDimensionImages.append(i)
                except ValueError:
                    pass
        print(f"Number of shapefiles with wrong patch dimensions = {len(wrongDimensionImages)}")
        print(f"Number of shapefiles with correct patch dimensions = {len(imagesWithCorrectDimension)}")

        if not save:
            return np.array(imagesWithCorrectDimension, dtype="float32")



# Raster("../data/5_bands/test/test_area_1.tif").stack_new_band("../data/5_bands_elevation/test/test_area_1_terrain.tif", "../data/5_bands_elevation/test", "test_area_1")
# Raster("../data/5_bands/test/test_area_2.tif").stack_new_band("../data/5_bands_elevation/test/test_area_2_terrain.tif", "../data/5_bands_elevation/test", "test_area_2")
# Raster("../data/5_bands/test/test_area_3.tif").stack_new_band("../data/5_bands_elevation/test/test_area_3_terrain.tif", "../data/5_bands_elevation/test", "test_area_3")

# Raster("../data/5_bands/train/tile_0.tif").calculate_ndvi("../data/5_bands_ndvi/train","tile_0")
# Raster("../data/5_bands/test/test_area_1.tif").calculate_ndvi("../data/5_bands_ndvi/test","test_area_1")
# Raster("../data/5_bands/test/test_area_2.tif").calculate_ndvi("../data/5_bands_ndvi/test","test_area_2")
# Raster("../data/5_bands/test/test_area_3.tif").calculate_ndvi("../data/5_bands_ndvi/test","test_area_3")