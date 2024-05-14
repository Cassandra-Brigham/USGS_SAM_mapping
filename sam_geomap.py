import numpy as np
import os
import leafmap
from samgeo.hq_sam import SamGeo, tms_to_geotiff
from samgeo import get_basemaps
import rasterio
from rasterio import features
import geopandas as gpd
from osgeo import gdal, osr, gdalconst
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, confusion_matrix
from shapely.geometry import box
import pandas as pd
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from collections import namedtuple


class FileManager:
    def __init__(self, folder, location, dem_name, planet_data):
        self.folder = folder
        self.location = location
        self.dem_name = dem_name
        self.planet_data = planet_data
        
        # Directories
        self.input_data = folder+'Input_data/'
        self.ML_ready = folder+'ML_ready/'
        self.ML_output = folder+'ML_output/'
        self.prompts = folder+'Prompts/'
        self.unit_masks = folder+'Unit_masks/'
        
        #Topographic dataset file names
        self.input_dem = None # original
        self.hillshade = None # hillshade in original CRS, 1 band
        self.roughness = None # roughness in original CRS, 1 band
        self.slope = None # slope in original CRS, 1 band
        
        self.prep_dem = None # Cropped and warped (epsg4326) DEM, 3 band
        self.prep_hillshade = None # Cropped and warped (epsg4326) hillshade, 3 band
        self.prep_roughness = None # Cropped and warped (epsg4326) roughness, 3 band
        self.prep_slope = None # Cropped and warped (epsg4326) slope, 3 band
        
        self.gaussian_dem = None # Gaussian filter on original CRS, 1 band DEM
        self.gaussian_hillshade= None # Gaussian filter on original CRS, 1 band hillshade
        self.gaussian_roughness= None # Gaussian filter on original CRS, 1 band roughness
        self.gaussian_slope= None # Gaussian filter on original CRS, 1 band slope
        
        self.prep_gaussian_dem = None # Gaussian filter on DEM, cropped, warped, 3 band
        self.prep_gaussian_hillshade= None # Gaussian filter on hillshade, cropped, warped, 3 band
        self.prep_gaussian_roughness= None # Gaussian filter on roughness, cropped, warped, 3 band
        self.prep_gaussian_slope= None # Gaussian filter on slope, cropped, warped, 3 band
        
        self.prep_fuzzy_gaussian_dem = None # Fuzzy Gaussian membership function on DEM, cropped, warped, 3 band
        self.prep_fuzzy_gaussian_hillshade= None # Fuzzy Gaussian membership function on hillshade, cropped, warped, 3 band
        self.prep_fuzzy_gaussian_roughness= None # Fuzzy Gaussian membership function on roughness, cropped, warped, 3 band
        self.prep_fuzzy_gaussian_slope= None # Fuzzy Gaussian membership function on slope, cropped, warped, 3 band
        
        #Planet dataset file names
        self.input_planet = None # input Planet image, original CRS, 4 band
        
        self.rgb_3band_orig = None
        self.ave_3band_orig = None
        self.ndvi_3band_orig = None
        self.ndwi_3band_orig = None
        
        self.prep_planet = None # cropped and warped (epsg4326) Planet image, 4 band
        
        self.rgb_3band = None # RGB image, cropped and warped (epsg4326), 3 band
        self.ave_3band = None # AVE image, cropped and warped (epsg4326), 3 band
        self.ndvi_3band = None # NDVI image, cropped and warped (epsg4326), 3 band
        self.ndwi_3band = None # NDWI image, cropped and warped (epsg4326), 3 band
                
        self.rgb_3band_gaussian = None # Gaussian filter on RGB image, cropped and warped (epsg4326), 3 band
        self.ave_3band_gaussian = None # Gaussian filter on AVE image, cropped and warped (epsg4326), 3 band
        self.ndvi_3band_gaussian = None # Gaussian filter on NDVI image, cropped and warped (epsg4326), 3 band
        self.ndwi_3band_gaussian = None # Gaussian filter on NDWI image, cropped and warped (epsg4326), 3 band
    
    def create_directories (self):

        def create_writable_directory(directory_path):
            try:
                # Create a new directory with write permissions (0o777 gives full permissions)
                os.makedirs(directory_path, mode=0o777)
                print("Directory created successfully.")
                return True
            except OSError as e:
                print(f"Failed to create directory: {e}")
                return False

        create_writable_directory(self.input_data)
        create_writable_directory(self.ML_ready)
        create_writable_directory(self.ML_output)
        create_writable_directory(self.prompts)
        create_writable_directory(self.unit_masks)
    
    def name_files(self):
        #Topo
        self.input_dem = self.folder+self.dem_name
        self.hillshade = self.input_data+'hillshade_'+self.location+".tif"
        self.roughness = self.input_data+'roughness_'+self.location+".tif"
        self.slope = self.input_data+'slope_'+self.location+".tif"
        
        self.prep_dem = self.ML_ready+self.location+'_EPSG_4326_DTM.tif'
        self.prep_hillshade = self.ML_ready+self.location+'_EPSG_4326_hillshade.tif'
        self.prep_roughness = self.ML_ready+self.location+'_EPSG_4326_roughness.tif'
        self.prep_slope = self.ML_ready+self.location+'_EPSG_4326_slope.tif'
        
        self.gaussian_dem = self.input_data+'gaussian_dem_'+self.location+".tif"
        self.gaussian_hillshade = self.input_data+'gaussian_hillshade_'+self.location+".tif"
        self.gaussian_roughness = self.input_data+'gaussian_roughness_'+self.location+".tif"
        self.gaussian_slope = self.input_data+'gaussian_slope_'+self.location+".tif"
        
        self.prep_gaussian_dem = self.ML_ready+self.location+'_EPSG_4326_DTM_Gaussian.tif'
        self.prep_gaussian_hillshade = self.ML_ready+self.location+'_EPSG_4326_hillshade_Gaussian.tif'
        self.prep_gaussian_roughness = self.ML_ready+self.location+'_EPSG_4326_roughness_Gaussian.tif'
        self.prep_gaussian_slope = self.ML_ready+self.location+'_EPSG_4326_slope_Gaussian.tif'
        
        #Planet
        self.input_planet = self.folder+self.planet_data
        
        self.prep_planet= self.input_data+self.location+'_Planet_crop_EPSG_4326.tif'
        
        self.rgb_3band_orig = self.input_data+self.location+'_Planet_rgb.tif'
        self.ave_3band_orig = self.input_data+self.location+'_Planet_ave.tif'
        self.ndvi_3band_orig = self.input_data+self.location+'_Planet_ndvi.tif'
        self.ndwi_3band_orig = self.input_data+self.location+'_Planet_ndwi.tif'
                
        self.rgb_3band = self.ML_ready+self.location+'_EPSG_4326_Planet_rgb.tif'
        self.ave_3band = self.ML_ready+self.location+'_EPSG_4326_Planet_ave.tif'
        self.ndvi_3band = self.ML_ready+self.location+'_EPSG_4326_Planet_ndvi.tif'
        self.ndwi_3band = self.ML_ready+self.location+'_EPSG_4326_Planet_ndwi.tif'
        
        self.rgb_3band_gaussian = self.ML_ready+self.location+'_EPSG_4326_Planet_rgb_Gaussian.tif'
        self.ave_3band_gaussian = self.ML_ready+self.location+'_EPSG_4326_Planet_ave_Gaussian.tif'
        self.ndvi_3band_gaussian = self.ML_ready+self.location+'_EPSG_4326_Planet_ndvi_Gaussian.tif'
        self.ndwi_3band_gaussian = self.ML_ready+self.location+'_EPSG_4326_Planet_ndwi_Gaussian.tif'

class PromptManager:
    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.single_foreground_prompts = None
        self.multiple_foreground_prompts = None
        self.multiple_background_prompts = None
        self.geologic_units = None
        self.coords_single = None
        self.coords_multiple = None
        self.labels_single = None
        self.labels_multiple = None
        
    def prompts_files (self):
        single_foreground_prompts = [f for f in os.listdir(self.file_manager.prompts) if os.path.isfile(os.path.join(self.file_manager.prompts, f)) and f.endswith("_single_foreground.shp")]
        multiple_foreground_prompts = [f for f in os.listdir(self.file_manager.prompts) if os.path.isfile(os.path.join(self.file_manager.prompts, f)) and f.endswith("_multiple_foreground.shp")]
        multiple_background_prompts = [f for f in os.listdir(self.file_manager.prompts) if os.path.isfile(os.path.join(self.file_manager.prompts, f)) and f.endswith("_multiple_background.shp")]
        
        self.geologic_units = [multiple_foreground_prompts[a][:-len("_multiple_foreground.shp")] for a in range(0,len(multiple_foreground_prompts))]
        
        self.single_foreground_prompts = [self.file_manager.prompts+a for a in single_foreground_prompts]
        self.multiple_foreground_prompts = [self.file_manager.prompts+a for a in multiple_foreground_prompts]
        self.multiple_background_prompts = [self.file_manager.prompts+a for a in multiple_background_prompts]

    def shp_to_coords(self, mode='other foreground'):
    
        def coord_list_from_shp_points(foreground_prompt_path, background_prompt_paths):
            # Function to read foreground shapefile and extract coordinates
            foreground_gdf = gpd.read_file(foreground_prompt_path)
            coordinates_list_foreground = []
            for geometry in foreground_gdf.geometry:
                if geometry is not None:
                    if geometry.type == 'MultiPoint':
                        for point in geometry.geoms:
                            coordinates_list_foreground.append([point.x, point.y])
                    elif geometry.type == 'Point':
                        coordinates_list_foreground.append([geometry.x, geometry.y])
            labels_list_foreground = [1] * len(coordinates_list_foreground)
            
            # Initialize lists for background coordinates and labels
            coordinates_list_background = []
            labels_list_background = []
            
            # Process each background shapefile
            for background_path in background_prompt_paths:
                background_gdf = gpd.read_file(background_path)
                for geometry in background_gdf.geometry:
                    if geometry is not None:
                        if geometry.type == 'MultiPoint':
                            for point in geometry.geoms:
                                coordinates_list_background.append([point.x, point.y])
                        elif geometry.type == 'Point':
                            coordinates_list_background.append([geometry.x, geometry.y])
            labels_list_background = [0] * len(coordinates_list_background)

            # Combine foreground and background lists
            coordinates_list = coordinates_list_foreground + coordinates_list_background
            labels_list = labels_list_foreground + labels_list_background

            return coordinates_list, labels_list

        single_foreground_prompts = [self.single_foreground_prompts[a] for a in range (0,len(self.single_foreground_prompts))]
        multiple_foreground_prompts = [self.multiple_foreground_prompts[a] for a in range (0,len(self.multiple_foreground_prompts))]
        multiple_background_prompts = [self.multiple_background_prompts[a] for a in range (0,len(self.multiple_background_prompts))]

        coords_single = []
        labels_single = []
        for a in range(0,len(single_foreground_prompts)):
            coords_single_temp, labels_single_temp = coord_list_from_shp_points(single_foreground_prompts[a], None)
            coords_single.append(coords_single_temp)
            labels_single.append(labels_single_temp)
        
        self.coords_single = coords_single 
        self.labels_single = labels_single

        if mode==None:
            coords_multiple = []
            labels_multiple = []
            for a in range(0,len(multiple_foreground_prompts)):
                coords_multiple_temp, labels_multiple_temp = coord_list_from_shp_points(multiple_foreground_prompts[a], None)
                coords_multiple.append(coords_multiple_temp)
                labels_multiple.append(labels_multiple_temp)

        elif mode == 'files':
            coords_multiple = []
            labels_multiple = []
            for a in range(0,len(multiple_foreground_prompts)):
                coords_multiple_temp, labels_multiple_temp = coord_list_from_shp_points(multiple_foreground_prompts[a], multiple_background_prompts[a])
                coords_multiple.append(coords_multiple_temp)
                labels_multiple.append(labels_multiple_temp)

        elif mode == 'other foreground':
            coords_multiple = []
            labels_multiple = []

            for i, foreground_file in enumerate(multiple_foreground_prompts):
                # Prepare a list of background files excluding the current foreground file
                background_files = [file for j, file in enumerate(multiple_foreground_prompts) if i != j]
                
                # Extract coordinates and labels for the current setup
                coords_temp, labels_temp = coord_list_from_shp_points(foreground_file, background_files)
                coords_multiple.append(coords_temp)
                labels_multiple.append(labels_temp)
            
        self.coords_multiple = coords_multiple
        self.labels_multiple = labels_multiple

class RasterManager:
    def __init__(self, file_manager, prompt_manager):
        self.file_manager = file_manager
        self.prompt_manager = prompt_manager
        self.bounds=None
    
    def calculate_topo_derivs(self,azimuth=315, altitude=45):
        dataset = gdal.Open(self.file_manager.input_dem, gdal.GA_Update)  # Open the dataset in update mode
        if dataset:
            band = dataset.GetRasterBand(1)  # Assuming a single-band dataset
            band.SetNoDataValue(-9999)  # Set the no-data value as needed
            band.FlushCache()  # Ensure changes are written
            dataset = None  # Close the dataset
        
        options1 = gdal.DEMProcessingOptions( azimuth=azimuth, altitude=altitude)
        gdal.DEMProcessing(self.file_manager.hillshade, self.file_manager.input_dem, "hillshade", options=options1)
        
    
        gdal.DEMProcessing(self.file_manager.roughness, self.file_manager.input_dem, "roughness")
        
        
        gdal.DEMProcessing(self.file_manager.slope, self.file_manager.input_dem, "slope")
   
    @staticmethod
    def warp_raster(input_file, output_file):
        input_ds = gdal.Open(input_file)
        input_proj = input_ds.GetProjection()
        source_srs = osr.SpatialReference(input_proj)

        target_srs = "EPSG:4326"
        resampling_method = "bilinear"
        output_format = "GTiff"

        warp_options = gdal.WarpOptions(
        format=output_format,
        srcSRS=source_srs,
        dstSRS=target_srs,
        resampleAlg=resampling_method)

        gdal.Warp(output_file, input_file, options=warp_options)
    
    def get_bounds(self, crop_distance):
        def native_utm_crs_from_aoi_bounds(bounds,datum):
            """
            Get the native UTM coordinate reference system from the 

            :param bounds: shapely Polygon of bounding box in EPSG:4326 CRS
            :param datum: string with datum name (e.g., "WGS84")
            :return: UTM CRS code
            """
            utm_crs_list = query_utm_crs_info(
                datum_name=datum,
                area_of_interest=AreaOfInterest(
                    west_lon_degree=bounds[0],
                    south_lat_degree=bounds[1],
                    east_lon_degree=bounds[2],
                    north_lat_degree=bounds[3],
                ),
            )
            utm_crs = CRS.from_epsg(utm_crs_list[0].code)
            return utm_crs
        
        with rasterio.open(self.file_manager.input_dem) as src:
            # Get bounds of the raster
            bounds = src.bounds
            crs = src.crs
            
        geometry = box(*bounds)
        gdf = gpd.GeoDataFrame(pd.Series(geometry,name='geometry'),geometry='geometry',crs=crs)
        
        gdf_4326 = gdf.to_crs(epsg=4326)
        bounds_4326 = gdf_4326.bounds
        
        crs_utm = native_utm_crs_from_aoi_bounds(list(bounds_4326.iloc[0,:]),"WGS84")
        gdf_utm = gdf.to_crs(crs_utm)
        bounds_utm = gdf_utm.bounds
        
        crop_array = pd.Series([crop_distance,crop_distance,-crop_distance,-crop_distance],index=['minx','miny','maxx','maxy'])
        cropped_bounds_utm = bounds_utm+crop_array
        bounding_box = box(list(cropped_bounds_utm.iloc[0,:])[0], list(cropped_bounds_utm.iloc[0,:])[1], list(cropped_bounds_utm.iloc[0,:])[2], list(cropped_bounds_utm.iloc[0,:])[3])
        gdf_cropped_utm = gpd.GeoDataFrame(pd.Series(bounding_box,name='geometry'),geometry='geometry',crs=crs_utm)
        
        gdf_4326_cropped = gdf_cropped_utm.to_crs(epsg=4326)
        bounds_4326_cropped = gdf_4326_cropped.bounds

        BoundingBox = namedtuple('BoundingBox', ['left', 'bottom', 'right', 'top'])

        bbox = BoundingBox(left=list(bounds_4326_cropped.iloc[0,:])[0], bottom=list(bounds_4326_cropped.iloc[0,:])[1], right=list(bounds_4326_cropped.iloc[0,:])[2], top=list(bounds_4326_cropped.iloc[0,:])[3])
        
        self.bounds = bbox
        
    @staticmethod
    def crop_raster(input_raster, output_raster, bounds):
        gdal.Warp(output_raster, input_raster, outputBounds=bounds)
    
    @staticmethod
    def make_three_band_image(image_path, out_image_path):
        # Open the DEM raster file
        with rasterio.open(image_path) as src:
            # Read the first band
            dtm_data = src.read(1).astype(float)

            # Set no data value and mask
            no_data_value = src.nodatavals[0]
            dtm_data[dtm_data == no_data_value] = np.nan

            # Normalize the data to 0-255, ignoring no data value
            min_val = np.nanmin(dtm_data)
            max_val = np.nanmax(dtm_data)
            dtm_normalized = (dtm_data - min_val) / (max_val - min_val) * 255
            dtm_normalized = np.nan_to_num(dtm_normalized, nan=0.1).astype(np.uint8)

            # Duplicate the normalized data to create 3 bands
            gray_3band = np.stack((dtm_normalized, dtm_normalized, dtm_normalized), axis=0)

            # Create a new 3-band grayscale image
            profile = src.profile
            profile.update(count=3, dtype=rasterio.uint8, nodata=0)

            with rasterio.open(out_image_path, 'w', **profile) as dst:
                dst.write(gray_3band)
 
    def gaussian_topo_data(self, scale):

        def apply_gaussian_filter(input_raster, output_raster, input_scale):
            # Open input raster
            dataset = gdal.Open(input_raster, gdal.GA_ReadOnly)
            if dataset is None:
                print("Error: Could not open input raster.")
                return

            # Get raster dimensions
            rows = dataset.RasterYSize
            cols = dataset.RasterXSize

            # Read raster data
            band = dataset.GetRasterBand(1)
            data = band.ReadAsArray()

            # Apply Gaussian filter 
            sigma = input_scale / dataset.GetGeoTransform()[1]  
            filtered_data = gaussian_filter(data, sigma=sigma)

            # Write filtered data to output raster
            driver = gdal.GetDriverByName("GTiff")
            output_dataset = driver.Create(output_raster, cols, rows, 1, gdal.GDT_Float32)
            output_dataset.GetRasterBand(1).WriteArray(filtered_data)

            # Set projection and transformation
            output_dataset.SetProjection(dataset.GetProjection())
            output_dataset.SetGeoTransform(dataset.GetGeoTransform())

            # Close datasets
            dataset = None
            output_dataset = None
            
        apply_gaussian_filter(self.file_manager.input_dem, self.file_manager.gaussian_dem, scale)
        apply_gaussian_filter(self.file_manager.hillshade, self.file_manager.gaussian_hillshade, scale)
        apply_gaussian_filter(self.file_manager.roughness, self.file_manager.gaussian_roughness, scale)
        apply_gaussian_filter(self.file_manager.slope, self.file_manager.gaussian_slope, scale)
        
    def prep_data(self, input_raster, output_raster):
        self.warp_raster(input_raster, input_raster[:-len(".tif")]+'_warp.tif')
        self.crop_raster(input_raster[:-len(".tif")]+'_warp.tif',input_raster[:-len(".tif")]+'_crop.tif',self.bounds)
        self.make_three_band_image(input_raster[:-len(".tif")]+'_crop.tif', output_raster)
    
    def prep_topo_data(self):
        
        self.prep_data(self.file_manager.input_dem, self.file_manager.prep_dem)
        self.prep_data(self.file_manager.hillshade, self.file_manager.prep_hillshade)
        self.prep_data(self.file_manager.roughness, self.file_manager.prep_roughness)
        self.prep_data(self.file_manager.slope, self.file_manager.prep_slope)
        self.prep_data(self.file_manager.gaussian_dem, self.file_manager.prep_gaussian_dem)
        self.prep_data(self.file_manager.gaussian_hillshade, self.file_manager.prep_gaussian_hillshade)
        self.prep_data(self.file_manager.gaussian_roughness, self.file_manager.prep_gaussian_roughness)
        self.prep_data(self.file_manager.gaussian_slope, self.file_manager.prep_gaussian_slope)
    
    @staticmethod    
    def gaussian_membership(value, mean, std):
        """
        Calculate the Gaussian membership value for a given input.
        """
        return np.exp(-0.5 * ((value - mean) ** 2 / std ** 2))

    def apply_fuzzy_gaussian_membership_to_3band(self,input_raster, output_raster, mean, std):
        
        means = [mean,mean,mean]
        stds = [std,std,std]
        # Open the input raster
        dataset = gdal.Open(input_raster, gdal.GA_ReadOnly)
        if dataset is None:
            print("Error: Could not open input raster.")
            return

        # Get raster dimensions
        rows = dataset.RasterYSize
        cols = dataset.RasterXSize
        num_bands = 3  # 3-band raster

        # Create the output raster
        driver = gdal.GetDriverByName("GTiff")
        output_dataset = driver.Create(output_raster, cols, rows, num_bands, gdal.GDT_Float32)

        # Copy projection and geotransform from input to output
        output_dataset.SetProjection(dataset.GetProjection())
        output_dataset.SetGeoTransform(dataset.GetGeoTransform())

        # Process each band
        for band_index in range(1, num_bands + 1):
            band = dataset.GetRasterBand(band_index)
            data = band.ReadAsArray()

            # Apply the fuzzy Gaussian membership function using the mean and std for the current band
            fuzzy_data = self.gaussian_membership(data, means[band_index-1], stds[band_index-1])

            # Write the result to the corresponding band of the output raster
            output_dataset.GetRasterBand(band_index).WriteArray(fuzzy_data)

        # Close the datasets
        dataset = None
        output_dataset = None
    
    @staticmethod
    def raster_stats_at_points_window(raster_path, points_shapefile_path, window_size=3):
        """
        Calculate the mean and standard deviation of raster values within a window around locations
        defined by a shapefile for all bands in the raster.
        
        Parameters:
        - raster_path: Path to the raster file.
        - points_shapefile_path: Path to the shapefile containing point geometries.
        - window_size: Size of the window to sample around each point (must be an odd integer).
        
        Returns:
        - A list means and standard deviations for the bands in the raster
        """
        # Load the point shapefile
        points = gpd.read_file(points_shapefile_path)
        
        # Open the raster file
        raster = gdal.Open(raster_path)
        gt = raster.GetGeoTransform()
        
        bands_means = []
        bands_stds = []

        # Iterate over each band
        for band in range(1, raster.RasterCount + 1):
            rb = raster.GetRasterBand(band)
            # Loop through each point in the shapefile
            values = []
            for point in points.geometry:
                if point is None:
                    continue
                x, y = point.x, point.y

                # Convert from geographic coordinates to raster pixel coordinates
                px = int((x - gt[0]) / gt[1])
                py = int((y - gt[3]) / gt[5])

                # Calculate the half window size to sample around the point
                half_win = window_size // 2

                # Sample the raster in a window around the given point
                try:
                    # Adjust px, py to ensure the window fits within the raster dimensions
                    win_x_start = max(px - half_win, 0)
                    win_y_start = max(py - half_win, 0)
                    win_x_size = min(window_size, raster.RasterXSize - win_x_start)
                    win_y_size = min(window_size, raster.RasterYSize - win_y_start)

                    val = rb.ReadAsArray(win_x_start, win_y_start, win_x_size, win_y_size).astype(float)
                    
                except ValueError:  # In case the point or window is outside the raster
                    continue

                # Ignore no data values if your raster has them; for example, -9999
                val = val[val != -9999]

                values.append(val)
            
            raster_values_at_points = np.concatenate(values, axis=0)
            raster_values_at_points_mean = np.mean(raster_values_at_points) if raster_values_at_points.size > 0 else 'N/A'
            raster_values_at_points_std = np.std(raster_values_at_points) if raster_values_at_points.size > 0 else 'N/A'

            bands_means.append(raster_values_at_points_mean)
            bands_stds.append(raster_values_at_points_std)

        return bands_means, bands_stds
    
    def fuzzy_gaussian_points(self, input_raster, output_raster, points_shapefile_path, window_size):
        means, stds = self.raster_stats_at_points_window(input_raster, points_shapefile_path, window_size)
        self.apply_fuzzy_gaussian_membership_to_3band(input_raster, output_raster, means[0], stds[0])

    def apply_fuzzy_gaussian(self, window_size=3, mode='prompts',  type = 'hillshade', mean=None, std=None):
        if mode == 'prompts':
            for i,unit in enumerate(self.prompt_manager.multiple_foreground_prompts):
                self.fuzzy_gaussian_points(self.file_manager.prep_dem, self.file_manager.ML_ready+self.file_manager.location+f'_{self.prompt_manager.geologic_units[i]}_EPSG_4326_DTM_fuzzy_Gaussian.tif', unit, window_size)
                self.fuzzy_gaussian_points(self.file_manager.prep_hillshade, self.file_manager.ML_ready+self.file_manager.location+f'_{self.prompt_manager.geologic_units[i]}_EPSG_4326_hillshade_fuzzy_Gaussian.tif', unit, window_size)
                self.fuzzy_gaussian_points(self.file_manager.prep_roughness, self.file_manager.ML_ready+self.file_manager.location+f'_{self.prompt_manager.geologic_units[i]}_EPSG_4326_roughness_fuzzy_Gaussian.tif' , unit, window_size)
                self.fuzzy_gaussian_points(self.file_manager.prep_slope, self.file_manager.ML_ready+self.file_manager.location+f'_{self.prompt_manager.geologic_units[i]}_EPSG_4326_slope_fuzzy_Gaussian.tif', unit, window_size)
        elif mode == 'parameters':
            if type == 'DTM':
                self.apply_fuzzy_gaussian_membership_to_3band(self.file_manager.prep_dem, self.file_manager.ML_ready+self.file_manager.location+f'_means_{mean}_std_{std}_EPSG_4326_DTM_fuzzy_Gaussian.tif', mean, std)
            elif type == 'hillshade':
                self.apply_fuzzy_gaussian_membership_to_3band(self.file_manager.prep_hillshade, self.file_manager.ML_ready+self.file_manager.location+f'_means_{mean}_std_{std}_EPSG_4326_hillshade_fuzzy_Gaussian.tif', mean, std)
            elif type == 'roughness':
                self.apply_fuzzy_gaussian_membership_to_3band(self.file_manager.prep_roughness, self.file_manager.ML_ready+self.file_manager.location+f'_means_{mean}_std_{std}_EPSG_4326_roughness_fuzzy_Gaussian.tif', mean, std)
            elif type == 'slope':
                self.apply_fuzzy_gaussian_membership_to_3band(self.file_manager.prep_slope, self.file_manager.ML_ready+self.file_manager.location+f'_means_{mean}_std_{std}_EPSG_4326_slope_fuzzy_Gaussian.tif', mean, std)

class PlanetManager:
    def __init__(self, file_manager,raster_manager,prompt_manager):
        self.file_manager = file_manager
        self.raster_manager = raster_manager
        self.prompt_manager = prompt_manager
        self.b1 = None
        self.b2 = None
        self.b3 = None
        self.b4 = None
        self.raster_src = None
        self.bounds = None
        self.gray_3band = None
        self.rgb_3band = None
        self.ave_3band= None
        self.ndvi_3band= None
        self.ndwi_3band= None

    
    @staticmethod
    def warp_raster(input_file, output_file):
        input_ds = gdal.Open(input_file)
        input_proj = input_ds.GetProjection()
        source_srs = osr.SpatialReference(input_proj)

        target_srs = "EPSG:4326"
        resampling_method = "near"
        output_format = "GTiff"

        warp_options = gdal.WarpOptions(
        format=output_format,
        srcSRS=source_srs,
        dstSRS=target_srs,
        resampleAlg=resampling_method)

        gdal.Warp(output_file, input_file, options=warp_options)
    
    @staticmethod
    def crop_raster(input_raster, output_raster, bounds):
        gdal.Warp(output_raster, input_raster, outputBounds=bounds)
    
    @staticmethod
    def get_planet_derivatives(input_raster):
        with rasterio.open(input_raster) as src:
            b1 = src.read(3).astype(float)
            b2 = src.read(2).astype(float)
            b3 = src.read(1).astype(float)
            b4 = src.read(4).astype(float)
            
        # Create a new 3-band grayscale image
            profile = src.profile
            profile.update(count=3, dtype=rasterio.float32)
        
        np.seterr(divide='ignore', invalid='ignore')

        ave_band = (b1+b2+b3)/3

        ndvi_band = (b4-b1)/(b4+b1)

        ndwi_band = (b2-b4)/(b2+b4)
        
        return profile, b1,b2,b3,b4,ave_band,ndvi_band,ndwi_band
    
    @staticmethod
    def make_3band(input_data):
        data_3band = np.stack((input_data,input_data,input_data), axis=0)
        return data_3band
    
    def save_3band_image(self,data,output_raster,profile):
        with rasterio.open(output_raster, 'w', **profile) as dst:
            dst.write(data)

    def prep_planet_data(self):
        self.warp_raster(self.file_manager.input_planet, self.file_manager.input_planet[:-len(".tif")]+'_warp.tif')
        self.crop_raster(self.file_manager.input_planet[:-len(".tif")]+'_warp.tif', self.file_manager.prep_planet,self.raster_manager.bounds)
        
        profile,b1,b2,b3,b4,ave_band,ndvi_band,ndwi_band = self.get_planet_derivatives(self.file_manager.prep_planet)
        
        rgb_3band = np.stack((b1,b2,b3), axis=0)
        
        self.save_3band_image(rgb_3band, self.file_manager.rgb_3band, profile)
        self.save_3band_image(self.make_3band(ave_band), self.file_manager.ave_3band, profile)
        self.save_3band_image(self.make_3band(ndvi_band), self.file_manager.ndvi_3band, profile)
        self.save_3band_image(self.make_3band(ndwi_band), self.file_manager.ndwi_3band, profile)
    
    @staticmethod
    def apply_gaussian_to_3band(input_raster, output_raster, input_scale):
        # Open input raster
        dataset = gdal.Open(input_raster, gdal.GA_ReadOnly)
        if dataset is None:
            print("Error: Could not open input raster.")
            return

        # Get raster dimensions
        rows = dataset.RasterYSize
        cols = dataset.RasterXSize
        num_bands = 3  # Assuming the output should have 3 bands (e.g., RGB)

        # Create output raster
        driver = gdal.GetDriverByName("GTiff")
        output_dataset = driver.Create(output_raster, cols, rows, num_bands, gdal.GDT_Float32)

        # Set projection and transformation to match the input
        output_dataset.SetProjection(dataset.GetProjection())
        output_dataset.SetGeoTransform(dataset.GetGeoTransform())

        for band_index in range(1, num_bands + 1):
            # Read data from the current band of the input raster
            band = dataset.GetRasterBand(band_index)
            data = band.ReadAsArray()

            # Apply Gaussian filter
            sigma = input_scale / dataset.GetGeoTransform()[1]  # Convert scale meters to pixels
            filtered_data = gaussian_filter(data, sigma=sigma)

            # Write filtered data to the corresponding band of the output raster
            output_dataset.GetRasterBand(band_index).WriteArray(filtered_data)

        # Close datasets
        dataset = None
        output_dataset = None
    
    def prep_gaussian_planet_data(self, scale):
        profile,b1,b2,b3,b4,ave_band,ndvi_band,ndwi_band = self.get_planet_derivatives(self.file_manager.input_planet)
        
        rgb_3band = np.stack((b1,b2,b3), axis=0)
        
        self.save_3band_image(rgb_3band, self.file_manager.rgb_3band_orig, profile)
        self.save_3band_image(self.make_3band(ave_band), self.file_manager.ave_3band_orig, profile)
        self.save_3band_image(self.make_3band(ndvi_band), self.file_manager.ndvi_3band_orig, profile)
        self.save_3band_image(self.make_3band(ndwi_band), self.file_manager.ndwi_3band_orig, profile)
        
        self.apply_gaussian_to_3band(self.file_manager.rgb_3band_orig,self.file_manager.input_planet[:-len(".tif")]+'_rgb_gaussian_orig.tif',scale)
        self.apply_gaussian_to_3band(self.file_manager.ave_3band_orig,self.file_manager.input_planet[:-len(".tif")]+'_ave_gaussian_orig.tif',scale)
        self.apply_gaussian_to_3band(self.file_manager.ndvi_3band_orig,self.file_manager.input_planet[:-len(".tif")]+'_ndvi_gaussian_orig.tif',scale)
        self.apply_gaussian_to_3band(self.file_manager.ndwi_3band_orig,self.file_manager.input_planet[:-len(".tif")]+'_ndwi_gaussian_orig.tif',scale)
        
        self.warp_raster(self.file_manager.input_planet[:-len(".tif")]+'_rgb_gaussian_orig.tif', self.file_manager.input_planet[:-len(".tif")]+'_rgb_gaussian_orig_warp.tif')
        self.warp_raster(self.file_manager.input_planet[:-len(".tif")]+'_ave_gaussian_orig.tif', self.file_manager.input_planet[:-len(".tif")]+'_ave_gaussian_orig_warp.tif')
        self.warp_raster(self.file_manager.input_planet[:-len(".tif")]+'_ndvi_gaussian_orig.tif', self.file_manager.input_planet[:-len(".tif")]+'_ndvi_gaussian_orig_warp.tif')
        self.warp_raster(self.file_manager.input_planet[:-len(".tif")]+'_ndwi_gaussian_orig.tif', self.file_manager.input_planet[:-len(".tif")]+'_ndwi_gaussian_orig_warp.tif')
        
        self.crop_raster(self.file_manager.input_planet[:-len(".tif")]+'_rgb_gaussian_orig_warp.tif', self.file_manager.rgb_3band_gaussian,self.raster_manager.bounds)
        self.crop_raster(self.file_manager.input_planet[:-len(".tif")]+'_ave_gaussian_orig_warp.tif', self.file_manager.ave_3band_gaussian,self.raster_manager.bounds)
        self.crop_raster(self.file_manager.input_planet[:-len(".tif")]+'_ndvi_gaussian_orig_warp.tif',self.file_manager.ndvi_3band_gaussian,self.raster_manager.bounds)
        self.crop_raster(self.file_manager.input_planet[:-len(".tif")]+'_ndwi_gaussian_orig_warp.tif',self.file_manager.ndwi_3band_gaussian,self.raster_manager.bounds)

    def apply_fuzzy_gaussian_membership_to_3band(self,input_raster, output_raster, means, stds):
        
        # Open the input raster
        dataset = gdal.Open(input_raster, gdal.GA_ReadOnly)
        if dataset is None:
            print("Error: Could not open input raster.")
            return

        # Get raster dimensions
        rows = dataset.RasterYSize
        cols = dataset.RasterXSize
        num_bands = 3  # 3-band raster

        # Create the output raster
        driver = gdal.GetDriverByName("GTiff")
        output_dataset = driver.Create(output_raster, cols, rows, num_bands, gdal.GDT_Float32)

        # Copy projection and geotransform from input to output
        output_dataset.SetProjection(dataset.GetProjection())
        output_dataset.SetGeoTransform(dataset.GetGeoTransform())

        # Process each band
        for band_index in range(1, num_bands + 1):
            band = dataset.GetRasterBand(band_index)
            data = band.ReadAsArray()

            # Apply the fuzzy Gaussian membership function using the mean and std for the current band
            fuzzy_data = self.raster_manager.gaussian_membership(data, means[band_index-1], stds[band_index-1])

            # Write the result to the corresponding band of the output raster
            output_dataset.GetRasterBand(band_index).WriteArray(fuzzy_data)

        # Close the datasets
        dataset = None
        output_dataset = None
    
    def fuzzy_gaussian_points(self, input_raster, output_raster, points_shapefile_path, window_size):
        means, stds = self.raster_manager.raster_stats_at_points_window(input_raster, points_shapefile_path, window_size)
        self.apply_fuzzy_gaussian_membership_to_3band(input_raster, output_raster, means, stds)
    
    def apply_fuzzy_gaussian(self, window_size=3, mode='prompts',  type = 'Planet_rgb', means=None, stds=None):
        if mode == 'prompts':
            for i,unit in enumerate(self.prompt_manager.multiple_foreground_prompts):
                self.fuzzy_gaussian_points(self.file_manager.rgb_3band, self.file_manager.ML_ready+self.file_manager.location+f'_{self.prompt_manager.geologic_units[i]}_EPSG_4326_Planet_rgb_fuzzy_Gaussian.tif', unit, window_size)
                self.fuzzy_gaussian_points(self.file_manager.ave_3band, self.file_manager.ML_ready+self.file_manager.location+f'_{self.prompt_manager.geologic_units[i]}_EPSG_4326_Planet_ave_fuzzy_Gaussian.tif', unit, window_size)
                self.fuzzy_gaussian_points(self.file_manager.ndvi_3band, self.file_manager.ML_ready+self.file_manager.location+f'_{self.prompt_manager.geologic_units[i]}_EPSG_4326_Planet_ndvi_fuzzy_Gaussian.tif' , unit, window_size)
                self.fuzzy_gaussian_points(self.file_manager.ndwi_3band, self.file_manager.ML_ready+self.file_manager.location+f'_{self.prompt_manager.geologic_units[i]}_EPSG_4326_Planet_ndwi_fuzzy_Gaussian.tif', unit, window_size)
        elif mode == 'parameters':
            if type == 'Planet_rgb':
                self.apply_fuzzy_gaussian_membership_to_3band(self.file_manager.rgb_3band, self.file_manager.ML_ready+self.file_manager.location+f'_means_{means[0]}_{means[1]}_{means[2]}_std_{stds[0]}_{stds[1]}_{stds[2]}_EPSG_4326_Planet_rgb_fuzzy_Gaussian.tif', means, stds)
            elif type == 'Planet_ave':
                self.apply_fuzzy_gaussian_membership_to_3band(self.file_manager.ave_3band, self.file_manager.ML_ready+self.file_manager.location+f'_means_{means[0]}_{means[1]}_{means[2]}_std_{stds[0]}_{stds[1]}_{stds[2]}_EPSG_4326_Planet_ave_fuzzy_Gaussian.tif', means, stds)
            elif type == 'Planet_ndvi':
                self.apply_fuzzy_gaussian_membership_to_3band(self.file_manager.ndvi_3band, self.file_manager.ML_ready+self.file_manager.location+f'_means_{means[0]}_{means[1]}_{means[2]}_std_{stds[0]}_{stds[1]}_{stds[2]}_EPSG_4326_Planet_ndvi_fuzzy_Gaussian.tif', means, stds)
            elif type == 'Planet_ndwi':
                self.apply_fuzzy_gaussian_membership_to_3band(self.file_manager.ndwi_3band, self.file_manager.ML_ready+self.file_manager.location+f'_means_{means[0]}_{means[1]}_{means[2]}_std_{stds[0]}_{stds[1]}_{stds[2]}_EPSG_4326_Planet_ndwi_fuzzy_Gaussian.tif', means, stds)
       

class SAMManager:
    def __init__(self, file_manager, prompt_manager):
        self.file_manager = file_manager
        self.prompt_manager = prompt_manager
        self.sam=None
        self.default_kwargs = {
                "points_per_side": 32,
                "points_per_batch": 64,
                "pred_iou_thresh": 0.88,
                "stability_score_thresh": 0.95,
                "stability_score_offset": 1.0,
                "box_nms_thresh": 0.7,
                "crop_n_layers": 0,
                "crop_nms_thresh": 0.7,
                "crop_overlap_ratio": 512 / 1500,
                "crop_n_points_downscale_factor": 1,
                "point_grids": None,
                "min_mask_region_area": 0,
                "output_mode": "binary_mask",
                }
        self.list_image_types = None
        
    def initiate_sam(self, kwargs, auto = False):
        self.sam = SamGeo(
        model_type="vit_h",
        automatic=auto,
        device=None,
        checkpoint_dir=None,
        sam_kwargs=None,
        **kwargs
        )

        self.list_image_types = ['DTM','hillshade','roughness','slope','DTM_Gaussian','Planet_rgb','Planet_ave','Planet_ndvi','Planet_ndwi','Planet_rgb_Gaussian','Planet_ave_Gaussian','Planet_ndvi_Gaussian','Planet_ndwi_Gaussian']

    def sam_predict_single(self):

        def find_filenames_matching_string(file_paths, pattern):
            matching_filenames = []
            for file_path in file_paths:
                filename = os.path.basename(file_path)
                if pattern in filename:
                    matching_filenames.append(filename)
            return matching_filenames
        
        for a in self.list_image_types:
            matching_image = find_filenames_matching_string(self.file_manager.ML_location,a)
            if not matching_image:  # If the current list is empty, skip to the next one
                continue
            else:
                self.sam.set_image(matching_image[0])
                
                for b in range(0,len(self.prompt_manager.geologic_units)):
                    
                    output_path = self.file_manager.folder+self.file_manager.location+'/ML_output/'+self.file_manager.location+'_mask_single_'+a+'_'+self.prompt_manager.geologic_units[b]+'.tif'
                    point_coords = self.prompt_manager.coords_single[b]
                    labels = self.prompt_manager.labels_single[b]
                    
                    self.sam.predict(point_coords, point_labels=labels, point_crs="EPSG:4326", output=output_path)

    def sam_predict_multiple(self):
        def find_file_by_pattern(directory, pattern):
            """
            Search for files in a given directory that contain the pattern in their filename.
            
            Parameters:
            - directory: Path to the directory to search within.
            - pattern: The pattern to look for in the filenames.
            
            Returns:
            - A list of filenames that contain the pattern.
            """
            matching_files = []
            # List all files and directories in the given directory
            for filename in os.listdir(directory):
                # Construct the full path to the item
                full_path = os.path.join(directory, filename)
                # Check if it is a file and if the pattern is in the filename
                if os.path.isfile(full_path) and pattern in filename:
                    # If a matching file is found, append its full path to the list
                    matching_files.append(full_path)
            return matching_files

        for a in self.list_image_types:
            try:
                matching_image = find_file_by_pattern(self.file_manager.ML_location, a)
                if not matching_image:
                    print(f"No matching images found for type {a}.")
                    continue

                self.sam.set_image(matching_image[0])

                for b in range(0, len(self.prompt_manager.geologic_units)):
                    output_path = f"{self.file_manager.folder}{self.file_manager.location}/ML_output/{self.file_manager.location}_mask_multiple_{a}_{self.prompt_manager.geologic_units[b]}.tif"
                    point_coords = self.prompt_manager.coords_multiple[b]
                    labels = self.prompt_manager.labels_multiple[b]

                    if not point_coords or not labels:
                        print(f"Missing coordinates or labels for {a}, unit {self.prompt_manager.geologic_units[b]}. Skipping...")
                        continue

                    self.sam.predict(point_coords, point_labels=labels, point_crs="EPSG:4326", output=output_path)
                    print(f"Prediction completed for {output_path}")

            except Exception as e:
                print(f"An error occurred during multiple prediction for {a}: {e}")

class MaskManager:
    def __init__(self, file_manager,prompt_manager,sam_manager):
        self.file_manager=file_manager
        self.prompt_manager=prompt_manager
        self.sam_manager = sam_manager
        self.unit_files=None
        self.unit_names=None
        self.unit_masks=None
        self.mask_transform = None
        self.mask_crs = None
        self.mask_width = None
        self.mask_height = None
        self.metrics = None
        #self.accuracy = None
        #self.precision = None
        #self.recall = None
        #self.f1 = None
        #self.iou = None
    
    def get_unit_files(self):
        def find_files_by_two_patterns(directory, pattern1, pattern2):
            """
            Search for files in a given directory that contain both of the specified patterns in their filename.
            
            Parameters:
            - directory: Path to the directory to search within.
            - pattern1: The first pattern to look for in the filenames.
            - pattern2: The second pattern to look for in the filenames.
            
            Returns:
            - A list of filenames that contain both of the patterns.
            """
            matching_files = []
            # List all files and directories in the given directory
            for filename in os.listdir(directory):
                # Construct the full path to the item
                full_path = os.path.join(directory, filename)
                # Check if it is a file and if both of the patterns are in the filename
                if os.path.isfile(full_path) and (pattern1 in filename and pattern2 in filename):
                    # If a matching file is found, append its full path to the list
                    matching_files.append(full_path)
            return matching_files
    
        self.unit_files = [find_files_by_two_patterns(self.file_manager.folder+self.file_manager.location+'/Units/',a,'.shp')[0] for a in self.prompt_manager.geologic_units if find_files_by_two_patterns(self.file_manager.folder+self.file_manager.location+'/Units/',a,'.shp')]
        self.unit_names = [os.path.basename(a)[:-len(".shp")] for a in self.unit_files]
        self.unit_masks = [self.file_manager.folder+self.file_manager.location+'/Unit_masks/'+a+'_binary_mask.tif' for a in self.unit_names]
    
    def shapefile_to_mask(self):
        example_mask = self.file_manager.folder+self.file_manager.location+'/ML_output/'+self.file_manager.location+'_mask_multiple_'+self.sam_manager.list_image_types[0]+'_'+self.prompt_manager.geologic_units[0]+'.tif'
        with rasterio.open(example_mask) as mask:
            self.mask_transform = mask.transform
            self.mask_crs = mask.crs
            self.mask_width = mask.width
            self.mask_height = mask.height
                
        for shapefile_in, mask_out in zip(self.unit_files,self.unit_masks):
            polygon_shp = gpd.read_file(shapefile_in)

            #check CRS and change if needed
            if polygon_shp.crs != self.mask_crs:
                polygon_shp = polygon_shp.to_crs(self.mask_crs)

            # Create an empty array
            binary_mask = np.zeros((self.mask_height, self.mask_width), dtype=np.uint8)

            # Get the geometry of the Quat unit polygon in the correct format
            geometry = [geom['geometry'] for geom in polygon_shp.geometry.__geo_interface__['features']]

            # Burn the Quat unit polygon into the array
            features.rasterize(
                shapes=geometry,
                out=binary_mask,
                transform=self.mask_transform,
                fill=0,  # Background
                default_value=1,  # Foreground
                dtype=np.uint8
            )

            #Define output binary image metadata
            metadata = {
                'driver': 'GTiff',
                'dtype': 'uint8',
                'nodata': None,
                'width': binary_mask.shape[1],
                'height': binary_mask.shape[0],
                'count': 1,
                'crs': self.mask_crs,
                'transform': self.mask_transform
            }

            #Save binary image to output path
            with rasterio.open(mask_out, 'w', **metadata) as dst:
                dst.write(binary_mask, 1)
        
    def get_performance_stats(self):
        def read_binary_raster(path):
            with rasterio.open(path) as src:
                return src.read(1)
        def find_file_by_pattern(directory, pattern):
            """
            Search for files in a given directory that contain the pattern in their filename.
            
            Parameters:
            - directory: Path to the directory to search within.
            - pattern: The pattern to look for in the filenames.
            
            Returns:
            - A list of filenames that contain the pattern.
            """
            matching_files = []
            # List all files and directories in the given directory
            for filename in os.listdir(directory):
                # Construct the full path to the item
                full_path = os.path.join(directory, filename)
                # Check if it is a file and if the pattern is in the filename
                if os.path.isfile(full_path) and pattern in filename:
                    # If a matching file is found, append its full path to the list
                    matching_files.append(full_path)
            return matching_files
        
        metrics = []
        
        for unit in self.unit_names:
            ground_truth_paths = find_file_by_pattern(self.file_manager.folder+self.file_manager.location+'/Unit_masks/', unit)
            ground_truth_path = ground_truth_paths[0] if ground_truth_paths else None
            all_model_outputs_dir = self.file_manager.folder+self.file_manager.location+'/ML_output/'
            all_model_outputs = find_file_by_pattern(all_model_outputs_dir, unit)
            
            for type in self.sam_manager.list_image_types:
                model_output_paths = [s for s in all_model_outputs if type in s]
                model_output_path = model_output_paths[0] if model_output_paths else None
                
                if model_output_path and ground_truth_path:
                    model_output = read_binary_raster(model_output_path)
                    ground_truth = read_binary_raster(ground_truth_path)
                    
                    model_output_flat = model_output.flatten()/255
                    model_output_flat = model_output_flat.astype('uint8')
                    ground_truth_flat = ground_truth.flatten()

                    # Compute confusion matrix elements
                    tn, fp, fn, tp = confusion_matrix(ground_truth_flat, model_output_flat, labels=[0, 1]).ravel()

                    # Calculate metrics
                    accuracy = accuracy_score(ground_truth_flat, model_output_flat)
                    precision = precision_score(ground_truth_flat, model_output_flat, zero_division=0)
                    recall = recall_score(ground_truth_flat, model_output_flat, zero_division=0)
                    f1 = f1_score(ground_truth_flat, model_output_flat, zero_division=0)
                    iou_temp = jaccard_score(ground_truth_flat, model_output_flat)

                    metrics_temp = {
                        'True Negatives': tn,
                        'False Positives': fp,
                        'False Negatives': fn,
                        'True Positives': tp,
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 Score': f1,
                        'IoU': iou_temp,
                    }

                    metrics.append(metrics_temp)
                    
                else:
                    continue
                
        self.metrics = metrics
        