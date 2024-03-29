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


class FileManager:
    def __init__(self, folder, location, dem_name, planet_data):
        self.folder = folder
        self.location = location
        self.dem_name = dem_name
        self.data_location = folder+location+'/Input_data/'
        self.ML_location = folder+location+'/ML_ready/'
        self.prompts_path = folder+location+'/Prompts/'
        self.input_dem_file = None
        self.output_hillshade_file = None
        self.output_roughness_file = None
        self.output_slope_file = None
        self.prep_dem = None
        self.prep_hillshade = None
        self.prep_roughness = None
        self.prep_slope = None
        self.gaussian_dem = None
        self.gaussian_roughness= None
        self.gaussian_slope= None
        self.prep_gaussian_dem = None
        self.prep_gaussian_hillshade= None
        self.prep_gaussian_roughness= None
        self.prep_gaussian_slope= None
        self.planet_data = planet_data
        self.input_planet = None
        self.prep_planet = None
        self.rgb_3band = None
        self.ave_3band = None
        self.ndvi_3band = None
        self.ndwi_3band = None
        self.rgb_3band_gaussian = None
        self.ave_3band_gaussian = None
        self.ndvi_3band_gaussian = None
        self.ndwi_3band_gaussian = None
        

    
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

        create_writable_directory(self.ML_location)
        create_writable_directory(self.folder+self.location+'/ML_output/')
        create_writable_directory(self.folder+self.location+'/Input_data/')
        create_writable_directory(self.folder+self.location+'/Prompts/')
        create_writable_directory(self.folder+self.location+'/Unit_masks/')
    
    def name_files(self):
        #Topo
        self.input_dem_file = self.data_location+self.dem_name
        self.output_hillshade_file = self.data_location+'hill_'+self.dem_name
        self.output_roughness_file = self.data_location+'roughness_'+self.dem_name
        self.output_slope_file = self.data_location+'slope_'+self.dem_name
        self.prep_dem = self.ML_location+self.location+'_EPSG_4326_DTM.tif'
        self.prep_hillshade = self.ML_location+self.location+'_EPSG_4326_hillshade.tif'
        self.prep_roughness = self.ML_location+self.location+'_EPSG_4326_roughness.tif'
        self.prep_slope = self.ML_location+self.location+'_EPSG_4326_slope.tif'
        self.gaussian_dem = self.data_location+'guassian_'+self.dem_name
        self.prep_gaussian_dem = self.ML_location+self.location+'_EPSG_4326_DTM_Gaussian.tif'
        

        #Planet
        self.input_planet = self.data_location+self.planet_data
        self.prep_planet= self.data_location+'Planet_crop_EPSG_4326.tif'
        self.rgb_3band = self.ML_location+self.location+'_EPSG_4326_Planet_rgb.tif'
        self.ave_3band = self.ML_location+self.location+'_EPSG_4326_Planet_ave.tif'
        self.ndvi_3band = self.ML_location+self.location+'_EPSG_4326_Planet_ndvi.tif'
        self.ndwi_3band = self.ML_location+self.location+'_EPSG_4326_Planet_ndwi.tif'
        self.rgb_3band_gaussian = self.ML_location+self.location+'_EPSG_4326_Planet_rgb_Gaussian.tif'
        self.ave_3band_gaussian = self.ML_location+self.location+'_EPSG_4326_Planet_ave_Gaussian.tif'
        self.ndvi_3band_gaussian = self.ML_location+self.location+'_EPSG_4326_Planet_ndvi_Gaussian.tif'
        self.ndwi_3band_gaussian = self.ML_location+self.location+'_EPSG_4326_Planet_ndwi_Gaussian.tif'

class RasterManager:
    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.raster_src = None
        self.bounds=None
    

    def calculate_topo_derivs(self,azimuth=315, altitude=45):
        gdal.DEMProcessing(self.file_manager.output_hillshade_file, self.file_manager.input_dem_file, "hillshade", azimuth=str(azimuth), altitude=str(altitude))
        gdal.DEMProcessing(self.file_manager.output_roughness_file, self.file_manager.input_dem_file, "roughness")
        gdal.DEMProcessing(self.file_manager.output_slope_file, self.file_manager.input_dem_file, "slope")
   
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
    
   
    def get_bounds(self):
        self.warp_raster(self.file_manager.input_dem_file, self.file_manager.input_dem_file[:-len(".tif")]+'_warp.tif')
        with rasterio.open(self.file_manager.input_dem_file[:-len(".tif")]+'_warp.tif') as src:
            # Get bounds of the raster
            self.bounds = src.bounds
    
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
            profile.update(count=3, dtype=rasterio.uint8, nodata=0.1)

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

            # Apply Gaussian filter with 5-meter scale
            sigma = input_scale / dataset.GetGeoTransform()[1]  # Convert 5 meters to pixels
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
            
        apply_gaussian_filter(self.file_manager.input_dem_file, self.file_manager.gaussian_dem, scale)
    
    def prep_data(self, input_raster, output_raster):
        self.warp_raster(input_raster, input_raster[:-len(".tif")]+'_warp.tif')
        self.crop_raster(input_raster[:-len(".tif")]+'_warp.tif',input_raster[:-len(".tif")]+'_crop.tif',self.bounds)
        self.make_three_band_image(input_raster[:-len(".tif")]+'_crop.tif', output_raster)
    
    def prep_topo_data(self):
        
        self.prep_data(self.file_manager.input_dem_file, self.file_manager.prep_dem)
        self.prep_data(self.file_manager.output_hillshade_file, self.file_manager.prep_hillshade)
        self.prep_data(self.file_manager.output_roughness_file, self.file_manager.prep_roughness)
        self.prep_data(self.file_manager.output_slope_file, self.file_manager.prep_slope)
        self.prep_data(self.file_manager.gaussian_dem, self.file_manager.prep_gaussian_dem)
    
class PlanetManager:
    def __init__(self, file_manager,raster_manager):
        self.file_manager = file_manager
        self.raster_manager = raster_manager
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
    
    def prep_planet_data(self):
        self.warp_raster(self.file_manager.input_planet, self.file_manager.input_planet[:-len(".tif")]+'_warp.tif')
        self.crop_raster(self.file_manager.input_planet[:-len(".tif")]+'_warp.tif', self.file_manager.prep_planet,self.raster_manager.get_bounds())

    def make_three_band_planet(self):
        with rasterio.open(self.file_manager.prep_planet) as src:

        # Read the image bands (assuming it's a 3-band RGB image)
            self.b1 = src.read(3).astype(float)
            self.b2 = src.read(2).astype(float)
            self.b3 = src.read(1).astype(float)
            self.b4 = src.read(4).astype(float)

        # Create a new 3-band grayscale image
            profile = src.profile
            profile.update(count=3, dtype=rasterio.float32)

        np.seterr(divide='ignore', invalid='ignore')
        self.rgb_3band = np.stack((self.b1,self.b2,self.b3), axis=0)

        with rasterio.open(self.file_manager.rgb_3band, 'w', **profile) as dst:
            dst.write(self.rgb_3band)

        ave_band = (self.b1+self.b2+self.b3)/3
        self.ave_3band = np.stack((ave_band,ave_band,ave_band), axis=0)

        with rasterio.open(self.file_manager.ave_3band, 'w', **profile) as dst:
            dst.write(self.ave_3band)

        
        ndvi_band = (self.b4-self.b1)/(self.b4+self.b1)
        self.ndvi_3band = np.stack((ndvi_band,ndvi_band,ndvi_band), axis=0)

        with rasterio.open(self.file_manager.ndvi_3band, 'w', **profile) as dst:
            dst.write(self.ndvi_3band)

        ndwi_band = (self.b2-self.b4)/(self.b2+self.b4)
        self.ndwi_3band = np.stack((ndwi_band,ndwi_band,ndwi_band), axis=0)

        with rasterio.open(self.file_manager.ndwi_3band, 'w', **profile) as dst:
            dst.write(self.ndwi_3band)
    
    def planet_gaussian(self, scale):
        self.raster_manager.gaussian_filter(self.file_manager.rgb_3band,self.file_manager.rgb_3band_gaussian,scale)
        self.raster_manager.gaussian_filter(self.file_manager.ave_3band,self.file_manager.ave_3band_gaussian,scale)
        self.raster_manager.gaussian_filter(self.file_manager.ndvi_3band,self.file_manager.ndvi_3band_gaussian,scale)
        self.raster_manager.gaussian_filter(self.file_manager.ndwi_3band,self.file_manager.ndwi_3band_gaussian,scale)

class PromptManager:
    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.single_foreground_prompts = None
        self.multiple_foreground_prompts = None
        self.multiple_background_prompts = None
        self.geologic_units = None
        self.coords_single= None
        self.coords_multiple= None
        self.labels_single= None
        self.labels_multiple= None
        
    def prompts_files (self):
        self.single_foreground_prompts = [f for f in os.listdir(self.file_manager.prompts_path) if os.path.isfile(os.path.join(self.file_manager.prompts_path, f)) and f.endswith("_single_foreground.shp")]
        self.multiple_foreground_prompts = [f for f in os.listdir(self.file_manager.prompts_path) if os.path.isfile(os.path.join(self.file_manager.prompts_path, f)) and f.endswith("_multiple_foreground.shp")]
        self.multiple_background_prompts = [f for f in os.listdir(self.file_manager.prompts_path) if os.path.isfile(os.path.join(self.file_manager.prompts_path, f)) and f.endswith("_multiple_background.shp")]
        
        self.geologic_units = [self.multiple_foreground_prompts[a][:-len("_multiple_foreground.shp")] for a in range(0,len(self.multiple_foreground_prompts))]

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

        single_foreground_prompts = [self.file_manager.prompts_path+self.single_foreground_prompts[a] for a in range (0,len(self.single_foreground_prompts))]
        multiple_foreground_prompts = [self.file_manager.prompts_path+self.multiple_foreground_prompts[a] for a in range (0,len(self.multiple_foreground_prompts))]
        multiple_background_prompts = [self.file_manager.prompts_path+self.multiple_background_prompts[a] for a in range (0,len(self.multiple_background_prompts))]

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
        
    def initiate_sam(self, sam_kwargs, auto = False):
        self.sam = SamGeo(
        checkpoint_dir= "/content/drive/MyDrive/USGS_ML_2024/geomap_10_examples/checkpoint/",
        model_type="vit_h",
        automatic=auto,
        **sam_kwargs
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
        