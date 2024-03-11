import numpy as np
import os
import leafmap
from samgeo.hq_sam import SamGeo, tms_to_geotiff
from samgeo import get_basemaps
import rasterio
import geopandas as gpd
from osgeo import gdal, osr, gdalconst
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class FileManager:
    def __init__(self, folder, location, dem_name, planet_data, unit, filetype):
        self.folder = folder
        self.location = location
        self.dem_name = dem_name
        self.data_location = folder+location+'/Input_data/'
        self.ML_location = folder+location+'/ML_ready/'
        self.prompts_path = folder+location+'/Prompts/'
        self.input_dem_file = data_location+dem_name
        self.output_hillshade_file = data_location+'hill_'+dem_name
        self.output_roughness_file = data_location+'roughness_'+dem_name
        self.output_slope_file = data_location+'slope_'+dem_name
        self.prep_dem = None
        self.prep_hillshade = None
        self.prep_roughness = None
        self.prep_slope = None
        self.gaussian_dem = None
        self.gaussian_roughness= None
        self.gaussian_slope= None
        self.prep_gaussian_hillshade= None
        self.prep_gaussian_roughness= None
        self.prep_gaussian_slope= None
        self.planet_data = planet_data
        self.input_planet = data_location+planet_data
        self.prep_planet = None
        self.rgb_3band = None
        self.ave_3band = None
        self.ndvi_3band = None
        self.ndwi_3band = None
        self.rgb_3band_gaussian = None
        self.ave_3band_gaussian = None
        self.ndvi_3band_gaussian = None
        self.ndwi_3band_gaussian = None
        self.unit=unit
        self.filetype=filetype

       
    @staticmethod
    def create_writable_directory(directory_path):
        try:
            # Create a new directory with write permissions (0o777 gives full permissions)
            os.makedirs(directory_path, mode=0o777)
            print("Directory created successfully.")
            return True
        except OSError as e:
            print(f"Failed to create directory: {e}")
            return False
    
    def create_directories (self):
        create_writable_directory(self.ML_location)
        create_writable_directory(self.folder+self.location+'/ML_output/')
    
    def name_files(self):
        #Topo
        self.prep_dem = self.ML_location+self.location+'_EPSG_4326_DTM.tif'
        self.prep_hillshade = self.ML_location+self.location+'_EPSG_4326_hillshade.tif'
        self.prep_roughness = self.ML_location+self.location+'_EPSG_4326_roughness.tif'
        self.prep_slope = self.ML_location+self.location+'_EPSG_4326_slope.tif'
        self.gaussian_dem = self.data_location+'guassian_'+self.dem_name
        self.prep_gaussian_dem = self.ML_location+self.location+'_EPSG_4326_DTM_Gaussian.tif'

        #Planet
        self.prep_planet= self.data_location+'Planet_crop_EPSG_4326.tif'
        self.rgb_3band = self.ML_location+self.location+'_EPSG_4326_Planet_rgb.tif'
        self.ave_3band = self.ML_location+self.location+'_EPSG_4326_lanet_ave.tif'
        self.ndvi_3band = self.ML_location+self.location+'_EPSG_4326_Planet_ndvi.tif'
        self.ndwi_3band = self.ML_location+self.location+'_EPSG_4326_Planet_ndwi.tif'
        self.rgb_3band_gaussian = self.ML_location+self.location+'_EPSG_4326_Planet_rgb_Gaussian.tif'
        self.ave_3band_gaussian = self.ML_location+self.location+'_EPSG_4326_lanet_ave_Gaussian.tif'
        self.ndvi_3band_gaussian = self.ML_location+self.location+'_EPSG_4326_Planet_ndvi_Gaussian.tif'
        self.ndwi_3band_gaussian = self.ML_location+self.location+'_EPSG_4326_Planet_ndwi_Gaussian.tif'

        #Prompts
            # Single foreground
        self.single_foreground_prompt = self.prompts_path+self.location+'_'+self.unit+'_SingleForeground'+self.filetype
            # Multiple foreground
        self.multiple_foreground_prompts = self.prompts_path+self.location+'_'+self.unit+'_MultipleForeground'+self.filetype
            # Multiple background
        self.multiple_background_prompts = self.prompts_path+self.location+'_'+self.unit+'_MultipleBackground'+self.filetype

        #Masks
            # Single point prompt
                #Topo
        self.mask_single_dem = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_dem_'+'_'+self.unit+'.tif'
        self.mask_single_hillshade = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_hillshade_'+'_'+self.unit+'.tif'
        self.mask_single_roughness = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_roughness_'+'_'+self.unit+'.tif'
        self.mask_single_slope = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_slope_'+'_'+self.unit+'.tif'
        self.mask_single_dem_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_dem_gaussian_'+'_'+self.unit+'.tif'
        self.mask_single_hillshade_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_hillshade_gaussian_'+'_'+self.unit+'.tif'
        self.mask_single_roughness_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_roughness_gaussian_'+'_'+self.unit+'.tif'
        self.mask_single_slope_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_slope_gaussian_'+'_'+self.unit+'.tif'
                #Planet
        self.mask_single_rgb_3band = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_rgb_3band_'+'_'+self.unit+'.tif'
        self.mask_single_ave_3band = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_ave_3band_'+'_'+self.unit+'.tif'
        self.mask_single_ndvi_3band = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_ndvi_3ban_'+'_'+self.unit+'.tif'
        self.mask_single_ndwi_3band = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_ndwi_3band_'+'_'+self.unit+'.tif'
        self.mask_single_rgb_3band_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_rgb_3band_gaussian_'+'_'+self.unit+'.tif'
        self.mask_single_ave_3band_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_ave_3band_gaussian_'+'_'+self.unit+'.tif'
        self.mask_single_ndvi_3band_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_ndvi_3band_gaussian_'+'_'+self.unit+'.tif'
        self.mask_single_ndwi_3band_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_single_'+'_ndwi_3band_gaussian_'+'_'+self.unit+'.tif'
            # Multiple point prompts
                #Topo
        self.mask_multiple_dem = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_dem_'+'_'+self.unit+'.tif'
        self.mask_multiple_hillshade = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_hillshade _'+'_'+self.unit+'.tif'
        self.mask_multiple_roughness = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_roughness_'+'_'+self.unit+'.tif'
        self.mask_multiple_slope = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_slope_'+'_'+self.unit+'.tif'
        self.mask_multiple_dem_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_dem_gaussian_'+'_'+self.unit+'.tif'
        self.mask_multiple_hillshade_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_hillshade_gaussian_'+'_'+self.unit+'.tif'
        self.mask_multiple_roughness_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_roughness_gaussian_'+'_'+self.unit+'.tif'
        self.mask_multiple_slope_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_slope_gaussian_'+'_'+self.unit+'.tif'
                #Planet
        self.mask_multiple_rgb_3band = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_rgb_3band_'+'_'+self.unit+'.tif'
        self.mask_multiple_ave_3band = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_ave_3band_'+'_'+self.unit+'.tif'
        self.mask_multiple_ndvi_3band = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_ndvi_3band_'+'_'+self.unit+'.tif'
        self.mask_multiple_ndwi_3band = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_ndwi_3band_'+'_'+self.unit+'.tif'
        self.mask_multiple_rgb_3band_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_rgb_3band_gaussian_'+'_'+self.unit+'.tif'
        self.mask_multiple_ave_3band_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_ave_3band_gaussian_'+'_'+self.unit+'.tif'
        self.mask_multiple_ndvi_3band_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_ndvi_3band_gaussian_'+'_'+self.unit+'.tif'
        self.mask_multiple_ndwi_3band_gaussian = self.folder+self.location+'/ML_output/'+self.location+'_mask_multiple_'+'_ndwi_3band_gaussian_'+'_'+self.unit+'.tif'


class RasterManager:
    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.raster_src = None
        self.bounds=None
    
    def get_bounds(self):
        with rasterio.open(self.file_manager.input_dem_file) as src:
            self.raster_src = src
            # Get bounds of the raster
            self.bounds = src.bounds
        return self.bounds

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
    
    @staticmethod
    def prep_data1(input_raster, output_raster, bounds):
        warp_raster(input_raster, input_raster[:-len(".tif")]+'_warp.tif')
        crop_raster(input_raster[:-len(".tif")]+'_warp.tif',input_raster[:-len(".tif")]+'_crop.tif',bounds)
        make_three_band_image(input_raster[:-len(".tif")]+'_crop.tif', output_raster)

    @staticmethod
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

    def gaussian_topo_data(self):
        apply_gaussian_filter(self.file_manager.input_dem_file, self.file_manager.gaussian_dem, 5)
    
    def prep_topo_data(self):
        prep_data1(self.file_manager.input_dem_file, self.file_manager.prep_dem, self.bounds)
        prep_data1(self.file_manager.hillshade, self.file_manager.prep_hillshade, self.bounds)
        prep_data1(self.file_manager.roughness, self.file_manager.prep_roughness, self.bounds)
        prep_data1(self.file_manager.slope, self.file_manager.prep_slope, self.bounds)
        prep_data1(self.file_manager.gaussian_dem, self.file_manager.prep_gaussian_dem, self.bounds)
    
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
    
    def get_bounds(self):
        with rasterio.open(self.file_manager.input_planet) as src:
            self.raster_src = src
            # Get bounds of the raster
            self.bounds = src.bounds
        return self.bounds
    
    def prep_planet_data(self):
        self.raster_manager.warp_raster(self.file_manager.input_planet, self.file_manager.input_planet[:-len(".tif")]+'warp.tif')
        self.raster_manager.crop_raster(self.file_manager.input_planet[:-len(".tif")]+'warp.tif', self.file_manager.prep_planet,self.bounds)

    def make_three_band_planet(self):
        with rasterio.open(self.file_manager.prep_planet) as src:

        # Read the image bands (assuming it's a 3-band RGB image)
            self.b1 = src.read(3)
            self.b2 = src.read(2)
            self.b3 = src.read(1)
            self.b4 = src.read(4)

        # Create a new 3-band grayscale image
        profile = src.profile
        profile.update(count=3, dtype=rasterio.uint16, nodata=0.1)

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
    
    def planet_gaussian(self):
        self.raster_manager.gaussian_filter(self.file_manager.rgb_3band,self.file_manager.rgb_3band_gaussian,5)
        self.raster_manager.gaussian_filter(self.file_manager.ave_3band,self.file_manager.ave_3band_gaussian,5)
        self.raster_manager.gaussian_filter(self.file_manager.ndvi_3band,self.file_manager.ndvi_3band_gaussian,5)
        self.raster_manager.gaussian_filter(self.file_manager.ndwi_3band,self.file_manager.ndwi_3band_gaussian,5)

class ImplementingSAM:
    def __init__(self, file_manager):
        self.file_manager = file_manager
        self.sam=None
    def initiate_sam(self):
        self.sam = SamGeo(
        model_type="vit_h",
        automatic=False,
        sam_kwargs={"points_per_side": 32}
    )
    