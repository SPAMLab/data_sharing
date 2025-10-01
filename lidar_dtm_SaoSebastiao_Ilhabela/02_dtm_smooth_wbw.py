# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------
# Scripts to create a smoothed DTM and LSPs from airborne lidar data 
#
# Scripts by Rebeca Durço Coelho and Gabriella Labate Frugis
#
# SPAMLab - Spatial Analysis and Modelling Lab
# Institute of Astronomy, Geophysical and Atmospheric Sciences
# Universidade de São Paulo - Brazil
#
# Project: 
# Multi-Scale Geomorphometric Analysis of Mass Movements in São Sebastião (SP, Brazil).
# funding: FAPESP (2023/11197-1)
# https://bv.fapesp.br/57077
# P.I.: Carlos Henrique Grohmann 
#
#
# The scripts use WhiteboxWorkflows as main processing engine.
# Some steps require the professional license
#
# ------------------------------------------------------------
# ------------------------------------------------------------
# ------------------------------------------------------------




# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
import os
import gc
from whitebox_workflows import WbEnvironment

# ------------------------------------------------------------
# Base directories
# ------------------------------------------------------------
base_dir = "/your/directory/here/"
dtm_dir = os.path.join(base_dir, "dtm_output")

# Input and output DEM paths
input_dem_path = os.path.join(dtm_dir, "dtm_filled.tif")
output_dem_path = os.path.join(dtm_dir, "dtm_smoothed.tif")

# Create output directory if needed
os.makedirs(dtm_dir, exist_ok=True)

# === Parameters - RemoveOffTerrainObjects ===
rott_filter_size = 7 
rott_slope_threshold = 11.0

# === Parameters - SmoothVegetationResidual (Requires Whitebox Workflows Professional (WbW-Pro) license) ===
sv_max_scale = 15
sv_dev_threshold = 0.06
sv_scale_threshold = 4
sv_num_passes = 2

# === Parameters - FeaturePreservingSmoothing ===
fps_filter_size = 3
fps_normal_diff_threshold = 12.0
fps_iterations = 1

# ======================== DEM SMOOTHING WORKFLOW ========================
wbe = WbEnvironment()
wbe.verbose = True

try:
    # Check if input DEM exists
    if not os.path.exists(input_dem_path):
        raise FileNotFoundError(f"Input DEM not found: {input_dem_path}")

    # Create temporary working directory
    temp_dir = os.path.join(dtm_dir, "temp_dem_processing")
    os.makedirs(temp_dir, exist_ok=True)
    wbe.working_directory = temp_dir
    print(f"Temporary working directory: {temp_dir}")

    # === Step 1: RemoveOffTerrainObjects ===
    print("\n--- Step 1: RemoveOffTerrainObjects ---")
    print(f"Parameters: filter_size={rott_filter_size}, slope_threshold={rott_slope_threshold}")
    dem_raster = wbe.read_raster(input_dem_path)

    dem_rotoff = wbe.remove_off_terrain_objects(
        dem=dem_raster,
        filter_size=rott_filter_size,
        slope_threshold=rott_slope_threshold
    )

    dem_rotoff_path = os.path.join(temp_dir, "dem_rotoff.tif")
    wbe.write_raster(dem_rotoff, dem_rotoff_path, compress=True)
    print(f"Output saved: {dem_rotoff_path}")

    del dem_raster, dem_rotoff
    gc.collect()

    # === Step 2: SmoothVegetationResidual (Requires Whitebox Workflows Professional (WbW-Pro)) ===
    current_input_path = dem_rotoff_path
    for i in range(sv_num_passes):
        print(f"\n--- Step 2: SmoothVegetationResidual - Pass {i+1}/{sv_num_passes} ---")
        current_raster = wbe.read_raster(current_input_path)

        dem_smooth = wbe.smooth_vegetation_residual(
            dem=current_raster,
            max_scale=sv_max_scale,
            dev_threshold=sv_dev_threshold,
            scale_threshold=sv_scale_threshold
        )

        current_input_path = os.path.join(temp_dir, f"dem_smooth_pass{i+1}.tif")
        wbe.write_raster(dem_smooth, current_input_path, compress=True)
        print(f"Output saved: {current_input_path}")

        del current_raster, dem_smooth
        gc.collect()

    # === Step 3: FeaturePreservingSmoothing ===
    print("\n--- Step 3: FeaturePreservingSmoothing ---")
    print(f"Parameters: filter_size={fps_filter_size}, normal_diff_threshold={fps_normal_diff_threshold}, iterations={fps_iterations}")
    input_fps_raster = wbe.read_raster(current_input_path)

    dem_fps = wbe.feature_preserving_smoothing(
        dem=input_fps_raster,
        filter_size=fps_filter_size,
        normal_diff_threshold=fps_normal_diff_threshold,
        iterations=fps_iterations
    )

    wbe.write_raster(dem_fps, output_dem_path, compress=True)
    print(f"\nFinal smoothed DEM saved to: {output_dem_path}")
    print("[DONE] DEM smoothing workflow completed.")

    del input_fps_raster, dem_fps
    gc.collect()

except Exception as e:
    print(f"\nProcessing error: {e}")

finally:
    if 'temp_dir' in locals() and os.path.exists(temp_dir):
        try:
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            os.rmdir(temp_dir)
            print(f"Temporary directory removed: {temp_dir}")
        except OSError as e:
            print(f"Warning: Failed to remove temporary directory: {e}")
