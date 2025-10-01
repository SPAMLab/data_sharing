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
from whitebox_workflows import WbEnvironment
import whitebox_workflows as wbw
import os
import glob
import gc

# ------------------------------------------------------------
# Whitebox environment setup
# ------------------------------------------------------------
wbe = WbEnvironment()
wbe.verbose = True

try:
    # ------------------------------------------------------------
    # Base directories
    # ------------------------------------------------------------
    base_dir = "/your/directory/here/"
    las_input_dir = os.path.join(base_dir, "lidar_raw")
    ground_dir = os.path.join(base_dir, "lidar_ground")
    joined_dir = os.path.join(base_dir, "lidar_merged")
    dtm_dir = os.path.join(base_dir, "dtm_output")

    # ------------------------------------------------------------
    # Create output directories
    # ------------------------------------------------------------
    os.makedirs(ground_dir, exist_ok=True)
    os.makedirs(joined_dir, exist_ok=True)
    os.makedirs(dtm_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Parameters (editable)
    # ------------------------------------------------------------
    max_tin_edge = 1000  # Max triangle edge for TIN interpolation
    cell_size = 2.0

    # ------------------------------------------------------------
    # Step 1: Filter ground points (class 2)
    # ------------------------------------------------------------
    las_files = [f for f in os.listdir(las_input_dir) if f.endswith('.las') and not f.endswith('_ground.las')]
    for las in las_files:
        ground_las_path = os.path.join(ground_dir, f'{las[:-4]}_ground.las')
        if os.path.isfile(ground_las_path):
            print(f"[1] [SKIPPED] {ground_las_path} already exists.")
            continue

        print(f"[1] Filtering ground: {las}")
        lidar = wbe.read_lidar(os.path.join(las_input_dir, las))
        ground = wbe.filter_lidar_classes(lidar, exclusion_classes=[1, *range(3, 19)])

        wbe.write_lidar(ground, ground_las_path)
        print(f"[1] Ground saved to: {ground_las_path}")

        del lidar, ground
        gc.collect()

    # ------------------------------------------------------------
    # Step 2: Merge *_ground.las files
    # ------------------------------------------------------------
    las_merged_path = os.path.join(joined_dir, "all_ground_joined.las")
    if os.path.exists(las_merged_path):
        print(f"[2] Merged file already exists, skipping: {las_merged_path}")
    else:
        ground_files = glob.glob(os.path.join(ground_dir, '*_ground.las'))
        if not ground_files:
            raise FileNotFoundError("No *_ground.las files found.")

        print(f"[2] Merging {len(ground_files)} ground LAS files...")

        lidar_list = [wbe.read_lidar(f) for f in ground_files]
        lidar_merged = wbe.lidar_join(lidar_list)
        wbe.write_lidar(lidar_merged, las_merged_path)

        print(f"[2] Merge completed: {las_merged_path}")

        del lidar_list, lidar_merged
        gc.collect()

    # ------------------------------------------------------------
    # Step 3: Generate DTM via TIN
    # ------------------------------------------------------------
    dtm_raw_path = os.path.join(dtm_dir, f'dtm_tin{max_tin_edge}.tif')
    if os.path.exists(dtm_raw_path):
        print(f"[3] DTM already exists, skipping TIN interpolation: {dtm_raw_path}")
    else:
        print(f"[3] Generating DTM via TIN...")
        lidar_data = wbe.read_lidar(las_merged_path)

        dem = wbe.lidar_tin_gridding(
            input_lidar=lidar_data,
            interpolation_parameter='elevation',
            returns_included='all',
            excluded_classes=[1, *range(3, 10), 17],
            cell_size=cell_size,
            max_triangle_edge_length=max_tin_edge
        )

        wbe.write_raster(dem, dtm_raw_path, compress=True)
        print(f"[3] DTM saved to: {dtm_raw_path}")

        del dem, lidar_data
        gc.collect()

    # ------------------------------------------------------------
    # Step 4: Fill missing values in DTM
    # ------------------------------------------------------------
    dtm_filled_path = os.path.join(dtm_dir, f'dtm_tin{max_tin_edge}_filled.tif')
    if os.path.exists(dtm_filled_path):
        print(f"[4] Filled DTM already exists, skipping: {dtm_filled_path}")
    else:
        print(f"[4] Filling gaps in DTM...")

        dtm_raw = wbe.read_raster(dtm_raw_path)
        dtm_filled = wbe.fill_missing_data(
            dem=dtm_raw,
            filter_size=51,
            weight=2.0,
            exclude_edge_nodata=True
        )

        wbe.write_raster(dtm_filled, dtm_filled_path, compress=True)
        print(f"[4] Filled DTM saved to: {dtm_filled_path}")

        del dtm_raw, dtm_filled
        gc.collect()

finally:
    print("[DONE] Processing completed successfully.")
