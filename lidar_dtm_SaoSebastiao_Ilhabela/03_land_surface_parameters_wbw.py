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
# Input and output directories
# ------------------------------------------------------------
base_dir = "/your/directory/here/"
dtm_dir = os.path.join(base_dir, "dtm_files")
input_dem_path = os.path.join(dtm_dir, "dtm.tif")

output_lsp_dir = os.path.join(dtm_dir, "lsp_outputs")
os.makedirs(output_lsp_dir, exist_ok=True)

# ======================== Land Surface Parameters PROCESSING ========================
wbe = None
try:
    # Initialize Whitebox Workflows environment
    print("Setting up Whitebox Workflows environment...")
    wbe = WbEnvironment()
    wbe.verbose = True
    wbe.working_directory = output_lsp_dir

    if not os.path.exists(input_dem_path):
        raise FileNotFoundError(f"DEM file '{input_dem_path}' not found.")

    dem = wbe.read_raster(input_dem_path)

    print("\nCorrecting DEM depressions for hydrological use...")
    filled_dem = wbe.breach_depressions_least_cost(dem, max_dist=900)
    print("Depression filling completed.")

    hydro_specific_tools = ["flow_accumulation", "drainage_direction"]

    # Define LSP functions to compute
    lsp_functions = {
        "hillshade": wbe.multidirectional_hillshade,
        "slope": wbe.slope,
        "aspect": wbe.aspect,
        "profile_curvature": wbe.profile_curvature,
        "plan_curvature": wbe.plan_curvature,
        "minimal_curvature": wbe.minimal_curvature,
        "maximal_curvature": wbe.maximal_curvature,
        "difference_from_mean_elevation": wbe.difference_from_mean_elevation,
        "geomorphons": wbe.geomorphons,
        "spherical_std_dev_of_normals": wbe.spherical_std_dev_of_normals,
        "flow_accumulation": wbe.qin_flow_accumulation,
        "wetness_index": wbe.wetness_index,
        "drainage_direction": wbe.d8_pointer,
        "streams": wbe.extract_streams,
        "shape_index": wbe.shape_index,  # Requires Whitebox Workflows Professional (WbW-Pro) license
    }

    outputs = {}
    for name, func in lsp_functions.items():
        print(f"\nProcessing {name}...")

        if name == "shape_index":
            print("'shape_index' requires a Whitebox Workflows Professional (WbW-Pro) license.")

        if name == "wetness_index":
            slope_raster = outputs["slope"]
            sca_raster = outputs["flow_accumulation"]
            result = func(specific_catchment_area=sca_raster, slope=slope_raster)

        elif name == "streams":
            flow_accum_raster = outputs["flow_accumulation"]
            channel_threshold = 4000.0
            result = func(flow_accumulation=flow_accum_raster, threshold=channel_threshold)

        elif name in hydro_specific_tools:
            result = func(filled_dem)

        elif name in [
            "plan_curvature",
            "profile_curvature",
            "minimal_curvature",
            "maximal_curvature"
        ]:
            result = func(dem, log_transform=True)

        else:
            result = func(dem)

        output_path = os.path.join(output_lsp_dir, f"{name}.tif")
        wbe.write_raster(result, output_path, compress=True)
        outputs[name] = result

        gc.collect()
        print(f"{name} saved to: {output_path}")

    del dem
    del filled_dem
    del outputs
    gc.collect()

except Exception as e:
    print(f"Processing error: {e}")
