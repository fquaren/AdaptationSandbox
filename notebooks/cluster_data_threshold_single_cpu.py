import os
import xarray as xr
import shutil
import numpy as np
from tqdm import tqdm

def compute_thresholds_for_subdir(subdir, input_dir, dem_dir):
    """Compute the elevation, temperature, and humidity values for a given subdirectory."""
    temperature_values = []
    humidity_values = []
    elevation_values = []

    subdir_path = os.path.join(input_dir, subdir)
    dem_file = f"{subdir}_dem.nc"
    dem_path = os.path.join(dem_dir, dem_file)

    if not os.path.isdir(subdir_path):
        return [], [], []  # Skip if not a directory

    if not os.path.exists(dem_path):
        print(f"Warning: DEM file {dem_path} not found, skipping {subdir}.")
        return [], [], []

    elevation_ds = xr.open_dataset(dem_path)
    elevation_values.append(elevation_ds['HSURF'].values.mean())

    files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(".nz")]
    for file in tqdm(files):
        with xr.open_dataset(file) as ds:
            temperature_values.append(ds['T_2M'].values.mean())
            humidity_values.append(ds['RELHUM_2M'].values.mean())

    return elevation_values, temperature_values, humidity_values

def compute_thresholds(input_dir, dem_dir):
    """Compute the median threshold for elevation, temperature, and humidity sequentially."""
    subdirs = [subdir for subdir in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir, subdir))]

    elevation_values = []
    temperature_values = []
    humidity_values = []

    for subdir in subdirs:
        res = compute_thresholds_for_subdir(subdir, input_dir, dem_dir)
        elevation_values.extend(res[0])
        temperature_values.extend(res[1])
        humidity_values.extend(res[2])

    threshold_elev = np.median(elevation_values)
    threshold_temp = np.median(temperature_values)
    threshold_hum = np.median(humidity_values)

    print(f"Computed Thresholds: Elevation={threshold_elev}, Temperature={threshold_temp}, Humidity={threshold_hum}")
    return threshold_elev, threshold_temp, threshold_hum

def classify_cluster(elevation, temperature, humidity, thresholds):
    """Classify the file into one of 8 clusters based on threshold conditions."""
    threshold_elev, threshold_temp, threshold_hum = thresholds
    cluster = 0
    if elevation > threshold_elev:
        cluster |= 1
    if temperature > threshold_temp:
        cluster |= 2
    if humidity > threshold_hum:
        cluster |= 4
    return cluster

def process_file(file_path, dem_path, output_dir, thresholds):
    """Process a single NetCDF file: classify it and move it to the correct cluster."""
    subdir = os.path.basename(os.path.dirname(file_path))
    
    if not os.path.exists(dem_path):
        print(f"Warning: DEM file {dem_path} not found, skipping {file_path}.")
        return

    elevation_ds = xr.open_dataset(dem_path)
    elevation = elevation_ds['HSURF'].values.mean()

    with xr.open_dataset(file_path) as ds:
        temperature = ds['T_2M'].values.mean()
        humidity = ds['RELHUM_2M'].values.mean()

    cluster = classify_cluster(elevation, temperature, humidity, thresholds)
    print(f"File {os.path.basename(file_path)} (in {subdir}) classified into cluster {cluster}")

    dest_path = os.path.join(output_dir, f'cluster_{cluster}', os.path.basename(file_path))
    shutil.copy(file_path, dest_path)
    print(f"Copied {os.path.basename(file_path)} to cluster_{cluster}")

def process_netcdf_files(input_dir, dem_dir, output_dir, thresholds):
    """Process all NetCDF files sequentially."""
    for i in range(8):
        os.makedirs(os.path.join(output_dir, f'cluster_{i}'), exist_ok=True)

    for subdir in sorted(os.listdir(input_dir)):
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        dem_file = f"{subdir}_dem.nc"
        dem_path = os.path.join(dem_dir, dem_file)
        files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(".nz")]

        for file in files:
            process_file(file, dem_path, output_dir, thresholds)

if __name__ == "__main__":
    data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data"
    input_directory = os.path.join(data_dir, "DA/8h-PS-RELHUM_2M-T_2M_cropped_gridded")
    output_directory = os.path.join(data_dir, "DA/8h-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_threshold")
    dem_directory = os.path.join(data_dir, "dem_squares")

    print("Computing thresholds...")
    computed_thresholds = compute_thresholds(input_directory, dem_directory)

    process_netcdf_files(input_directory, dem_directory, output_directory, computed_thresholds)
