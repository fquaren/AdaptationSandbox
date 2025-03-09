import os
import xarray as xr
import shutil
import multiprocessing
import numpy as np
from tqdm import tqdm

def compute_thresholds_for_subdir(subdir, input_dir, dem_dir):
    """Compute the elevation, temperature, and humidity values for a given subdirectory."""
    temperature_values = []
    humidity_values = []
    elevation_values = []

    subdir_path = os.path.join(input_dir, subdir)
    dem_file = f"dem_{subdir}.nc"
    dem_path = os.path.join(dem_dir, dem_file)

    if not os.path.isdir(subdir_path):
        return [], [], []  # Skip if not a directory

    # Process corresponding DEM file
    if not os.path.exists(dem_path):
        print(f"Warning: DEM file {dem_path} not found, skipping {subdir}.")
        return [], [], []

    elevation_ds = xr.open_dataset(dem_path)
    elevation_values.append(elevation_ds['HSURF'].values.mean())  # Store mean elevation

    # Process all NetCDF files in the subdirectory
    files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(".nz")]
    for file in tqdm(files):
        with xr.open_dataset(file) as ds:
            temperature_values.append(ds['T_2M'].values.mean())
            humidity_values.append(ds['RELHUM_2M'].values.mean())

    return elevation_values, temperature_values, humidity_values

def compute_thresholds(input_dir, dem_dir, num_workers=None):
    """Compute the median (or mean) threshold for elevation, temperature, and humidity in parallel."""
    # Prepare the list of subdirectories to process
    subdirs = [subdir for subdir in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir, subdir))]
    
    # Parallelize the task of computing the values for each subdirectory
    num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(compute_thresholds_for_subdir, [(subdir, input_dir, dem_dir) for subdir in subdirs])

    # Flatten the results and aggregate the values
    elevation_values = []
    temperature_values = []
    humidity_values = []
    for res in results:
        elevation_values.extend(res[0])
        temperature_values.extend(res[1])
        humidity_values.extend(res[2])

    # Compute median (or mean) thresholds
    threshold_elev = np.median(elevation_values)  # Use np.mean(elevation_values) for mean
    threshold_temp = np.median(temperature_values)
    threshold_hum = np.median(humidity_values)

    print(f"Computed Thresholds: Elevation={threshold_elev}, Temperature={threshold_temp}, Humidity={threshold_hum}")
    return threshold_elev, threshold_temp, threshold_hum

def classify_cluster(elevation, temperature, humidity, thresholds):
    """Classify the file into one of 8 clusters based on threshold conditions."""
    threshold_elev, threshold_temp, threshold_hum = thresholds
    cluster = 0
    if elevation > threshold_elev:
        cluster |= 1  # Set bit 1
    if temperature > threshold_temp:
        cluster |= 2  # Set bit 2
    if humidity > threshold_hum:
        cluster |= 4  # Set bit 3
    return cluster

def process_file(args):
    """Process a single NetCDF file: classify it and move it to the correct cluster."""
    file_path, dem_path, output_dir, thresholds = args

    # Extract subdirectory name (e.g., "0_0")
    subdir = os.path.basename(os.path.dirname(file_path))

    # Open the DEM file for elevation data
    if not os.path.exists(dem_path):
        print(f"Warning: DEM file {dem_path} not found, skipping {file_path}.")
        return

    elevation_ds = xr.open_dataset(dem_path)
    elevation = elevation_ds['HSURF'].values.mean()  # Compute mean elevation

    # Open the NetCDF file and extract temperature & humidity
    with xr.open_dataset(file_path) as ds:
        temperature = ds['T_2M'].values.mean()
        humidity = ds['RELHUM_2M'].values.mean()

    # Classify the file into a cluster
    cluster = classify_cluster(elevation, temperature, humidity, thresholds)
    print(f"File {os.path.basename(file_path)} (in {subdir}) classified into cluster {cluster}")

    # Move file to the corresponding cluster folder
    dest_path = os.path.join(output_dir, f'cluster_{cluster}', os.path.basename(file_path))
    shutil.copy(file_path, dest_path)
    print(f"Copied {os.path.basename(file_path)} to cluster_{cluster}")

def process_netcdf_files_parallel(input_dir, dem_dir, output_dir, thresholds, num_workers=None):
    """Process all NetCDF files in parallel using multiprocessing."""
    
    # Ensure cluster directories exist
    for i in range(8):
        os.makedirs(os.path.join(output_dir, f'cluster_{i}'), exist_ok=True)

    # Create a list of files to process
    file_list = []
    
    for subdir in sorted(os.listdir(input_dir)):  # Ensure sorted order
        subdir_path = os.path.join(input_dir, subdir)

        # Ensure it's a valid directory
        if not os.path.isdir(subdir_path):
            continue

        # Corresponding DEM file
        dem_file = f"dem_{subdir}.nc"
        dem_path = os.path.join(dem_dir, dem_file)

        # Get all NetCDF files in the subdirectory
        files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(".nz")]

        # Store arguments for parallel processing
        for file in files:
            file_list.append((file, dem_path, output_dir, thresholds))

    # Run in parallel using multiprocessing
    num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)  # Use max available CPUs minus 1
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_file, file_list)

# Example usage
if __name__ == "__main__":
    # data_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data"
    data_dir = "/Users/fquareng/data"
    input_directory = os.path.join(data_dir, "1h_2D_sel_cropped_gridded")
    output_directory = os.path.join(data_dir, "1h_2D_sel_cropped_gridded_clustered_threshold")
    dem_directory =  os.path.join(data_dir, "dem_squares")

    print("Computing thresholds...")
    computed_thresholds = compute_thresholds(input_directory, dem_directory)

    process_netcdf_files_parallel(input_directory, dem_directory, output_directory, computed_thresholds)
