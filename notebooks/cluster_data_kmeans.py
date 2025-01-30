import os
import xarray as xr
import shutil
import multiprocessing
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

def collect_data_for_subdir(subdir, input_dir, dem_dir):
    """Collect temperature, humidity, and elevation values for a given subdirectory."""
    temperature_values = []
    humidity_values = []
    elevation_values = []
    pressure_values = []

    subdir_path = os.path.join(input_dir, subdir)
    dem_file = f"dem_{subdir}.nc"
    dem_path = os.path.join(dem_dir, dem_file)

    if not os.path.isdir(subdir_path):
        print(f"Warning: Skipping non-directory {subdir_path}")
        return [], [], []  # Skip if not a directory

    # Process corresponding DEM file
    if not os.path.exists(dem_path):
        print(f"Warning: DEM file {dem_path} not found, skipping {subdir}.")
        return [], [], []

    try:
        elevation_ds = xr.open_dataset(dem_path)
        mean_elevation = elevation_ds['HSURF'].values.mean()  # Compute mean elevation
    except Exception as e:
        print(f"Error reading DEM file {dem_path}: {e}")
        return [], [], []

    # Process all NetCDF files in the subdirectory
    files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(".nz")]
    for file in tqdm(files, desc=f"Processing {subdir}"):
        try:
            with xr.open_dataset(file) as ds:
                temp_values = ds['T_2M'].values.flatten()  # Flatten to ensure 1D array
                hum_values = ds['RELHUM_2M'].values.flatten()
                sp_values = ds['PS'].values.flatten()

                # Repeat elevation for each time step
                elev_values = np.full_like(temp_values, mean_elevation)

                temperature_values.extend(temp_values)
                humidity_values.extend(hum_values)
                pressure_values.extend(sp_values)
                elevation_values.extend(elev_values)
        except Exception as e:
            print(f"Error processing file {file}: {e}")

    return elevation_values, temperature_values, humidity_values, pressure_values


def collect_data(input_dir, dem_dir, num_workers=None):
    """Collect data from all subdirectories using multiprocessing."""
    subdirs = [subdir for subdir in sorted(os.listdir(input_dir)) if os.path.isdir(os.path.join(input_dir, subdir))]
    
    # Prepare arguments for parallel execution
    subdir_args = [(subdir, input_dir, dem_dir) for subdir in subdirs]
    print(f"Processing {len(subdir_args)} subdirectories.")

    num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)  # Use available CPUs - 1
    print("Using {} workers for parallel processing.".format(num_workers))

    print("Collecting data in parallel...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = list(tqdm(pool.starmap(collect_data_for_subdir, subdir_args), total=len(subdir_args)))

    # Combine results
    elevation_values, temperature_values, humidity_values, pressure_values = zip(*results)

    # Flatten lists
    elevation_values = np.concatenate(elevation_values)
    temperature_values = np.concatenate(temperature_values)
    humidity_values = np.concatenate(humidity_values)
    pressure_values = np.concatenate(pressure_values)
    assert elevation_values.shape == temperature_values.shape == humidity_values.shape == pressure_values.shape, "Data shapes do not match."

    print(f"Collected {len(elevation_values)} samples for clustering.")

    return elevation_values, temperature_values, humidity_values, pressure_values


def apply_kmeans_clustering(elevation_values, temperature_values, humidity_values, pressure_values, max_clusters=10):
    """Apply KMeans clustering to the data using the elbow method to determine optimal clusters."""
    # Ensure all arrays have the same length
    assert elevation_values.shape == temperature_values.shape == humidity_values.shape == pressure_values.shape, "Data shapes do not match."

    # Stack the values into a single matrix for clustering
    data = np.stack([elevation_values, temperature_values, humidity_values, pressure_values], axis=1)

    # # Calculate inertia for different numbers of clusters
    # inertias = []
    # for n_clusters in range(1, max_clusters + 1):
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    #     kmeans.fit(data)
    #     inertias.append(kmeans.inertia_)

    # # Plot the inertia vs. number of clusters to find the elbow
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(1, max_clusters + 1), inertias, marker='o')
    # plt.title("Elbow Method for Optimal Clusters")
    # plt.xlabel("Number of Clusters")
    # plt.ylabel("Inertia")
    # plt.grid(True)
    # plt.savefig('elbow_method.png')
    # plt.show()

    # Choose the optimal number of clusters (elbow point)
    # optimal_clusters = np.argmin(np.diff(inertias)) + 2  # Plus 2 due to diff
    optimal_clusters = 5
    print(f"Optimal number of clusters: {optimal_clusters}")

    # Perform KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    kmeans.fit(data)

    return kmeans, optimal_clusters


def process_file(args):
    """Process a single NetCDF file: classify it into a cluster and move it."""
    file_path, dem_path, output_dir, kmeans_model = args

    # Open the DEM file for elevation data
    if not os.path.exists(dem_path):
        print(f"Warning: DEM file {dem_path} not found, skipping {file_path}.")
        return

    try:
        elevation_ds = xr.open_dataset(dem_path)
        elevation = elevation_ds['HSURF'].values.mean()  # Compute mean elevation
    except Exception as e:
        print(f"Error reading DEM file {dem_path}: {e}")
        return

    # Open the NetCDF file and extract temperature, humidity, and pressure
    try:
        with xr.open_dataset(file_path) as ds:
            temperature = ds['T_2M'].values.mean()
            humidity = ds['RELHUM_2M'].values.mean()
            pressure = ds['PS'].values.mean()
    except Exception as e:
        print(f"Error reading NetCDF file {file_path}: {e}")
        return

    # Prepare the data point
    data_point = np.array([elevation, temperature, humidity, pressure]).reshape(1, -1)

    # Predict the cluster label using the KMeans model
    cluster_label = kmeans_model.predict(data_point)[0]

    # Move the file to the corresponding cluster folder
    dest_path = os.path.join(output_dir, f'cluster_{cluster_label}', os.path.basename(file_path))
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    shutil.copy(file_path, dest_path)


def process_netcdf_files_parallel(input_dir, dem_dir, output_dir, kmeans_model, num_clusters, num_workers=None):
    """Process all NetCDF files in parallel using multiprocessing."""
    
    # Ensure cluster directories exist, and clean them if necessary
    for i in range(num_clusters):
        cluster_dir = os.path.join(output_dir, f'cluster_{i}')
        if os.path.exists(cluster_dir):
            shutil.rmtree(cluster_dir)  # Clean up old files
        os.makedirs(cluster_dir, exist_ok=True)

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
            file_list.append((file, dem_path, output_dir, kmeans_model))

    # Run in parallel using multiprocessing
    num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)  # Use max available CPUs minus 1
    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(process_file, file_list)

# Example usage
if __name__ == "__main__":
    input_directory = "/Users/fquareng/data/1h_2D_sel_cropped_blurred_x8_gridded"
    output_directory = "/Users/fquareng/data/1h_2D_sel_cropped_blurred_x8_clustered_kmeans"
    dem_directory = "/Users/fquareng/data/dem_squares"

    print("Collecting data...")
    elevation_values, temperature_values, humidity_values, pressure_values = collect_data(input_dir=input_directory, dem_dir=dem_directory)

    # Apply KMeans clustering with elbow method to determine optimal clusters
    print("Applying KMeans clustering...")
    kmeans_model, num_clusters = apply_kmeans_clustering(elevation_values, temperature_values, humidity_values, pressure_values, max_clusters=10)

    # Process the NetCDF files and classify them into the KMeans clusters
    process_netcdf_files_parallel(input_directory, dem_directory, output_directory, kmeans_model, num_clusters)