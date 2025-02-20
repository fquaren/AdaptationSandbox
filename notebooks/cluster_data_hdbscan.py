import os
import multiprocessing
import dask.array as da
import hdbscan
import hnswlib
import xarray as xr
from tqdm import tqdm
from sklearn.decomposition import PCA
from functools import partial


def process_subdir(subdir, input_dir, dem_dir):
    """Process a single subdirectory to collect elevation and climate data."""
    subdir_path = os.path.join(input_dir, subdir)
    dem_file = f"dem_{subdir}.nc"
    dem_path = os.path.join(dem_dir, dem_file)

    if not os.path.exists(dem_path):
        return None  # Skip if DEM file is missing

    elevation_ds = xr.open_dataset(dem_path)
    mean_elevation = elevation_ds['HSURF'].values.mean()

    # Process NetCDF files
    files = [os.path.join(subdir_path, f) for f in os.listdir(subdir_path) if f.endswith(".nz")]

    elevation_values, temperature_values, humidity_values, pressure_values = [], [], [], []

    for file in files:
        ds = xr.open_dataset(file)
        temp_values = da.from_array(ds['T_2M'].values.flatten(), chunks=10000)
        hum_values = da.from_array(ds['RELHUM_2M'].values.flatten(), chunks=10000)
        sp_values = da.from_array(ds['PS'].values.flatten(), chunks=10000)
        elev_values = da.full_like(temp_values, mean_elevation)

        temperature_values.append(temp_values)
        humidity_values.append(hum_values)
        pressure_values.append(sp_values)
        elevation_values.append(elev_values)

    if elevation_values:
        return (
            da.concatenate(elevation_values),
            da.concatenate(temperature_values),
            da.concatenate(humidity_values),
            da.concatenate(pressure_values)
        )
    else:
        return None  # No data collected


def collect_data(input_dir, dem_dir):
    """Parallelized data collection using multiprocessing."""
    subdirs = sorted([subdir for subdir in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, subdir))])

    process_func = partial(process_subdir, input_dir=input_dir, dem_dir=dem_dir)  # NEW: Use `partial`

    with multiprocessing.Pool(processes=8) as pool:
        results = list(tqdm(pool.imap(process_func, subdirs), total=len(subdirs), desc="Collecting Data"))

    # Filter out None results (subdirectories without data)
    results = [r for r in results if r is not None]

    # Concatenate results along each feature
    elevation_values = da.concatenate([r[0] for r in results])
    temperature_values = da.concatenate([r[1] for r in results])
    humidity_values = da.concatenate([r[2] for r in results])
    pressure_values = da.concatenate([r[3] for r in results])

    return elevation_values, temperature_values, humidity_values, pressure_values


def apply_hdbscan_clustering(elevation_values, temperature_values, humidity_values, pressure_values):
    """Apply optimized HDBSCAN clustering."""
    data = da.stack([elevation_values, temperature_values, humidity_values, pressure_values], axis=1)

    # Convert to NumPy (HDBSCAN does not support Dask)
    data_np = data.compute()

    # Dimensionality Reduction (PCA) - Reduce from 4D to 2D for faster clustering
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data_np)

    # HDBSCAN Clustering
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom')
    cluster_labels = hdbscan_model.fit_predict(reduced_data)

    # Train fast kNN with HNSW (Hierarchical Navigable Small World)
    dim = 2  # After PCA reduction
    knn_index = hnswlib.Index(space='l2', dim=dim)
    knn_index.init_index(max_elements=len(reduced_data), ef_construction=200, M=16)
    knn_index.add_items(reduced_data[cluster_labels != -1], cluster_labels[cluster_labels != -1])

    return hdbscan_model, knn_index, cluster_labels


if __name__ == "__main__":
    input_directory = "/Users/fquareng/data/1h_2D_sel_cropped_blurred_x8_gridded"
    output_directory = "/Users/fquareng/data/1h_2D_sel_cropped_blurred_x8_clustered_hdbscan"
    dem_directory = "/Users/fquareng/data/dem_squares"

    print("Collecting data...")
    elevation_values, temperature_values, humidity_values, pressure_values = collect_data(input_directory, dem_directory)

    print("Applying optimized HDBSCAN clustering...")
    hdbscan_model, knn_index, cluster_labels = apply_hdbscan_clustering(elevation_values, temperature_values, humidity_values, pressure_values)