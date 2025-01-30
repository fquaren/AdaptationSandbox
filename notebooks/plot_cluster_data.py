import os
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm

def collect_data_for_cluster(cluster_dir, dem_dir):
    """Collect temperature, humidity, and elevation data for all files in a cluster's directory."""
    temperature_values = []
    humidity_values = []
    elevation_values = []
    
    # Loop through all NetCDF files in the cluster directory
    for file_name in tqdm(os.listdir(cluster_dir)):
        if file_name.endswith(".nz"):
            file_path = os.path.join(cluster_dir, file_name)
            
            # Open the NetCDF file and extract temperature and humidity
            with xr.open_dataset(file_path) as ds:
                temperature = ds['T_2M'].values.mean()
                humidity = ds['RELHUM_2M'].values.mean()
                
                # Corresponding DEM file for elevation
                dem_file = f"dem_{file_name.split('_')[0]}_{file_name.split('_')[1]}.nc"
                dem_path = os.path.join(dem_dir, dem_file)

                # Read the DEM file to get elevation
                if os.path.exists(dem_path):
                    elevation_ds = xr.open_dataset(dem_path)
                    elevation = elevation_ds['HSURF'].values.mean()
                else:
                    print(f"Warning: DEM file {dem_path} not found, skipping elevation.")
                    elevation = None

                if elevation is not None:
                    temperature_values.append(temperature)
                    humidity_values.append(humidity)
                    elevation_values.append(elevation)

    return temperature_values, humidity_values, elevation_values

def collect_data_for_all_clusters(input_dir, dem_dir, num_clusters):
    """Parallelize the collection of data for all clusters."""
    # Create a list of arguments for each cluster
    cluster_dirs = [
        (os.path.join(input_dir, f'cluster_{i}'), dem_dir) for i in range(num_clusters)
    ]
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        # Use tqdm to display progress
        result = list(tqdm(pool.starmap(collect_data_for_cluster, cluster_dirs), total=num_clusters))
    
    # Prepare the data in a dictionary format
    cluster_data = {}
    for cluster, (temp_values, humidity_values, elevation_values) in enumerate(result):
        cluster_data[cluster] = {
            'T_2M': temp_values,
            'RELHUM_2M': humidity_values,
            'HSURF': elevation_values
        }

    return cluster_data


def save_plot(fig, figures_dir, filename):
    """Save the generated plot to the specified directory."""
    # Create the figures directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)
    
    # Save the figure to the figures directory
    fig.savefig(os.path.join(figures_dir, filename), dpi=300)
    plt.close(fig)  # Close the figure to avoid it being displayed after saving


def plot_scatter(input_dir, dem_dir, figures_directory, num_clusters=7):
    """Plot scatter plots of temperature vs. relative humidity, relative humidity vs. elevation, and temperature vs. elevation for each cluster."""
    # Collect data for each cluster in parallel
    print("Collecting data for all clusters...")
    cluster_data = collect_data_for_all_clusters(input_dir, dem_dir, num_clusters)

    # Plot Temperature vs. Relative Humidity
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in range(num_clusters):
        sns.scatterplot(x=cluster_data[cluster]['T_2M'], y=cluster_data[cluster]['RELHUM_2M'], label=f"Cluster {cluster}", ax=ax, alpha=0.5)
    ax.set_title("Temperature vs. Relative Humidity by Cluster")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Relative Humidity (%)")
    ax.legend(title="Cluster")
    ax.grid(True)
    save_plot(fig, figures_directory, "temperature_vs_humidity.png")

    # Plot Relative Humidity vs. Elevation
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in range(num_clusters):
        sns.scatterplot(x=cluster_data[cluster]['RELHUM_2M'], y=cluster_data[cluster]['HSURF'], label=f"Cluster {cluster}", ax=ax, alpha=0.5)
    ax.set_title("Relative Humidity vs. Elevation by Cluster")
    ax.set_xlabel("Relative Humidity (%)")
    ax.set_ylabel("Elevation (m)")
    ax.legend(title="Cluster")
    ax.grid(True)
    save_plot(fig, figures_directory, "humidity_vs_elevation.png")

    # Plot Temperature vs. Elevation
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster in range(num_clusters):
        sns.scatterplot(x=cluster_data[cluster]['T_2M'], y=cluster_data[cluster]['HSURF'], label=f"Cluster {cluster}", ax=ax, alpha=0.5)
    ax.set_title("Temperature vs. Elevation by Cluster")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Elevation (m)")
    ax.legend(title="Cluster")
    ax.grid(True)
    save_plot(fig, figures_directory, "temperature_vs_elevation.png")


if __name__ == "__main__":
    # input_directory = "/Users/fquareng/data/1h_2D_sel_cropped_blurred_x8_clustered_threshold"
    input_directory = "/Users/fquareng/data/1h_2D_sel_cropped_blurred_x8_clustered_kmeans"
    dem_directory = "/Users/fquareng/data/dem_squares"
    # figures_directory = "/Users/fquareng/phd/AdaptationSandbox/figures/threshold"
    figures_directory = "/Users/fquareng/phd/AdaptationSandbox/figures/kmeans"
    plot_scatter(input_directory, dem_directory, figures_directory)