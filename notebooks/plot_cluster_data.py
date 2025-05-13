import os
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from tqdm import tqdm
import itertools
import seaborn as sns
import pandas as pd


def compute_equivalent_potential_temperature(T, RH, P_surf):
    """Compute equivalent potential temperature θ_e (K) for GPU-accelerated tensors."""
    L_v = 2.5e6  # Latent heat of vaporization (J/kg)
    cp = 1005    # Specific heat of dry air (J/kg/K)
    epsilon = 0.622  # Ratio of molecular weights of water vapor to dry air

    # Compute saturation vapor pressure (hPa) using Tetens' formula
    e_s = 6.112 * np.exp((17.67 * T) / (T + 243.5))
    e = RH * e_s  # Actual vapor pressure (hPa)
    
    # Mixing ratio (kg/kg)
    w = epsilon * (e / (P_surf - e))
    
    # Compute potential temperature θ
    theta = T * (1000 / P_surf) ** 0.286
    
    # Compute equivalent potential temperature θ_e
    theta_e = theta * np.exp((L_v * w) / (cp * T))
    
    return theta_e


def collect_data_for_cluster(cluster_index, cluster_dir, dem_dir):
    """Collect temperature, humidity, pressure, and elevation data for all files in a cluster's directory."""
    temperature_values = []
    humidity_values = []
    pressure_values = []
    elevation_values = []
    
    # Loop through all NetCDF files in the cluster directory
    for file_name in tqdm(os.listdir(cluster_dir)):
        if file_name.endswith(".nz"):
            file_path = os.path.join(cluster_dir, file_name)
            
            # Open the NetCDF file and extract variables
            with xr.open_dataset(file_path) as ds:
                temperature = ds['T_2M'].values.mean()
                humidity = ds['RELHUM_2M'].values.mean()
                pressure = ds['PS'].values.mean()

                # Corresponding DEM file for elevation
                dem_file = f"{file_name.split('_')[0]}_{file_name.split('_')[1]}_dem.nc"
                dem_path = os.path.join(dem_dir, dem_file)

                # Read the DEM file to get elevation
                if os.path.exists(dem_path):
                    elevation_ds = xr.open_dataset(dem_path)
                    elevation = elevation_ds['HSURF'].values.mean()
                else:
                    elevation = None  # Skip if elevation data is missing

                if elevation is not None:
                    temperature_values.append(temperature)
                    humidity_values.append(humidity)
                    pressure_values.append(pressure)
                    elevation_values.append(elevation)

    return cluster_index, temperature_values, humidity_values, pressure_values, elevation_values


def collect_data_for_all_clusters(input_dir, dem_dir, num_clusters):
    """Parallelize the collection of data for all clusters."""
    cluster_dirs = [
        (i, os.path.join(input_dir, f'cluster_{i}'), dem_dir) for i in range(num_clusters)
    ]
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        results = list(tqdm(pool.starmap(collect_data_for_cluster, cluster_dirs), total=num_clusters))
    
    # Prepare the data in a dictionary format
    cluster_data = {i: {'T_2M': [], 'Theta_e':[], 'RELHUM_2M': [], 'PS': [], 'HSURF': []} for i in range(num_clusters)}
    
    for cluster_index, temp_values, humidity_values, pressure_values, elevation_values in results:
        cluster_data[cluster_index]['T_2M'] = temp_values
        theta_e = []
        for t, rh, p in zip(temp_values, humidity_values, pressure_values):
            theta_e.append(compute_equivalent_potential_temperature(t, rh, p))
        cluster_data[cluster_index]['Theta_e'] = theta_e
        cluster_data[cluster_index]['RELHUM_2M'] = humidity_values
        cluster_data[cluster_index]['PS'] = pressure_values
        cluster_data[cluster_index]['HSURF'] = elevation_values

    return cluster_data


def save_plot(fig, figures_dir, filename):
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(os.path.join(figures_dir, filename), dpi=300)
    plt.close(fig)


def plot_cluster_scatter_grids(variables, cluster_data, num_clusters, variable_labels, variable_ranges, figures_directory):
    """
    Plots scatter plots for all unique combinations of the 4 variables (PS, T_2M, HSURF, RELHUM_2M).
    Each cluster gets its own subplot within a grid layout.
    
    Parameters:
        cluster_data (dict): Dictionary containing data for each cluster.
        num_clusters (int): Number of clusters.
        variable_labels (dict): Labels for variables.
        variable_ranges (dict): Axis limits for variables.
        figures_directory (str): Directory where figures should be saved.
    """
    
    variable_combinations = list(itertools.combinations(variables, 2))  # Get unique variable pairs

    for x_var, y_var in variable_combinations:
        # Determine grid layout
        cols = int(np.ceil(np.sqrt(num_clusters)))  # Number of columns
        rows = int(np.ceil(num_clusters / cols))  # Number of rows

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Adjust figure size dynamically
        axes = axes.flatten()  # Flatten in case of multi-row layout

        for cluster in range(num_clusters):
            ax = axes[cluster]  # Get corresponding subplot
            sns.scatterplot(x=cluster_data[cluster][x_var], y=cluster_data[cluster][y_var],
                            ax=ax, alpha=0.7)
            
            ax.set_title(f"Cluster {cluster}")
            ax.set_xlabel(variable_labels[x_var])
            ax.set_ylabel(variable_labels[y_var])
            ax.set_xlim(variable_ranges[x_var])
            ax.set_ylim(variable_ranges[y_var])
            ax.grid(True)

        # Remove unused subplots if num_clusters is not a perfect square
        for i in range(num_clusters, len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(f"{variable_labels[x_var]} vs {variable_labels[y_var]} for Each Cluster", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title

        # Save the plot
        filename = f"{x_var.lower()}_vs_{y_var.lower()}_per_cluster.png"
        save_plot(fig, figures_directory, filename)
        plt.show()


def plot_variable_distributions(variables, cluster_data, num_clusters, variable_labels, variable_ranges, figures_directory):
    """
    Plots the distribution of multiple variables across clusters, including mean and standard deviation.

    Parameters:
        cluster_data (dict): Dictionary containing data for each cluster.
        num_clusters (int): Number of clusters.
        all_data (dict): Dictionary containing all values for each variable.
        variable_labels (dict): Labels for variables.
        variable_ranges (dict): Axis limits for variables.
        figures_directory (str): Directory where figures should be saved.
    """

    for var in variables:
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot distribution for each cluster
        for cluster in range(num_clusters):
            sns.histplot(cluster_data[cluster][var], kde=True, label=f"Cluster {cluster}", ax=ax)

        # Set titles and labels
        ax.set_title(f"{variable_labels[var]} Distribution")
        ax.set_xlabel(variable_labels[var])
        ax.set_ylabel("Frequency")
        ax.set_xlim(variable_ranges[var])
        ax.legend(title="Cluster")

        # Save the plot
        filename = f"{var.lower()}_distribution_all_clusters.png"
        save_plot(fig, figures_directory, filename)
        plt.show()
    

def plot_cluster_boxplots(variables, cluster_data, num_clusters, variable_labels, figures_directory):
    """
    Plots separate boxplots for each variable, grouped by cluster, using multiple y-axes (faceted subplots).
    """

    data = []
    for cluster in range(num_clusters):
        for var in variables:
            for value in cluster_data[cluster][var]:
                data.append({"Cluster": f"Cluster {cluster}", "Variable": var, "Value": value})

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(len(variables), 1, figsize=(10, 6 * len(variables)), sharex=True)

    if len(variables) == 1:
        axes = [axes]  # Ensure axes is iterable when there's only one variable

    for ax, var in zip(axes, variables):
        sns.boxplot(x="Cluster", y="Value", data=df[df["Variable"] == var], ax=ax)
        ax.set_title(f"Boxplot of {variable_labels[var]} by Cluster")
        ax.set_ylabel(variable_labels[var])
        ax.grid(True)

    axes[-1].set_xlabel("Cluster")  # Label x-axis only on the last subplot

    fig.tight_layout()
    
    filename = "boxplot_variables_per_cluster.png"
    save_plot(fig, figures_directory, filename)
    plt.show()


def plot_all(input_dir, dem_dir, figures_directory):
    
    num_clusters = len([f for f in os.listdir(input_dir) if f.startswith("cluster_")])
    print(f"Number of clusters found: {num_clusters}")
    print("Collecting data for all clusters...")
    cluster_data = collect_data_for_all_clusters(input_dir, dem_dir, num_clusters)

    # Determine global min/max values
    all_temperatures = sum([cluster_data[c]['T_2M'] for c in range(num_clusters)], [])
    all_theta_e = sum([cluster_data[c]['Theta_e'] for c in range(num_clusters)], [])
    all_humidities = sum([cluster_data[c]['RELHUM_2M'] for c in range(num_clusters)], [])
    all_pressures = sum([cluster_data[c]['PS'] for c in range(num_clusters)], [])
    all_elevations = sum([cluster_data[c]['HSURF'] for c in range(num_clusters)], [])

    variable_ranges = {
        "T_2M": (min(all_temperatures), max(all_temperatures)),
        "Theta_e": (min(all_theta_e), max(all_theta_e)),
        "RELHUM_2M": (min(all_humidities), max(all_humidities)),
        "PS": (min(all_pressures), max(all_pressures)),
        "HSURF": (min(all_elevations), max(all_elevations))
    }

    variable_labels = {
        "T_2M": "Temperature (K)",
        "Theta_e": "Equivalent potential temperature (K)",
        "RELHUM_2M": "Relative Humidity (%)",
        "PS": "Pressure (Pa)",
        "HSURF": "Elevation (m)"
    }

    variables = ["T_2M", "Theta_e", "RELHUM_2M", "PS", "HSURF"]

    # BOX PLOT =======================================================
    plot_cluster_boxplots(variables, cluster_data, num_clusters, variable_labels, figures_directory)

    # SCATTER PLOTS ==================================================
    plot_cluster_scatter_grids(variables, cluster_data, num_clusters, variable_labels, variable_ranges, figures_directory)
    
    # VARIABLE DISTRIBUTIONS =========================================
    plot_variable_distributions(variables, cluster_data, num_clusters, variable_labels, variable_ranges, figures_directory)


if __name__ == "__main__":
    cluster_method = "threshold" # "threshold", "kmeans", "hierarchical"
    input_directory = f"/Users/fquareng/data/1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_{cluster_method}_12"
    # input_directory = f"/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/DA/1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_{cluster_method}_12"
    dem_directory = "/Users/fquareng/data/dem_squares"
    # dem_directory = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/dem_squares"
    figures_directory = f"/Users/fquareng/phd/AdaptationSandbox/figures/new_dataset_{cluster_method}_12"
    
    print("Plotting cluster data for method:", cluster_method)
    
    plot_all(input_directory, dem_directory, figures_directory)