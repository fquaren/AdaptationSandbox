"""
This script processes a set of .nz files containing temperature data and crops the data into smaller
grids based on coordinates specified in a CSV file. For each entry in the CSV (representing the bottom-left
coordinates of a grid), the script crops the corresponding data square from the .nz file and saves it into a new
file, organizing the output into directories based on cluster labels. The processing is done concurrently using a
ProcessPoolExecutor to speed up the operation, especially for large datasets.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed

# Path to the directory containing the .nz files and the CSV
data_dir = "/Users/fquareng/data/1h_2D_sel_cropped_blurred_x8"
output_dir = "/Users/fquareng/data/1h_2D_sel_cropped_blurred_x8_clustered"
csv_file = "/Users/fquareng/data/domains_clustering.csv"

# Read the CSV file
df = pd.read_csv(csv_file)

# Extract Bottom_Left_X, Bottom_Left_Y, and Cluster Label
bottom_left_x = df["Bottom_Left_X"].values
bottom_left_y = df["Bottom_Left_Y"].values
cluster_labels = df["Cluster Label"].values

# Function to crop data from the xarray dataset
def crop_data(dataset, x_idx, y_idx, grid_size):
    x_min = x_idx
    x_max = x_idx + grid_size[0]
    y_min = y_idx
    y_max = y_idx + grid_size[1]
    
    # Crop the data in the given x and y range
    cropped_data = dataset.isel(rlon=slice(x_min, x_max), rlat=slice(y_min, y_max))
    return cropped_data

# Function to process each file and save the cropped data
def process_file(file, df, output_dir, data_dir):
    file_path = os.path.join(data_dir, file)
    print("Reading file:", file)

    # Open the .nz file using xarray
    ds = xr.open_dataset(file_path, engine="netcdf4")
    
    # Get the spatial coordinates and temperature data from the dataset
    rlon = ds.rlon.values
    rlat = ds.rlat.values
    automatic_grid_size = (int(len(rlon)/12), int(len(rlat)/12))
    
    # Loop through each entry in the CSV and process
    for i, row in df.iterrows():
        x_idx = automatic_grid_size[0] * row["Bottom_Left_X"]
        y_idx = automatic_grid_size[1] * row["Bottom_Left_Y"]
        
        # Crop the data based on the indices
        cropped_data = crop_data(ds, x_idx, y_idx, grid_size=automatic_grid_size)

        # Create the cluster label directory if it doesn't exist
        cluster_label_dir = os.path.join(output_dir, str(row["Cluster Label"]))
        os.makedirs(cluster_label_dir, exist_ok=True)

        # Save the cropped data to a new file
        output_file = os.path.join(cluster_label_dir, f"{row['Bottom_Left_X']}_{row['Bottom_Left_Y']}_{file}")
        cropped_data.to_netcdf(output_file)

        print(f"Saved cropped data for cluster {row['Cluster Label']} from file {file} at coordinates ({row['Bottom_Left_X']}, {row['Bottom_Left_Y']})")

# Main function to process files concurrently
def main():
    # List all .nz files in the data directory
    files = [f for f in os.listdir(data_dir) if f.endswith('.nz')]

    # Create a ProcessPoolExecutor to process files concurrently
    with ProcessPoolExecutor() as executor:
        futures = []
        
        # Submit the task of processing each file
        for file in files:
            futures.append(executor.submit(process_file, file, df, output_dir, data_dir))

        # Wait for all tasks to complete
        for future in as_completed(futures):
            pass  # Optionally handle exceptions here

if __name__ == "__main__":
    main()