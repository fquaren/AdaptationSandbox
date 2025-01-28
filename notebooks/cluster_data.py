import os
import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed

# Paths
data_dir = "/Users/fquareng/data/1h_2D_sel_cropped"
output_dir = "/Users/fquareng/data/1h_2D_sel_cropped_clustered"
csv_file = "/Users/fquareng/data/domains_clustering.csv"

# Read the CSV file
df = pd.read_csv(csv_file)

# Function to find indices based on coordinates
def find_indices(coord, grid):
    return (np.abs(grid - coord)).argmin()

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
    print(f"Processing file: {file}")

    try:
        # Open the .nz file using xarray
        with xr.open_dataset(file_path, engine="netcdf4") as ds:
            rlon = ds.rlon.values
            rlat = ds.rlat.values
            automatic_grid_size = (int(len(rlon)/12), int(len(rlat)/12))
            print(f"Automatic grid size: {automatic_grid_size}")

            for _, row in df.iterrows():
                # Find grid indices based on bottom-left coordinates
                x_idx = find_indices(row["Bottom_Left_X"], rlon)
                y_idx = find_indices(row["Bottom_Left_Y"], rlat)

                # Crop the data
                cropped_data = crop_data(ds, x_idx, y_idx, grid_size=automatic_grid_size)

                # Create the cluster label directory if it doesn't exist
                cluster_label_dir = os.path.join(output_dir, str(row["Cluster Label"]))
                os.makedirs(cluster_label_dir, exist_ok=True)

                # Save the cropped data
                output_file = os.path.join(cluster_label_dir, f"{row['Bottom_Left_X']}_{row['Bottom_Left_Y']}_{file}")
                cropped_data.to_netcdf(output_file)
                print(f"Saved cropped data to {output_file}")

    except Exception as e:
        print(f"Error processing file {file}: {e}")

# Main function to process files concurrently
def main():
    # List all .nz files in the data directory
    files = [f for f in os.listdir(data_dir) if f.endswith('.nz')]

    # Create a ProcessPoolExecutor to process files concurrently
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file, df, output_dir, data_dir): file for file in files}
        
        # Wait for all tasks to complete
        for future in as_completed(futures):
            file = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing file {file}: {e}")

if __name__ == "__main__":
    main()