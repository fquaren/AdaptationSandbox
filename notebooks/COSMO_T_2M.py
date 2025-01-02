import os
import xarray as xr
import dask.array as da
from dask.distributed import Client
from dask.diagnostics import ProgressBar

def process_files(input_directory, output_file, variable_to_extract):
    # Initialize Dask Client for parallel processing
    client = Client(processes=True, threads_per_worker=2, n_workers=6, memory_limit="4GB")
    print(client)

    # List to store dask arrays and coordinates
    arrays = []
    time_coords = []  # To store the time coordinates for later
    lat_coords = None  # To store the latitude coordinates (from the first file)
    lon_coords = None  # To store the longitude coordinates (from the first file)

    # Loop through NetCDF files
    for filename in os.listdir(input_directory):
        if filename.endswith(".nz"):  # Only process NetCDF files
            file_path = os.path.join(input_directory, filename)
            print(f"Processing: {file_path}")

            # Open the dataset lazily using optimized chunk sizes
            ds = xr.open_dataset(file_path, chunks={"time": 1000})  # Larger chunks for efficiency

            # Extract the variable and store the dask array
            if variable_to_extract in ds.variables:
                arrays.append(ds[variable_to_extract])  # Append Dask array
                time_coords.append(ds["time"])  # Store the time coordinate for concatenation
                
                # Store lat and lon coordinates (assuming they are the same across all files)
                if lat_coords is None or lon_coords is None:
                    lat_coords = ds["rlat"]  # Get the latitudes from the first file
                    lon_coords = ds["rlon"]  # Get the longitudes from the first file
            else:
                print(f"Variable {variable_to_extract} not found in {filename}")

    # Concatenate datasets along 'time' dimension using Dask
    if arrays:
        # Dask array concatenation
        combined_array = da.concatenate(arrays, axis=0)  # Concatenate along time dimension (axis 0)

        # Concatenate time coordinates
        combined_time = da.concatenate(time_coords, axis=0)

        # Create an xarray DataArray from the combined Dask array
        combined_data = xr.DataArray(combined_array, 
                                    dims=["time", *arrays[0].dims[1:]], 
                                    coords={"time": combined_time, "lat": lat_coords, "lon": lon_coords},
                                    name="T_2M")
        
        # Fill missing values if any
        combined_data = combined_data.fillna(float('nan'))

        # Save to a new NetCDF file with progress bar
        with ProgressBar():
            combined_data.to_netcdf(output_file, engine="netcdf4", compute=True)

        print(f"Saved concatenated data to {output_file}")
    else:
        print("No data extracted. Ensure the variable name is correct or check the input files.")


if __name__ == '__main__':
    # Define paths and variable name
    input_directory = "/Users/fquareng/data/1h_2D/"
    output_file = "/Users/fquareng/data/T_2M.nc"
    variable_to_extract = "T_2M"

    process_files(input_directory, output_file, variable_to_extract)