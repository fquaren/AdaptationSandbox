import os
import dask
import xarray as xr
from dask.distributed import Client, as_completed, LocalCluster
from dask.diagnostics import ProgressBar
import logging
import glob
from scipy.ndimage import gaussian_filter

def main():

    downsampling_factor = 8

    print(f"INFO: Downsampling with a factor of {downsampling_factor}.")

    # Set up the logger to write to a log file
    logging.basicConfig(
        filename="process_log.log",  # Specify the log file name
        level=logging.INFO,  # Set the logging level to INFO
        format="%(asctime)s - %(levelname)s - %(message)s"  # Log format with timestamp and level
    )

    # Set up Dask's distributed client with an optimized local cluster
    cluster = LocalCluster(
        n_workers=12,  # Set number of workers equal to the number of CPU cores
        threads_per_worker=1,  # One thread per worker to avoid oversubscription
        memory_limit="2GB",  # Limit memory per worker to prevent excessive memory use
        dashboard_address=":8787",  # Optional: Expose Dask Dashboard for monitoring
    )
    client = Client(cluster)  # Attach the Dask client to the local cluster

    logging.info(f"Dask client connected to cluster with {cluster.workers} workers.")

    # Define the input and output directories
    input_dir = "/Users/fquareng/data/1h_2D_sel_cropped"  # Replace with your input directory path
    output_dir = f"/Users/fquareng/data/1h_2D_sel_cropped_blurred_x{downsampling_factor}"  # Replace with your output directory path
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all NetCDF files in the input directory using glob
    input_files = glob.glob(os.path.join(input_dir, "*.nz"))  # Adjust file extension if necessary

    # Log the number of files found
    logging.info(f"Found {len(input_files)} files to process.")

    # List of variables to select from the NetCDF files
    selected_vars = ["RELHUM_2M", "T_2M", "PS"]

    # Function to process a single file (for parallel processing)
    def process_file(file):
        try:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file.replace(".nz", f"_blurred_x{downsampling_factor}.nz"))

            # Log the start of processing a file
            logging.info(f"Starting to process file: {file}")

            # Open the NetCDF file with Dask chunking
            ds = xr.open_dataset(input_path) #, chunks={"time": 1, "lat": 100, "lon": 100})  # Adjust chunking as needed
            # print(ds.dims)  # Check dataset dimensions
            # print(ds.chunks)
            # # Ensure chunk consistency using unify_chunks
            # ds = ds.unify_chunks()
            # # Now ds should have consistent chunking
            # print(ds.chunks)

            # Select the desired variables
            ds_selected = ds[selected_vars]

            # Apply Gaussian blur or any other operations (example using gaussian_filter)
            def apply_gaussian_filter(data, sigma=1.0):
                """
                Apply a Gaussian filter to the data array.
                """
                return gaussian_filter(data, sigma=sigma, mode="nearest")

            # Function to apply the Gaussian filter to the dataset
            def apply_gaussian_to_dataset(ds_selected, sigma=1.0):
                """
                Apply the Gaussian filter to each variable in the dataset.
                """
                def apply_filter(block):
                    """
                    Apply Gaussian filter to each variable in the block and return
                    an xarray.DataArray with explicit dimension names.
                    """
                    result = {}
                    for var in ds_selected.data_vars:
                        # Apply Gaussian filter to each variable in the block
                        filtered_data = apply_gaussian_filter(block[var], sigma=sigma)
                        
                        # Explicitly define the dimension names when constructing the DataArray
                        result[var] = xr.DataArray(
                            filtered_data,
                            dims=block[var].dims,  # Ensure dimensions are passed correctly
                            coords=block[var].coords,  # Ensure coordinates are preserved
                            name=var
                        )
                    
                    return result

                # Use map_blocks to apply the function
                ds_blurred_dict = ds_selected.map_blocks(apply_filter)

                # Convert the dictionary back into an xarray Dataset
                ds_blurred = xr.Dataset(ds_blurred_dict)
                
                return ds_blurred
            
            ds_blurred = apply_gaussian_to_dataset(ds_selected, sigma=2.0)
            
            # # Apply Gaussian filter to each variable
            # ds_blurred = ds_selected.map_blocks(
            #     lambda block: {var: apply_gaussian_filter(block[var]) for var in ds_selected.data_vars},
            #     dtype=ds_selected.dtype
            # )

            # Downsample the data based on the downsampling factor
            downsampled = ds_blurred.isel(
                rlat=slice(0, None, downsampling_factor),  # Adjust downsampling factor
                rlon=slice(0, None, downsampling_factor)
            )

            # Write the processed data to a new NetCDF file
            downsampled.to_netcdf(output_path)

            # Log the successful processing of the file
            logging.info(f"Successfully processed: {file}")

            return f"Processed: {file}"

        except Exception as e:
            # Log any errors encountered during processing
            logging.error(f"Error processing {file}: {e}")
            return f"Error processing {file}: {e}"

    # Create a list of Dask tasks
    tasks = [dask.delayed(process_file)(os.path.basename(file)) for file in input_files]

    # Run the tasks concurrently with Dask's distributed scheduler
    with ProgressBar():
        futures = client.compute(tasks)  # Schedule tasks
        for future in as_completed(futures):  # Wait for tasks to complete
            result = future.result()
            logging.info(result)  # Log the task result (success or error)
            print(result)  # Output progress to the console

    # Close the Dask client after processing
    client.close()

    # Log completion
    logging.info("Batch processing completed.")

if __name__ == '__main__':
    # Adjust the downsampling factor here, e.g., downsampling_factor=5
    main()