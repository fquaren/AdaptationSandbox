import os
import dask
import xarray as xr
from dask.distributed import Client, as_completed, LocalCluster
from dask.diagnostics import ProgressBar
import logging

def main():
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

    # Define the input and output directories
    input_dir = "/Users/fquareng/data/1h_2D/"
    output_dir = "/Users/fquareng/data/1h_2D_sel"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all NetCDF files in the input directory
    input_files = [f for f in os.listdir(input_dir) if f.endswith(".nz")]

    # Log the number of files found
    logging.info(f"Found {len(input_files)} files to process.")

    # Function to process a single file (for parallel processing)
    def process_file(file):
        try:
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)

            # Log the start of processing a file
            logging.info(f"Starting to process file: {file}")

            # Open the NetCDF file with Dask chunking
            ds = xr.open_dataset(input_path, chunks={"time": 1, "lat": 256, "lon": 256})  # Adjust chunking to balance memory usage

            # Select the desired variables
            selected_vars = ["RELHUM_2M", "T_2M", "PS"]
            ds_selected = ds[selected_vars]

            # Write the processed data to a new NetCDF file
            ds_selected.to_netcdf(output_path)

            # Log the successful processing of the file
            logging.info(f"Successfully processed: {file}")

            return f"Processed: {file}"

        except Exception as e:
            # Log any errors encountered during processing
            logging.error(f"Error processing {file}: {e}")
            return f"Error processing {file}: {e}"

    # Create a list of Dask tasks
    tasks = [dask.delayed(process_file)(file) for file in input_files]

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
    main()