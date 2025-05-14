import os
import xarray as xr
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def convert_file(nc_file, out_dir, chunking=None, overwrite=False):
    zarr_name = os.path.splitext(os.path.basename(nc_file))[0] + ".zarr"
    zarr_path = os.path.join(out_dir, zarr_name)

    if os.path.exists(zarr_path) and not overwrite:
        return f"Skipped {zarr_path}"

    try:
        ds = xr.open_dataset(nc_file)

        if chunking:
            ds = ds.chunk(chunking)

        ds.to_zarr(zarr_path, mode="w")
        return f"Converted {nc_file} -> {zarr_path}"
    except Exception as e:
        return f"Failed {nc_file}: {str(e)}"

def batch_convert_netcdf_to_zarr(nc_dir, out_dir, chunking={"time": 1}, overwrite=False, n_workers=4):
    os.makedirs(out_dir, exist_ok=True)
    nc_files = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(".nz")]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(convert_file, f, out_dir, chunking, overwrite)
            for f in nc_files
        ]
        for result in tqdm(futures):
            print(result.result())

if __name__ == "__main__":
    # Example usage
    # input_root_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/DA/1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_threshold_12"
    input_root_dir = "/Users/fquareng/data/1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_threshold_12"
    cluster_dirs = [d for d in os.listdir(input_root_dir) if os.path.isdir(os.path.join(input_root_dir, d))]
    for cluster_dir in cluster_dirs:
        # Process each cluster directory
        print(f"Processing cluster: {cluster_dir}")
        
        # Define input and output directories
        input_dir = os.path.join(input_root_dir, cluster_dir)
        output_dir = os.path.join("/Users/fquareng/data/1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_threshold_12_blurred_zarr", cluster_dir)
    
        chunking = {"time": 1}  # Adjust based on your use case
        batch_convert_netcdf_to_zarr(input_dir, output_dir, chunking=chunking, overwrite=False, n_workers=4)

