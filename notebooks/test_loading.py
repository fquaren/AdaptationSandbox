import time
from torch.utils.data import Dataset
import numpy as np
import torch
import xarray as xr
import os
import re

def benchmark_dataset_loading(dataset, n_samples=100):
    start = time.time()
    for i in range(n_samples):
        _ = dataset[i % len(dataset)]  # loop over indices
    end = time.time()
    avg_time = (end - start) / n_samples
    return avg_time


def get_file_splits(input_dir, target_dir, excluded_cluster):
    """
    Get file splits for training, validation, and testing datasets.
    Args:
        input_dir (str): Directory containing input files.
        target_dir (str): Directory containing target files.
        excluded_cluster (str): Cluster to be excluded from training.
    Returns:
        dict: Dictionary containing file splits for training, validation, and testing datasets.
    """
    train_inputs, val_inputs = [], []
    train_targets, val_targets = [], []

    for cluster in sorted(["cluster_8"]):
        input_path = os.path.join(input_dir, cluster)
        target_path = os.path.join(target_dir, cluster)
        if not os.path.isdir(input_path) or not os.path.isdir(target_path):
            continue

        all_input_files = sorted([f for f in os.listdir(input_path) if f.endswith(".nz")])
        all_target_files = sorted([f for f in os.listdir(target_path) if f.endswith(".nz")])
        pattern = re.compile(r"(\d{1,2})_(\d{1,2})_lffd(\d{4})(\d{2})(\d{2})\d{6}")
        
        for input_file, target_file in zip(all_input_files, all_target_files):
            assert pattern.search(input_file).group(3,4,5) == pattern.search(target_file).group(3,4,5), "Input and target files must match."
            try:
                _, _, year, month = SingleVariableDataset_v2._extract_numbers(None, input_file)
            except ValueError:
                continue

            input_file_path = os.path.join(input_path, input_file)
            target_file_path = os.path.join(target_path, target_file)

            if cluster != excluded_cluster:
                if year == 2019 and month % 2 == 1:
                    train_inputs.append(input_file_path)
                    train_targets.append(target_file_path)
                if year == 2017 and month in [3, 6, 9, 12]:
                    val_inputs.append(input_file_path)
                    val_targets.append(target_file_path)

    file_splits = {
        "train": (train_inputs, train_targets),
        "val": (val_inputs, val_targets),
    }
    
    return file_splits


class SingleVariableDataset_v2(Dataset):
    def __init__(self, variable, input_files, target_files, elev_dir, transform=None):
        self.variable = variable
        self.input_files = input_files
        self.target_files = target_files
        self.elev_dir = elev_dir
        self.elev_files = self._map_elevation_files()
        self.transform = transform

    def _extract_numbers(self, filename):
        match = re.match(r"(\d{1,2})_(\d{1,2})_lffd(\d{4})(\d{2})", os.path.basename(filename))
        if match:
            A, B, year, month = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))
            return A, B, year, month
        raise ValueError(f"Filename {filename} does not match expected pattern A_B_lffdYYYYMM*.nc")
    
    def _extract_numbers_from_dem(self, filename):
        match = re.match(r"(\d{1,2})_(\d{1,2})_", os.path.basename(filename))
        if match:
            A, B = int(match.group(1)), int(match.group(2))
            return A, B
        raise ValueError(f"Filename {filename} does not match expected pattern A_B_lffdYYYYMM*.nc")
    
    def _map_elevation_files(self):
        """Creates a mapping of (A, B) -> elevation file path, ensuring exact matches."""
        elev_files = {}
        for file in os.listdir(self.elev_dir):
            if file.endswith(".nc"):
                try:
                    A, B = self._extract_numbers_from_dem(file)
                    elev_files[(A, B)] = os.path.join(self.elev_dir, file)
                except ValueError:
                    continue  # Ignore files that don't match the pattern
        return elev_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        target_file = self.target_files[idx]
        A, B, _, _ = self._extract_numbers(input_file)
        # filename = os.path.basename(input_file)

        # Ensure the correct elevation file exists
        if (A, B) not in self.elev_files:
            raise FileNotFoundError(f"No elevation file found for {A}_{B} in {self.elev_dir}")

        elev_file = self.elev_files[(A, B)]

        with xr.open_dataset(elev_file) as ds:
            elevation_data = torch.tensor(ds["HSURF"].values, dtype=torch.float32).unsqueeze(0)

        with xr.open_dataset(input_file) as ds:
            input_data = torch.tensor(ds[self.variable].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
        with xr.open_dataset(target_file) as ds:
            target_data = torch.tensor(ds[self.variable].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
        
        return input_data, elevation_data, target_data #, filename



# Define directories
input_dir = "/Users/fquareng/data/1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_threshold_12_blurred"
target_dir = "/Users/fquareng/data/1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_threshold_12"
elev_dir = "/Users/fquareng/data/dem_squares"
excluded_cluster = "cluster_0"  # Cluster to be excluded from training

# Set up file listss
file_splits = get_file_splits(input_dir, target_dir, excluded_cluster)
train_inputs, train_targets = file_splits["train"]

ds = xr.open_mfdataset(train_inputs, concat_dim="time", combine="nested", chunks={"time": 1, "rlat": 16, "rlon": 16})

input_zarr_path = os.path.join(input_dir, excluded_cluster, "train_inputs_cluster_0.zarr")
print(f"Saving to {input_zarr_path}...")
ds.to_zarr(input_zarr_path, mode="w", consolidated=True)

print("Reading from Zarr...")
ds_zarr = xr.open_zarr(input_zarr_path, consolidated=True)
print(ds_zarr)


# Create both datasets
ds_nc = SingleVariableDataset_v2("T_2M", *file_splits["train"], elev_dir)

# Run benchmark
n_samples = 1000
time_nc = benchmark_dataset_loading(ds_nc, n_samples=n_samples)

print(f"Average read time over {n_samples} samples:")
print(f"NetCDF: {time_nc:.4f} s/sample")
# print(f"Zarr:   {time_zarr:.4f} s/sample")