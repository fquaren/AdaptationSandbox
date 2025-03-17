import numpy as np
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import xarray as xr
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse


class SingleVariableDataset(Dataset):
    def __init__(self, variable, input_files, elev_dir, target_files):
        """
        Args:
            input_files (list): Paths to low-resolution variable (temperature) NetCDF files.
            elev_dir (str): Directory containing elevation NetCDF files named as "A_B_something.nc".
            target_files (list): Paths to corresponding high-resolution target NetCDF files.
        """
        self.variable = variable
        self.input_files = input_files
        self.target_files = target_files
        self.elev_dir = elev_dir

        # Preload the mapping of elevation files for faster access
        self.elev_files = self._map_elevation_files()

    def _extract_numbers(self, filename):
        """Extracts A and B from a filename like '3_6_lffd20101208160000.nz'."""
        match = re.match(r"(\d{1,2})_(\d{1,2})_", os.path.basename(filename))  # Matches 0-11 correctly
        if match:
            A, B = int(match.group(1)), int(match.group(2))
            if 0 <= A <= 11 and 0 <= B <= 11:
                return A, B
        raise ValueError(f"Filename {filename} does not match expected pattern A_B_*.nz")

    def _map_elevation_files(self):
        """Creates a mapping of (A, B) -> elevation file path, ensuring exact matches."""
        elev_files = {}
        for file in os.listdir(self.elev_dir):
            if file.endswith(".nc"):
                try:
                    A, B = self._extract_numbers(file)
                    elev_files[(A, B)] = os.path.join(self.elev_dir, file)
                except ValueError:
                    continue  # Ignore files that don't match the pattern
        return elev_files

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        """Loads variable input, elevation, and corresponding high-resolution target"""
        input_file = self.input_files[idx]
        target_file = self.target_files[idx]

        # Extract A and B from input filename
        A, B = self._extract_numbers(input_file)

        # Ensure the correct elevation file exists
        if (A, B) not in self.elev_files:
            raise FileNotFoundError(f"No elevation file found for {A}_{B} in {self.elev_dir}")

        elev_file = self.elev_files[(A, B)]

        # Load variable data
        with xr.open_dataset(input_file) as ds:
            input = torch.tensor(ds[self.variable].sel(time=ds[self.variable].time[0]).values, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # Load high-resolution target (same variable as input)
        with xr.open_dataset(target_file) as ds:
            target = torch.tensor(ds[self.variable].sel(time=ds[self.variable].time[0]).values, dtype=torch.float32).unsqueeze(0)  # [1, H_target, W_target]

        # Load the correct elevation data
        with xr.open_dataset(elev_file) as ds:
            elevation_data = torch.tensor(ds["HSURF"].values, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        return input, elevation_data, target


def split_dataset(input_files, target_files):
    """
    Splits input and target files into training, validation, and test sets.
    """
    train_inputs, val_inputs, test_inputs = [], [], []
    train_targets, val_targets, test_targets = [], [], []

    pattern = re.compile(r"(\d{1,2})_(\d{1,2})_lffd(\d{4})(\d{2})(\d{2})\d{6}")

    for input_file, target_file in zip(input_files, target_files):
        # Check input file and target file contain the same date
        assert pattern.search(input_file).group(3,4,5) == pattern.search(target_file).group(3,4,5), "Input and target files must match."

        pattern.search(input_file).group()
        match = pattern.search(input_file)
        if match:
            day = int(match.group(5))
            if day <= 21:
                train_inputs.append(input_file)
                train_targets.append(target_file)
            elif 22 <= day < 25:
                test_inputs.append(input_file)
                test_targets.append(target_file)
            elif day >= 25:
                val_inputs.append(input_file)
                val_targets.append(target_file)

    return (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets)


def split_dataset_3h(input_files, target_files):
    """
    Splits input and target files into training, validation, and test sets.
    Only selects files where the hour is a multiple of 3 (every 3 hours).
    
    Args:
        input_files (list of str): List of input file paths.
        target_files (list of str): List of corresponding target file paths.
    
    Returns:
        tuple: (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets)
    """
    train_inputs, val_inputs, test_inputs = [], [], []
    train_targets, val_targets, test_targets = [], [], []

    pattern = re.compile(r"(\d{1,2})_(\d{1,2})_lffd(\d{4})(\d{2})(\d{2})(\d{2})\d{4}")

    for input_file, target_file in zip(input_files, target_files):
        match_input = pattern.search(input_file)
        match_target = pattern.search(target_file)

        if match_input and match_target:
            # Extract day and hour
            day = int(match_input.group(5))   # Group 5 is the day (DD)
            hour = int(match_input.group(6))  # Group 6 is the hour (hh)

            # Ensure input and target files correspond to the same date
            assert match_input.group(3, 4, 5, 6) == match_target.group(3, 4, 5, 6), \
                "Input and target files must match."

            # Only select data points at 3-hour intervals
            if hour % 3 == 0:
                if day <= 21:
                    train_inputs.append(input_file)
                    train_targets.append(target_file)
                elif 22 <= day < 25:
                    test_inputs.append(input_file)
                    test_targets.append(target_file)
                elif day >= 25:
                    val_inputs.append(input_file)
                    val_targets.append(target_file)

    return (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets)


def get_dataloaders(variable, input_dir, target_dir, elev_file, batch_size=16):
    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".nz")])
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".nz")])

    # Check input and target files contain the same number of samples
    assert len(input_files) == len(target_files), "Number of input and target files must match."

    (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets) = split_dataset_3h(input_files, target_files)

    train_dataset = SingleVariableDataset(variable, train_inputs, elev_file, train_targets)
    val_dataset = SingleVariableDataset(variable, val_inputs, elev_file, val_targets)
    test_dataset = SingleVariableDataset(variable, test_inputs, elev_file, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_cluster_dataloaders(variable, input_path, target_path, dem_dir, batch_size=4):
    """
    Load train and validation DataLoaders for all clusters.

    Args:
        variable (str): Variable name for dataset.
        data_root (str): Root directory containing cluster subdirectories.
        batch_size (int): Batch size for DataLoader.

    Returns:
        train_loaders (dict): Dictionary of cluster train DataLoaders.
        val_loaders (dict): Dictionary of cluster validation DataLoaders.
        test_loaders (dict): Dictionary of cluster testing DataLoaders.
    """
    train_loaders, val_loaders, test_loaders = {}, {}, {}
    for cluster_name in os.listdir(input_path):
        input_dir = os.path.join(input_path, cluster_name)
        target_dir = os.path.join(target_path, cluster_name)
        # Ensure directories exist
        if not (os.path.isdir(input_dir) and os.path.isdir(target_dir)):
            continue

        # Load train and validation data
        train_loader, val_loader, test_loader = get_dataloaders(variable, input_dir, target_dir, dem_dir, batch_size)

        train_loaders[cluster_name] = train_loader
        val_loaders[cluster_name] = val_loader
        test_loaders[cluster_name] = test_loader

    return train_loaders, val_loaders, test_loaders


def upscale_array_spline(arr: np.ndarray, new_size=(128, 128), kx=1, ky=1) -> np.ndarray:
    """
    Upscales a 2D numpy array from 16x16 to 128x128 using `RectBivariateSpline`.

    Parameters:
    - arr (np.ndarray): Input 2D array of shape (16, 16).
    - new_size (tuple): Desired output size (128, 128).
    - kx, ky (int): Degrees of the spline in x and y directions (default=3 for cubic).

    Returns:
    - np.ndarray: Upscaled 2D array of shape (128, 128).
    """
    # Original grid points
    arr = arr.squeeze()
    x = np.arange(arr.shape[1])
    y = np.arange(arr.shape[0])

    # New grid points
    x_new = np.linspace(0, arr.shape[1] - 1, new_size[1])
    y_new = np.linspace(0, arr.shape[0] - 1, new_size[0])

    # Create spline interpolator
    interpolator = RectBivariateSpline(y, x, arr, kx=kx, ky=ky)

    # Evaluate the interpolator at new grid points
    return interpolator(y_new, x_new)


def evaluate_and_plot_interpolation(model, criterion, test_loaders, save_path, save=True):
    """
    Evaluates the model on the test datasets from multiple clusters, computes test loss, and plots results.
    
    Args:
        model: The PyTorch model to evaluate.
        config (dict): The configuration containing criterion and other settings.
        test_loaders (dict): A dictionary where keys are cluster names and values are DataLoader instances for testing.
        save_path (str): Directory to save the evaluation results.
        device (str): Device to run the evaluation on ('cpu', 'cuda', etc.).
        save (bool): Whether to save the plots and losses.
    
    Returns:
        float: The mean test loss across all clusters.
    """
    
    if model == "linear":
        kx, ky = 1, 1
    if model == "quadratic":
        kx, ky = 2, 2
    if model == "cubic":
        kx, ky = 3, 3
    
    criterion = criterion
    all_test_losses = []
    # all_predictions, all_targets = [], []
    
    # Iterate through each cluster
    for cluster_name, test_loader in test_loaders.items():
        print("Evaluating cluster", cluster_name)
        test_losses = []
        predictions, targets, elevations, inputs = [], [], [], []
        
        for temperature, elevation, target in tqdm(test_loader):
            temperature = temperature.numpy()
            output = upscale_array_spline(arr=temperature, kx=kx, ky=ky)
            output = torch.from_numpy(output).unsqueeze(0).unsqueeze(0)
            loss = criterion(output, target)
            test_losses.append(loss)
            predictions.append(output)
            targets.append(target)
            elevations.append(elevation)
            inputs.append(temperature)
        
        test_losses = np.array(test_losses)
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0) 
        elevations = np.concatenate(elevations, axis=0)
        inputs = np.concatenate(inputs, axis=0)

        # Save test losses and predictions for the current cluster
        np.save(os.path.join(save_path, f"{cluster_name}_test_losses.npy"), test_losses)
        # np.save(os.path.join(save_path, f"{cluster_name}_predictions.npy"), predictions) 
        # TODO save also filename
        
        all_test_losses.extend(test_losses)  # Accumulate all cluster test losses
        # all_predictions.extend(predictions)  # Accumulate all cluster predictions
        # all_targets.extend(targets)  # Accumulate all cluster targets

        # Get top 5 and bottom 5 examples based on loss
        top_5_idx = test_losses.argsort()[-5:][::-1]  # Highest loss
        bottom_5_idx = test_losses.argsort()[:5]      # Lowest loss
        
        # Plot results for the current cluster
        fig, axes = plt.subplots(5, 4, figsize=(10, 15))
        plt.suptitle(f"WORST 5 - Mean Test Loss for {cluster_name}: {test_losses.mean():.4f}")
        for i, idx in enumerate(top_5_idx):
            for j, (data, cmap, title) in enumerate(zip(
                [inputs[idx][0], predictions[idx][0], targets[idx][0], elevations[idx][0]], 
                ['coolwarm', 'coolwarm', 'coolwarm', 'viridis'], 
                [f"Input", f"Prediction (Loss: {test_losses[idx]:.4f})", 
                f"Target", f"Elevation"])):
                
                # Display image
                img = axes[i, j].imshow(data, cmap=cmap)
                axes[i, j].set_title(title)
                axes[i, j].axis("off")  # Hide axes
                
                # Add colorbar
                cbar = fig.colorbar(img, ax=axes[i, j], fraction=0.046, pad=0.04)  
                cbar.ax.tick_params(labelsize=8)  # Adjust tick size
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_path, f"evaluation_results_worst_{cluster_name}.png"))
        
        fig, axes = plt.subplots(5, 4, figsize=(10, 15))
        plt.suptitle(f"BEST 5 - Mean Test Loss for {cluster_name}: {test_losses.mean():.4f}")
        for i, idx in enumerate(bottom_5_idx):
            for j, (data, cmap, title) in enumerate(zip(
                [inputs[idx][0], predictions[idx][0], targets[idx][0], elevations[idx][0]], 
                ['coolwarm', 'coolwarm', 'coolwarm', 'viridis'], 
                [f"Input", f"Prediction (Loss: {test_losses[idx]:.4f})", 
                f"Target", f"Elevation"])):
                
                # Display image
                img = axes[i, j].imshow(data, cmap=cmap)
                axes[i, j].set_title(title)
                axes[i, j].axis("off")  # Hide axes
                
                # Add colorbar
                cbar = fig.colorbar(img, ax=axes[i, j], fraction=0.046, pad=0.04)  
                cbar.ax.tick_params(labelsize=8)  # Adjust tick size

        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_path, f"evaluation_results_best_{cluster_name}.png"))

    # Combine test losses across all clusters and compute the mean
    all_test_losses = np.array(all_test_losses)
    mean_test_loss = all_test_losses.mean()

    # Save the combined test losses
    np.save(os.path.join(save_path, "all_clusters_test_losses.npy"), all_test_losses)
    
    # Final summary
    print(f"Mean Test Loss across all clusters: {mean_test_loss:.4f}")
    
    return mean_test_loss


data_path = "/Users/fquareng/data/"
dem_path = "dem_squares"

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default=None, help="Path of model to evaluate")
parser.add_argument("--method", type=str, default=None, help="Interpolation model")
args = parser.parse_args()
exp_path = os.path.join("/Users/fquareng/experiments/", args.exp_name)
os.makedirs(exp_path, exist_ok=True)

_, _, test_loaders = get_cluster_dataloaders(
    variable="T_2M",
    input_path=os.path.join(data_path, "1h_2D_sel_cropped_gridded_clustered_threshold_blurred"),
    target_path=os.path.join(data_path, "1h_2D_sel_cropped_gridded_clustered_threshold"),
    dem_dir=os.path.join(data_path, dem_path),
    batch_size=1
)

mean_test_loss = evaluate_and_plot_interpolation(
    model=args.method,
    criterion=nn.MSELoss(),
    test_loaders=test_loaders,
    save_path=exp_path,
    save=True,
)