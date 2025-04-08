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


def compute_equivalent_potential_temperature(T, RH, P_surf):
    """Compute equivalent potential temperature θ_e (K) for GPU-accelerated tensors."""
    L_v = 2.5e6  # Latent heat of vaporization (J/kg)
    cp = 1005    # Specific heat of dry air (J/kg/K)
    epsilon = 0.622  # Ratio of molecular weights of water vapor to dry air

    # Compute saturation vapor pressure (hPa) using Tetens' formula
    e_s = 6.112 * torch.exp((17.67 * T) / (T + 243.5))
    e = RH * e_s  # Actual vapor pressure (hPa)
    
    # Mixing ratio (kg/kg)
    w = epsilon * (e / (P_surf - e))
    
    # Compute potential temperature θ
    theta = T * (1000 / P_surf) ** 0.286
    
    # Compute equivalent potential temperature θ_e
    theta_e = theta * torch.exp((L_v * w) / (cp * T))
    
    return theta_e

class NormalizeTransform:
    def __call__(self, temp, elev):
        temp = (temp - temp.mean()) / temp.std()
        elev = (elev - elev.mean()) / elev.std()
        return temp, elev

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

        # Ensure the correct elevation file exists
        if (A, B) not in self.elev_files:
            raise FileNotFoundError(f"No elevation file found for {A}_{B} in {self.elev_dir}")

        elev_file = self.elev_files[(A, B)]

        with xr.open_dataset(input_file) as ds:
            input_data = torch.tensor(ds[self.variable].isel(time=0).values, dtype=torch.float32).unsqueeze(0)

        with xr.open_dataset(target_file) as ds:
            target_data = torch.tensor(ds[self.variable].isel(time=0).values, dtype=torch.float32).unsqueeze(0)

        with xr.open_dataset(elev_file) as ds:
            elevation_data = torch.tensor(ds["HSURF"].values, dtype=torch.float32).unsqueeze(0)

        if self.transform == "normalize":
            transform = NormalizeTransform()
            # Normalize the data
            input_data, elevation_data = transform(input_data, elevation_data)
            # Ensure the target data is normalized
            target_data = (target_data - target_data.mean()) / target_data.std()
        if self.transform == "theta_e":
            # Load the necessary data
            with xr.open_dataset(input_file) as ds:
                input_RH = torch.tensor(ds["RELHUM_2M"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
                input_P_surf = torch.tensor(ds["PS"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
            # Apply transfromation
            input_data = compute_equivalent_potential_temperature(input_data, input_RH, input_P_surf)
            with xr.open_dataset(target_file) as ds:
                target_RH = torch.tensor(ds["RELHUM_2M"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
                target_P_surf = torch.tensor(ds["PS"].isel(time=0).values, dtype=torch.float32).unsqueeze(0)
            # Apply transfromation
            target_data = compute_equivalent_potential_temperature(target_data, target_RH, target_P_surf)
        
        return input_data, elevation_data, target_data


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
    train_inputs, val_inputs, test_inputs = [], [], []
    train_targets, val_targets, test_targets = [], [], []

    for cluster in sorted(os.listdir(input_dir)):
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

            if cluster == excluded_cluster:
                if year == 2017 and month in [3, 6, 9, 12]:
                    val_inputs.append(input_file_path)
                    val_targets.append(target_file_path)
            elif year == 2019 and month % 2 == 1:
                train_inputs.append(input_file_path)
                train_targets.append(target_file_path)
            elif year == 2015 and month % 2 == 1:
                test_inputs.append(input_file_path)
                test_targets.append(target_file_path)

    return {
        "train": (train_inputs, train_targets),
        "val": (val_inputs, val_targets),
        "test": (test_inputs, test_targets),
    }


def get_dataloaders(input_dir, target_dir, elev_dir, variable, batch_size=8, num_workers=1, transform=None):
    """
    Create dataloaders for training, validation, and testing datasets.
    Args:
        input_dir (str): Directory containing input files.
        target_dir (str): Directory containing target files.
        elev_dir (str): Directory containing elevation files.
        variable (str): Variable name to load from the dataset.
        batch_size (int): Batch size for dataloaders.
        num_workers (int): Number of workers for dataloaders.
        transform: Transformations to apply to the data.
    Returns:
        dict: Dictionary containing dataloaders for each cluster.
    """
    cluster_names = sorted([c for c in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, c))])
    dataloaders = {}

    for excluded_cluster in cluster_names:
        print(f"Excluding cluster: {excluded_cluster}")

        file_splits = get_file_splits(input_dir, target_dir, excluded_cluster)

        train_dataset = SingleVariableDataset_v2(variable, *file_splits["train"], elev_dir, transform)
        val_dataset = SingleVariableDataset_v2(variable, *file_splits["val"], elev_dir, transform)
        test_dataset = SingleVariableDataset_v2(variable, *file_splits["test"], elev_dir, transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

        dataloaders[excluded_cluster] = {"train": train_loader, "val": val_loader, "test": test_loader}

    return dataloaders


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
        
        all_test_losses.extend(test_losses)  # Accumulate all cluster test losses

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
input_path = os.path.join(data_path, "1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_threshold_12_blurred")
target_path = os.path.join(data_path, "1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_threshold_12")

parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default=None, help="Path of model to evaluate")
parser.add_argument("--method", type=str, default=None, help="Interpolation model")
args = parser.parse_args()
exp_path = args.exp_name #os.path.join("/Users/fquareng/experiments/", args.exp_name)
# os.makedirs(exp_path, exist_ok=True)

dataloaders = get_dataloaders(
    variable="T_2M",
    input_dir=input_path,
    target_dir=target_path,
    elev_dir=os.path.join(data_path, dem_path),
    batch_size=1
)
test_loaders = {k: v["test"] for k, v in dataloaders.items()}

mean_test_loss = evaluate_and_plot_interpolation(
    model=args.method,
    criterion=nn.MSELoss(),
    test_loaders=test_loaders,
    save_path=exp_path,
    save=True,
)