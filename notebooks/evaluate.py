from tqdm import tqdm
import os
import glob
import torch
import torch.nn as nn
import xarray as xr
import numpy as np
import re
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

class UNet8x(nn.Module):
    def __init__(self):
        super(UNet8x, self).__init__()
        
        # Elevation Downsampling Block (to match variable resolution)
        self.downsample_elevation = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 2x downsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 2x downsampling (total 4x)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 2x downsampling (total 8x)
            nn.ReLU(inplace=True),
        )

        # Define encoding layers
        self.encoder1 = self.conv_block(65, 64)  # 32 (from elevation) + 1 (variable)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)
        
        # Define decoding layers
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(256 + 256, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(128 + 128, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(64 + 64, 64)
        
        # Additional upsampling layers for 8x resolution
        self.upconv_final1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 2x
        self.upconv_final2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 4x
        self.upconv_final3 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # 8x
        self.output = nn.Conv2d(64, 1, kernel_size=1)  # Final output layer
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, variable, elevation):  
        # Downsample elevation data to match variable resolution
        elevation_downsampled = self.downsample_elevation(elevation)

        # Check dimensions
        assert variable.shape[2:] == elevation_downsampled.shape[2:], \
            f"Selected variable and elevation dimensions do not match, {variable.shape[2:], elevation_downsampled.shape[2:]}"

        # Concatenate the two inputs
        x = torch.cat((variable, elevation_downsampled), dim=1)  
        
        # Encoder
        e1 = self.encoder1(x)  
        p1 = self.pool(e1)
        e2 = self.encoder2(p1)
        p2 = self.pool(e2)
        e3 = self.encoder3(p2)
        p3 = self.pool(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder
        d3 = self.upconv3(b)
        d3 = torch.cat((d3, e3), dim=1)  # Skip connection
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)  # Skip connection
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)  # Skip connection
        d1 = self.decoder1(d1)
        
        # Additional upsampling for 8x resolution
        d_final1 = self.upconv_final1(d1)
        d_final2 = self.upconv_final2(d_final1)
        d_final3 = self.upconv_final3(d_final2)
        return self.output(d_final3)
    


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


def get_dataloaders(variable, input_dir, target_dir, elev_file, batch_size=4):
    input_files = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".nz")])
    target_files = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".nz")])

    # Check input and target files contain the same number of samples
    assert len(input_files) == len(target_files), "Number of input and target files must match."

    (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets) = split_dataset(input_files, target_files)

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
    print(input_path)
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


def evaluate_and_plot_step_1(model, criterion, test_loaders, save_path, device="cuda", save=True):
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
    model.eval()
    
    criterion = criterion
    all_test_losses = []
    all_predictions, all_targets = [], []
    
    # Iterate through each cluster
    for cluster_name, test_loader in test_loaders.items():
        test_losses = []
        predictions, targets = [], []
        
        with torch.no_grad():
            for temperature, elevation, target in test_loader:
                temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
                output = model(temperature, elevation)
                loss = criterion(output, target).item()
                test_losses.append(loss)
                predictions.append(output.cpu().numpy())
                targets.append(target.cpu().numpy())
        
        test_losses = np.array(test_losses)
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Save test losses for the current cluster
        np.save(os.path.join(save_path, f"{cluster_name}_test_losses.npy"), test_losses)
        
        all_test_losses.extend(test_losses)  # Accumulate all cluster test losses
        all_predictions.extend(predictions)  # Accumulate all cluster predictions
        all_targets.extend(targets)  # Accumulate all cluster targets

        # Get top 5 and bottom 5 examples based on loss
        top_5_idx = test_losses.argsort()[-5:][::-1]  # Highest loss
        bottom_5_idx = test_losses.argsort()[:5]      # Lowest loss
        
        # Plot results for the current cluster
        _, axes = plt.subplots(5, 2, figsize=(10, 15))
        plt.suptitle(f"Mean Test Loss for {cluster_name}: {test_losses.mean():.4f}")
        
        # Plot results
        _, axes = plt.subplots(5, 2, figsize=(10, 15))
        plt.suptitle("Mean Test Loss: {:.4f}".format(test_losses.mean()))
        for i, idx in enumerate(top_5_idx):
            axes[i, 0].imshow(predictions[idx][0], cmap='coolwarm')
            axes[i, 0].set_title(f"Top {i+1} - Prediction (Loss: {test_losses[idx]:.4f})")
            axes[i, 1].imshow(targets[idx][0], cmap='coolwarm')
            axes[i, 1].set_title(f"Top {i+1} - Target")
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_path, f"evaluation_results_best_{cluster_name}.png"))
        
        _, axes = plt.subplots(5, 2, figsize=(10, 15))
        plt.suptitle("Mean Test Loss: {:.4f}".format(test_losses.mean()))
        for i, idx in enumerate(bottom_5_idx):
            axes[i, 0].imshow(predictions[idx][0], cmap='coolwarm')
            axes[i, 0].set_title(f"Bottom {i+1} - Prediction (Loss: {test_losses[idx]:.4f})")
            axes[i, 1].imshow(targets[idx][0], cmap='coolwarm')
            axes[i, 1].set_title(f"Bottom {i+1} - Target")
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_path, f"evaluation_results_worst_{cluster_name}.png"))

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

exp_path = "/Users/fquareng/experiments/x50d/"

model_path = "/Users/fquareng/experiments/x50d/best_model.pth"
model = UNet8x()

device = "gpu"
if device == "gpu" and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS is available!")
else:
    device = torch.device("cpu")
    print("Neither NVIDIA nor MPS not available, using CPU.")

model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device)

_, _, test_loaders = get_cluster_dataloaders(
    variable="T_2M",
    input_path=os.path.join(data_path, "1h_2D_sel_cropped_gridded_clustered_threshold_blurred"),
    target_path=os.path.join(data_path, "1h_2D_sel_cropped_gridded_clustered_threshold"),
    dem_dir=os.path.join(data_path, dem_path),
    batch_size=4
)

evaluate_and_plot_step_1(
    model=model,
    criterion=nn.MSELoss(),
    test_loaders=test_loaders,
    save_path=exp_path,
    device=device,
    save=True
)