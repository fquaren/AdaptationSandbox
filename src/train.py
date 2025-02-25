import os
import re
import time
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

# ------------------ U-Net Model ------------------
class UNet8x(nn.Module):
    def __init__(self):
        super(UNet8x, self).__init__()
        
        # Elevation Downsampling Block (to match temperature resolution)
        self.downsample_elevation = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 2x downsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 2x downsampling (total 4x)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 2x downsampling (total 8x)
            nn.ReLU(inplace=True),
        )
        
        # # Elevation Upscaling Block (to match temperature resolution)
        # self.upscale_elevation = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        # Define encoding layers
        self.encoder1 = self.conv_block(65, 64)  # 32 (from elevation) + 1 (temperature)
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
    
    def forward(self, temperature, elevation):  
        # Downsample elevation data to match temperature resolution
        elevation_downsampled = self.downsample_elevation(elevation)
        
        # # Upscale elevation data to match temperature resolution
        # elevation_downsampled = self.upscale_elevation(elevation)

        # Check dimensions
        assert temperature.shape[2:] == elevation_downsampled.shape[2:], f"Temperature and elevation dimensions do not match, {temperature.shape[2:], elevation_downsampled.shape[2:]}"

        # Concatenate the two inputs
        x = torch.cat((temperature, elevation_downsampled), dim=1)  
        
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

# ------------------ Dataset Class ------------------
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


# ------------------ DataLoader Function ------------------
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

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4, device="cuda", save_path="best_model.pth", patience=10):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for temperature, elevation, target in train_loader:
            temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(temperature, elevation)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for temperature, elevation, target in val_loader:
                temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
                output = model(temperature, elevation)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print("Early stopping triggered.")
            break
    
    print("Training complete! Best model saved as:", save_path)
    
    # Save losses data
    np.save(os.path.join(save_path, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(save_path, "val_losses.npy"), np.array(val_losses))

    # Plot Loss Curves
    # plt.figure(figsize=(10,5))
    # plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    # plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.title("Training and Validation Loss")
    # plt.savefig(os.path.join(fig_path, "loss_curves.png"))

def evaluate_and_plot(model, test_loader, device="cuda"):
    model.eval()
    criterion = nn.MSELoss()
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
    
    # Get top 5 and bottom 5 examples based on loss
    top_5_idx = test_losses.argsort()[-5:][::-1]  # Highest loss
    bottom_5_idx = test_losses.argsort()[:5]      # Lowest loss
    
    # Plot results
    fig, axes = plt.subplots(5, 2, figsize=(10, 15))
    for i, idx in enumerate(top_5_idx):
        axes[i, 0].imshow(predictions[idx][0], cmap='coolwarm')
        axes[i, 0].set_title(f"Top {i+1} - Prediction (Loss: {test_losses[idx]:.4f})")
        axes[i, 1].imshow(targets[idx][0], cmap='coolwarm')
        axes[i, 1].set_title(f"Top {i+1} - Target")
    
    for i, idx in enumerate(bottom_5_idx):
        axes[i, 0].imshow(predictions[idx][0], cmap='coolwarm')
        axes[i, 0].set_title(f"Bottom {i+1} - Prediction (Loss: {test_losses[idx]:.4f})")
        axes[i, 1].imshow(targets[idx][0], cmap='coolwarm')
        axes[i, 1].set_title(f"Bottom {i+1} - Target")
    
    plt.tight_layout()
    plt.show()
    print(f"Test Loss: {test_losses.mean():.4f}")

# ------------------ Load, Train and Evaluate ------------------
data_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data"
input_dir = os.path.join(data_path, "1h_2D_sel_cropped_gridded_clustered_threshold_blurred/cluster_0")
target_dir = os.path.join(data_path, "1h_2D_sel_cropped_gridded_clustered_threshold/cluster_0")
elevation_dir = os.path.join(data_path, "dem_squares")
models_dir = "/scratch/fquareng/models/UNet-Baseline"
batch_size = 16
num_epochs = 50
patience = 5
device = "cuda" if torch.cuda.is_available() else "cpu"

exp_id = f"test_{int(time.time())}"
best_model_path = os.path.join(models_dir, f"best_model_{exp_id}.pth")

variable = "T_2M"

print("Loading the data ...")
train_loader, val_loader, test_loader = get_dataloaders(variable, input_dir, target_dir, elevation_dir, batch_size)

print("Initializing the model ...")
model = UNet8x()

print("Beginning training model ...")
train_model(model, train_loader, val_loader, num_epochs, device=device, save_path=best_model_path, patience=patience)

print("Loading best model ...")
model.load_state_dict(torch.load(best_model_path))
model.to(device)

evaluate_and_plot(model, test_loader, device)



def main():
    return 0

if __name__ == "__main__":
    main()