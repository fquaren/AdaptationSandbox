import torch
import torch.nn as nn
import torch.optim as optim
import xarray as xr
import os
from torch.utils.data import Dataset, DataLoader
import re
import time

# ------------------ U-Net Model ------------------
class UNet8x(nn.Module):
    def __init__(self):
        super(UNet8x, self).__init__()
        
        # Elevation Downsampling Block (to match temperature resolution)
        self.downsample_elevation = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 4x downsampling
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 2x downsampling (total 8x)
            nn.ReLU(inplace=True),
        )

        # Define encoding layers
        self.encoder1 = self.conv_block(33, 64)  # 32 (from elevation) + 1 (temperature)
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
        
        # Additional upsampling layers for 10x resolution
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
        
        # Additional upsampling for 10x resolution
        d_final1 = self.upconv_final1(d1)
        d_final2 = self.upconv_final2(d_final1)
        d_final3 = self.upconv_final3(d_final2)
        return self.output(d_final3)

# ------------------ Dataset Class ------------------
import torch
import xarray as xr
import os
import re
from torch.utils.data import Dataset

class SingleVariableDataset(Dataset):
    def __init__(self, input_files, elev_dir, target_files):
        """
        Args:
            input_files (list): Paths to low-resolution variable (temperature) NetCDF files.
            elev_dir (str): Directory containing elevation NetCDF files named as "A_B_something.nc".
            target_files (list): Paths to corresponding high-resolution target NetCDF files.
        """
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
        var_file = self.input_files[idx]
        target_file = self.target_files[idx]

        # Extract A and B from input filename
        A, B = self._extract_numbers(var_file)

        # Ensure the correct elevation file exists
        if (A, B) not in self.elev_files:
            raise FileNotFoundError(f"No elevation file found for {A}_{B} in {self.elev_dir}")

        elev_file = self.elev_files[(A, B)]

        # Load variable data (temperature)
        with xr.open_dataset(var_file) as ds:
            variable = torch.tensor(ds.variable.values, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        # Load high-resolution target
        with xr.open_dataset(target_file) as ds:
            target = torch.tensor(ds.target_variable.values, dtype=torch.float32).unsqueeze(0)  # [1, H_target, W_target]

        # Load the correct elevation data
        with xr.open_dataset(elev_file) as ds:
            elevation_data = torch.tensor(ds.elevation.values, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        return variable, elevation_data, target


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


def get_dataloaders(input_dir, target_dir, elev_file, batch_size=4):
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".nz")]
    target_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".nz")]

    # Check input and target files contain the same number of samples
    assert len(input_files) == len(target_files), "Number of input and target files must match."

    (train_inputs, train_targets), (val_inputs, val_targets), (test_inputs, test_targets) = split_dataset(input_files, target_files)

    train_dataset = SingleVariableDataset(train_inputs, elev_file, train_targets)
    val_dataset = SingleVariableDataset(val_inputs, elev_file, val_targets)
    test_dataset = SingleVariableDataset(test_inputs, elev_file, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ------------------ Training Function ------------------
def train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4, device="cuda", save_path="best_model.pth"):
    """
    Trains the U-Net model with given training and validation data.
    """
    model.to(device)
    criterion = nn.MSELoss()  # Regression task
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Learning rate decay

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for temperature, elevation, target in train_loader:
            temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(temperature, elevation)  # Forward pass
            loss = criterion(output, target)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for temperature, elevation, target in val_loader:
                temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
                output = model(temperature, elevation)
                val_loss += criterion(output, target).item()

        val_loss /= len(val_loader)
        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

    print("Training complete! Best model saved as:", save_path)


# ------------------ Load and Train ------------------
data_path = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data"
input_dir = os.path.join(data_path, "1h_2D_sel_cropped_gridded_clustered_threshold/cluster_0")
target_dir = os.path.join(data_path, "1h_2D_sel_cropped_gridded_clustered_threshold_blurred/cluster_0")
elevation_dir = os.path.join(data_path, "dem_squares")
models_dir = "/scratch/fquareng/models/UNet-baseline"  # Directory to save trained models
batch_size = 4
num_epochs = 50
device = "cuda" if torch.cuda.is_available() else "cpu"

# Test id for saving the best model
test_id = f"test_{int(time.time())}"  # e.g., test_1615292950
best_model_path = os.path.join(models_dir, f"best_model_{test_id}.pth")

# Load data
train_loader, val_loader, test_loader = get_dataloaders(input_dir, target_dir, elevation_dir, batch_size)

# Initialize model
model = UNet8x()

# Train model
train_model(model, train_loader, val_loader, num_epochs, device=device, save_path=best_model_path)

# Load best model
model.load_state_dict(torch.load(best_model_path))
model.to(device)


# ------------------ Evaluate on Test Set ------------------
def evaluate(model, test_loader, device="cuda"):
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for temperature, elevation, target in test_loader:
            temperature, elevation, target = temperature.to(device), elevation.to(device), target.to(device)
            output = model(temperature, elevation)
            test_loss += criterion(output, target).item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

evaluate(model, test_loader, device)