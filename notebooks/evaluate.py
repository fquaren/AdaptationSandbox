from tqdm import tqdm
import os
import glob
import torch
import torch.nn as nn
import xarray as xr
import numpy as np


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


data_path = "/Users/fquareng/data/"
dem_path = "dem_squares"
# model_path = "/Users/fquareng/models/UNet_x8_on_7.pth"
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

for N in tqdm(list(range(8))):
    target_path = f"1h_2D_sel_cropped_gridded_clustered_threshold/cluster_{N}/"
    source_path = f"1h_2D_sel_cropped_gridded_clustered_threshold_blurred/cluster_{N}/"
    source_files = sorted(glob.glob(os.path.join(data_path, source_path, "*.nz")))
    target_files = sorted(glob.glob(os.path.join(data_path, target_path, "*.nz")))

    mse_list = []
    for source_path, target_path in tqdm(zip(source_files, target_files)):
        
        A = source_path.split("/")[-1].split("_")[0]
        B = source_path.split("/")[-1].split("_")[1]
        ds = xr.open_dataset(source_path)

        t2m = ds["T_2M"].sel(time=ds["T_2M"].time[0])
        elev = xr.open_mfdataset(os.path.join(data_path, dem_path, f"{A}_{B}_*"))["HSURF"]

        input_data = torch.tensor(t2m.values).float().unsqueeze(0).unsqueeze(0).to(device) # Add batch dimension
        elev_data = torch.tensor(elev.values).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(variable=input_data, elevation=elev_data)
        predicted_output = output.squeeze().cpu().numpy()

        ds = xr.open_mfdataset(target_path)
        target = ds["T_2M"].sel(time=ds["T_2M"].time[0])

        diff = target - predicted_output
        squared_diff = diff**2
        mse = squared_diff.mean().values
        mse_list.append(mse)

    print(np.mean(mse_list))
    print(np.std(mse_list))