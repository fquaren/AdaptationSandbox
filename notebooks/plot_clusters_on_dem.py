import os
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# --- Load DEM Data ---
data_path = "/users/fquareng/lffd20940101000000c.nc"
ds = xr.open_dataset(data_path, engine="netcdf4")
hsurf = ds.HSURF.sel(time="2095-03-20T18:00:00")

# --- Directory containing cluster folders ---
base_dir = "/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/DA/1d-PS-RELHUM_2M-T_2M_cropped_gridded_clustered_threshold_12"

# --- Extract coordinates and cluster labels ---
cluster_data = {}

for cluster_folder in sorted(os.listdir(base_dir)):  
    cluster_match = re.match(r"cluster_(\d+)", cluster_folder)  
    if cluster_match:  
        cluster_label = int(cluster_match.group(1))  
        cluster_path = os.path.join(base_dir, cluster_folder)  
        cluster_data[cluster_label] = {"coords_x": [], "coords_y": []}  
        
        for filename in os.listdir(cluster_path):  
            match = re.match(r"(\d+)_(\d+)_.*\.nz", filename)  
            if match:  
                coord_x, coord_y = int(match.group(1)), int(match.group(2))  
                cluster_data[cluster_label]["coords_x"].append(coord_x)
                cluster_data[cluster_label]["coords_y"].append(coord_y)

# --- Convert indices to longitude & latitude ---
for cluster_label in cluster_data:
    cluster_data[cluster_label]["coords_x"] = [ds.rlon[64::128][x] for x in cluster_data[cluster_label]["coords_x"]]
    cluster_data[cluster_label]["coords_y"] = [ds.rlat[64::128][y] for y in cluster_data[cluster_label]["coords_y"]]

# --- Plotting each cluster separately ---
for cluster_label, data in cluster_data.items():
    plt.figure(figsize=(12, 8))
    
    # Plot the DEM background
    plt.pcolormesh(ds.rlon, ds.rlat, hsurf, cmap='coolwarm', shading='auto')
    plt.colorbar(label='Elevation [m]')
    
    # Overlay the scatter plot of the current cluster
    plt.scatter(data["coords_x"], data["coords_y"], color='red', edgecolors='k', alpha=0.7, s=200)
    
    # Add grid lines
    for lon, lat in zip(ds.rlon[::128], ds.rlat[::128]):
        plt.axvline(x=lon, color='black', linestyle='--', linewidth=0.5)
        plt.axhline(y=lat, color='black', linestyle='--', linewidth=0.5)
    
    # Labels and Title
    plt.xlabel('Rotated longitude')
    plt.ylabel('Rotated latitude')
    plt.title(f'Cluster {cluster_label}')
    
    # Save the plot
    output_path = f"/users/fquareng/cluster_{cluster_label}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tightlayout()
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

