"""
    # Script Description

This Python script processes a set of NetCDF files containing Digital Elevation Model (DEM) data to compute various topographic metrics and visualize their relationships.
The script includes the following features:

## Features

### 1. Topographic Metric Calculations
- **Mean Elevation**: Average elevation within each DEM square.
- **Mean Slope**: Average slope magnitude derived from elevation gradients.
- **Mean Curvature**: Average curvature calculated as the magnitude of second derivatives of elevation.
- **Mean TRI (Topographic Ruggedness Index)**: Standard deviation of elevations as a measure of surface roughness.
- **Combined Metric**: A sum of all the above metrics to rank DEM squares.

### 2. Interactive Scatter Plots
- Generates scatter plots of **mean elevation** vs. each metric, including the combined metric.
- Enables interactivity with **Matplotlib’s mplcursors**, allowing users to hover over a scatter point and display the DEM data for the corresponding square in a separate plot.

### 3. Data Organization
- Stores all computed metrics and DEM data in a dictionary for efficient access during visualization and interaction.

## Intended Usage
- Helps rank and analyze DEM squares based on their topographic features.
- Provides insights into terrain variability through interactive visualizations.

This script is useful for analyzing and comparing DEM tiles for applications in glaciology, topography, and other geospatial studies.
"""


import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import mplcursors  # For interactivity

# Function to compute slope
def compute_slope(dem_data):
    sx, sy = np.gradient(dem_data)
    slope = np.sqrt(sx**2 + sy**2)
    return slope

def compute_mean_slope(dem_data):
    slope = compute_slope(dem_data)
    return np.mean(slope)

def compute_curvature(dem_data):
    sx, sy = np.gradient(dem_data)
    sxx = np.gradient(sx)[0]
    syy = np.gradient(sy, axis=0)[0]
    curvature = np.sqrt(sxx**2 + syy**2)
    return curvature

def compute_tri(dem_data):
    tri = np.std(dem_data)
    return tri

# Path where your NetCDF files are located
nc_files_path = "/Users/fquareng/data/dem_squares"  # Update this path
nc_files = [f for f in os.listdir(nc_files_path) if f.endswith('.nc')]

# Dictionary to store metrics and DEM data
metrics = {}

for nc_file in nc_files:
    file_path = os.path.join(nc_files_path, nc_file)
    ds = xr.open_dataset(file_path, engine="netcdf4")
    dem_data = ds.HSURF.values  # Extract DEM values
    
    mean_elevation = np.mean(dem_data)
    mean_slope = compute_mean_slope(dem_data)
    mean_curvature = np.mean(compute_curvature(dem_data))
    mean_tri = compute_tri(dem_data)
    combined_metric = mean_elevation + mean_slope + mean_curvature + mean_tri
    
    coords = nc_file.split('_')
    bottom_left_x = int(coords[1])
    bottom_left_y = int(coords[2].split('.')[0])
    
    metrics[(bottom_left_x, bottom_left_y)] = {
        "mean_elevation": mean_elevation,
        "mean_slope": mean_slope,
        "mean_curvature": mean_curvature,
        "mean_tri": mean_tri,
        "combined_metric": combined_metric,
        "dem_data": dem_data,  # Store the DEM data for plotting
    }

# Extract metrics for plotting
mean_elevation = [values["mean_elevation"] for values in metrics.values()]
mean_slope = [values["mean_slope"] for values in metrics.values()]
mean_curvature = [values["mean_curvature"] for values in metrics.values()]
mean_tri = [values["mean_tri"] for values in metrics.values()]
combined_metric = [values["combined_metric"] for values in metrics.values()]
coords = list(metrics.keys())

# Scatter plot with interactive DEM display
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(mean_elevation, mean_slope, color='skyblue', alpha=0.7)
ax.set_xlabel('Mean Elevation [m]')
ax.set_ylabel('Mean Slope [m/m]')
ax.set_title('Mean Elevation vs Mean Slope')
ax.grid(True)

# Add hover functionality
cursor = mplcursors.cursor(scatter, hover=True)

# Callback to display the DEM plot
@cursor.connect("add")
def on_add(sel):
    index = sel.index  # Get the index of the hovered point
    x, y = coords[index]  # Get the coordinates of the square
    dem_data = metrics[(x, y)]["dem_data"]  # Fetch the DEM data for this square
    
    # Create a new figure for the DEM plot
    plt.figure(figsize=(8, 6))
    plt.imshow(dem_data, cmap='terrain', origin='lower')
    plt.colorbar(label='Elevation [m]')
    plt.title(f'DEM for Square ({x}, {y})')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Display the scatter plot
plt.show()
