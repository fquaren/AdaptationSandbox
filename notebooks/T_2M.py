import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns  # For KDE plot
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs


def extract_temperature_region(dataset, lat_start, lon_start, region_size=100, variable="T_2M"):
    """
    Extract the temperature time series for a region defined by lat_start, lon_start and a given region size.
    """
    # Define the region's grid bounds
    lat_end = lat_start + region_size
    lon_end = lon_start + region_size
    
    # Select the temperature values for the region (assuming rlat, rlon are grid coordinates)
    region_data = dataset[variable].sel(rlat=slice(lat_start, lat_end), rlon=slice(lon_start, lon_end))
    
    return region_data

def plot_temperature_distribution(region_data_list, labels):
    """
    Plot the temperature distribution (histogram and KDE) for selected regions.
    """
    plt.figure(figsize=(12, 8))
    
    for i, region_data in enumerate(region_data_list):
        # Flatten the region data to create a 1D array of temperatures for KDE and histogram
        region_values = region_data.values.flatten()
        
        # Plot histogram and KDE for the region
        label = labels[i]
        sns.histplot(region_values, kde=True, stat="density", bins=30, label=label)
    
    plt.title("Temperature Distribution for Selected Regions")
    plt.xlabel("2m Temperature (°C)")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def main(input_file, regions):
    """
    Main function to load the dataset and process the regions.
    """
    # Load the dataset
    dataset = xr.open_dataset(input_file)
    
    # Extract temperature data for the selected regions
    region_data_list = []
    labels = []
    
    for lon_start, lat_start in regions:
        region_data = extract_temperature_region(dataset, lat_start, lon_start)
        region_data_list.append(region_data)
        labels.append(f"Lat: {lat_start}, Lon: {lon_start}")
    
    # Plot the distribution of the temperature time series for each region
    plot_temperature_distribution(region_data_list, labels)


if __name__ == '__main__':
    input_file = "/Users/fquareng/data/T_2M.nc"  # Replace with the actual dataset file path
    
    # Define the starting points of the regions (latitudes and longitudes)
    # These represent the bottom-left corners of 50x50 regions.
    # Example: [(lat_start_1, lon_start_1), (lat_start_2, lon_start_2), ...]
    regions = [(300, 400), (800, 700), (900, 1300), (200, 1000)]  # Modify with appropriate region coordinates
    
    # Run the analysis
    main(input_file, regions)