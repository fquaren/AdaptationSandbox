import rasterio
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

def retrieve_data_from_point(dem_file, temp_file, variable="T_2M", time_step=0):
    # Load DEM data
    with rasterio.open(dem_file) as dem_src:
        altitude = dem_src.read(1)  # Read the first band
        dem_transform = dem_src.transform

    # Load temperature data
    temp_ds = xr.open_dataset(temp_file)
    temperature = temp_ds[variable].isel(time=time_step).values  # Select the first time step

    # Load latitude and longitude data
    latitudes = temp_ds["rlat"].values
    longitudes = temp_ds["rlon"].values

    # Ensure data alignment
    if altitude.shape != temperature.shape:
        raise ValueError("Altitude and temperature dimensions do not match. Check input files.")

    # Flatten data for easier manipulation
    altitude_flat = altitude.flatten()
    temperature_flat = temperature.flatten()

    # Generate index arrays for latitude and longitude
    lat_grid, lon_grid = np.meshgrid(latitudes, longitudes, indexing="ij")
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()

    # Remove NaN values
    valid_mask = ~np.isnan(altitude_flat) & ~np.isnan(temperature_flat)
    altitude_flat = altitude_flat[valid_mask]
    temperature_flat = temperature_flat[valid_mask]
    lat_flat = lat_flat[valid_mask]
    lon_flat = lon_flat[valid_mask]

    def on_click(event):
        # Get the clicked point's x and y data
        x, y = event.xdata, event.ydata

        if x is None or y is None:
            print("Click outside the scatter plot")
            return

        # Find the closest point in the dataset
        distances = np.sqrt((altitude_flat - x) ** 2 + (temperature_flat - y) ** 2)
        min_index = np.argmin(distances)

        # Retrieve original data
        selected_lat = lat_flat[min_index]
        selected_lon = lon_flat[min_index]
        selected_altitude = altitude_flat[min_index]
        selected_temperature = temperature_flat[min_index]

        print(f"Selected Point:")
        print(f"Latitude: {selected_lat}, Longitude: {selected_lon}")
        print(f"Altitude: {selected_altitude}, Temperature: {selected_temperature}")

    # Plot scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(altitude_flat, temperature_flat, s=10, alpha=0.5, label="Data Points")
    plt.title("Altitude vs Temperature Scatter Plot")
    plt.xlabel("Altitude (m)")
    plt.ylabel("Temperature (°C)")
    plt.legend()

    # Connect the click event
    cid = plt.gcf().canvas.mpl_connect("button_press_event", on_click)

    plt.show()

# Example usage
dem_file = '/Users/fquareng/data/europe_dem_low_res.tif'
temp_file = '/Users/fquareng/data/T_2M.nc'
retrieve_data_from_point(dem_file, temp_file)