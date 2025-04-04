import os
import re
from collections import defaultdict

def get_selected_files(file_list, target_year):
    """
    Selects the first 24 hours of each month from a given year.
    
    Parameters:
        file_list (list): List of filenames.
        target_year (int): The year to filter files from.
    
    Returns:
        list: Selected filenames.
    """
    # Sort files to ensure chronological order
    file_list.sort()

    # Regular expression pattern to extract date and hour: lffdYYYYMMDDhhmmss
    date_pattern = re.compile(r"lffd(\d{4})(\d{2})(\d{2})(\d{2})\d{4}")  
    
    files_by_month = defaultdict(list)

    for file in file_list:
        match = date_pattern.search(file)
        if match:
            year, month, day, hour = match.groups()
            
            # Convert year to integer and compare with target_year
            if int(year) == target_year:
                year_month = f"{year}{month}"  # Format: YYYYMM
                files_by_month[year_month].append((int(hour), file))  # Store hour as int for sorting

    # Sort each month's files by hour
    for year_month in files_by_month:
        files_by_month[year_month].sort()

    selected_files = []

    # Select the first 24 hours of each month
    for year_month, day_files in files_by_month.items():
        selected_day_files = [file for hour, file in day_files if hour < 24]  # Select hours 00-23

        selected_files.extend(selected_day_files)

    return selected_files

# Example usage:
target_year = 2018  # Set the target year
directory_path = "/capstor/store/cscs/c2sm/scclim/climate_simulations/RUN_2km_cosmo6_climate/output/lm_f/1h_2D/"
file_list = os.listdir(directory_path)  # Get list of files from directory
selected_files = get_selected_files(file_list, target_year)

# Save to a text file
output_file = "selected_files.txt"
with open(output_file, "w") as f:
    for filename in selected_files:
        f.write(filename + "\n")

print(f"Selected {len(selected_files)} files from the year {target_year}.")