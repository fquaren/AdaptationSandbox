import os
import shutil
from collections import defaultdict
from datetime import datetime
import concurrent.futures

def sort_files_by_month(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all NetCDF files in the input directory with the correct pattern
    with os.scandir(input_dir) as it:
        input_files = [entry.name for entry in it if entry.is_file() and entry.name.endswith(".nz") and entry.name.startswith("lffd")]
    
    # Create a dictionary to store files grouped by month and day (YYYYMM -> day -> file list)
    files_by_month_day = defaultdict(lambda: defaultdict(list))

    # Loop through each file and sort based on the month and day
    for file in input_files:
        # Extract the date (YYYYMMDD) from the filename (assuming filename format: lffdYYYYMMDDhhmmss.nz)
        timestamp_str = file[4:12]  # Extract the 'YYYYMMDD' part of the filename
        month_str = timestamp_str[:6]  # First 6 characters represent 'YYYYMM'
        day_str = timestamp_str[6:8]  # Characters 7-8 represent 'DD'

        # Group files by their month and day (YYYYMM -> DD)
        files_by_month_day[month_str][day_str].append(file)

    # Function to move files for a specific month and create README
    def move_files_and_create_readme(month, files_by_day):
        month_dir = os.path.join(output_dir, month)
        os.makedirs(month_dir, exist_ok=True)

        # Create the README file
        readme_path = os.path.join(month_dir, "README.txt")
        with open(readme_path, "w") as readme_file:
            readme_file.write(f"File sorting for month: {month}\n")
            readme_file.write(f"Total files in this month: {sum(len(files) for files in files_by_day.values())}\n")
            readme_file.write("\nFile counts per day:\n")
            
            # Write the count of files for each day
            for day, files in files_by_day.items():
                readme_file.write(f"  Day {day}: {len(files)} files\n")
        
        # Move the files to their respective month directory
        for day, files in files_by_day.items():
            day_dir = os.path.join(month_dir, day)
            os.makedirs(day_dir, exist_ok=True)
            for file in files:
                input_path = os.path.join(input_dir, file)
                output_path = os.path.join(day_dir, file)
                shutil.move(input_path, output_path)
                print(f"Moved {file} to {day_dir}")

    # Use ThreadPoolExecutor to move files and create readme files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for month, files_by_day in files_by_month_day.items():
            # Submit each month's file move task and README creation to the executor
            futures.append(executor.submit(move_files_and_create_readme, month, files_by_day))

        # Wait for all tasks to finish
        concurrent.futures.wait(futures)

    print("File sorting and README generation completed.")

if __name__ == '__main__':
    # Define the input and output directories
    input_dir = "/Users/fquareng/data/1h_2D_sel/"
    output_dir = "/Users/fquareng/data/1h_2D_sel_sorted"
    
    # Call the function to sort the files and generate README files
    sort_files_by_month(input_dir, output_dir)