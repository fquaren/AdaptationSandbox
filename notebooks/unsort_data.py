import os
import shutil
from collections import defaultdict
from datetime import datetime
import concurrent.futures

def unsort_files(input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # List all directories (months) in the input directory
    with os.scandir(input_dir) as it:
        months = [entry.name for entry in it if entry.is_dir()]

    # Create a dictionary to store files grouped by their original filenames
    files_by_original_name = defaultdict(list)

    # Loop through each month directory
    for month in months:
        month_dir = os.path.join(input_dir, month)

        # Loop through each day in the month directory
        with os.scandir(month_dir) as it:
            days = [entry.name for entry in it if entry.is_dir()]

        # Process files in each day directory
        for day in days:
            day_dir = os.path.join(month_dir, day)

            # Get all files in the current day directory
            with os.scandir(day_dir) as it:
                files = [entry.name for entry in it if entry.is_file()]

            # Group the files by their original filenames (assuming they were sorted before)
            for file in files:
                files_by_original_name[file].append(os.path.join(day_dir, file))

    # Function to move files back to the original directory
    def move_files_back(file, file_paths):
        for file_path in file_paths:
            original_path = os.path.join(output_dir, file)
            shutil.move(file_path, original_path)
            print(f"Moved {file_path} back to {output_dir}")

    # Use ThreadPoolExecutor to move files back to the original directory in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for file, file_paths in files_by_original_name.items():
            futures.append(executor.submit(move_files_back, file, file_paths))

        # Wait for all tasks to finish
        concurrent.futures.wait(futures)

    print("File unsorting completed.")

if __name__ == '__main__':
    # Define the input and output directories
    input_dir = "/Users/fquareng/data/1h_2D_sel_sorted"  # Directory with the sorted data
    output_dir = "/Users/fquareng/data/1h_2D_sel"      # Original directory to restore files to
    
    # Call the function to unsort the files
    unsort_files(input_dir, output_dir)