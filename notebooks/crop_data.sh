"""
README: NetCDF Cropping Script

This script processes multiple NetCDF files by cropping the first 6 pixels from the `rlat` and `rlon` dimensions using the `ncks` command. It uses GNU Parallel to optimize processing by running the tasks on multiple CPU cores.

Requirements

- GNU Parallel: For parallel processing of files.
- NCO (NetCDF Operators): For NetCDF file manipulation.
- bash: The script is written for bash shell.

Installation
1. Install GNU Parallel (if not installed):
    - On macOS, you can install it using Homebrew:
      brew install parallel
2. Install NCO (if not installed):
    - On macOS, install via Homebrew:
      brew install nco

Script Overview

- Functionality: The script loops over all NetCDF files in a source directory, crops the first 6 pixels along the `rlat` and `rlon` dimensions using `ncks`, and saves the cropped files to an output directory.
- Parallel Processing: The script processes multiple files in parallel to speed up execution. The number of parallel jobs can be adjusted based on the number of available CPU cores.

Usage

Configuration

1. Set the source directory (source_dir) where your input `.nz` NetCDF files are located.
2. Set the output directory (output_dir) where the cropped NetCDF files will be saved.

Run the Script

1. Save the script as `crop_nc_files.sh` or any desired name.
2. Provide execute permissions to the script:
    chmod +x crop_nc_files.sh
3. Run the script:
    ./crop_nc_files.sh

How It Works

1. The script uses `find` to locate all `.nz` files in the specified source directory.
2. The `process_file` function is defined to crop each NetCDF file using `ncks` with the dimensions `rlat` and `rlon`, removing the first 6 pixels from each dimension.
3. The script then uses `parallel` to execute this function on multiple files at once, making the process faster by utilizing multiple CPU cores.

Customization

- Number of parallel jobs: The `-j 4` flag in the `parallel` command specifies 4 jobs running in parallel. Adjust this number based on the number of CPU cores you want to utilize (e.g., `-j 8` for 8 jobs).
  
    Example:
    parallel -j 8 process_file

- Dimensions: The script crops the first 6 pixels from the `rlat` and `rlon` dimensions. You can modify the slicing in the `ncks` command if you need different cropping or to work with other dimensions.

Example Output

The script will print the following for each processed file:
Processed: /path/to/input/file1.nz -> /path/to/output/file1.nz
Processed: /path/to/input/file2.nz -> /path/to/output/file2.nz
...

Troubleshooting

- If you encounter issues with `ncks` or `parallel` not being found, ensure they are installed and available in your system's `PATH`.
- If the output files are larger than expected, check that the NetCDF files are not being written with extra compression or other options in the `ncks` command.

"""

#!/bin/bash

# Directory containing input NetCDF files
source_dir="/Users/fquareng/data/1h_2D_sel/"

# Directory to store cropped files
output_dir="/Users/fquareng/data/1h_2D_sel_cropped/"

# Function to process a single file
process_file() {
    input_file=$1
    output_file="${output_dir}$(basename $input_file)"
    
    # Crop and compress the file using ncks
    ncks -d rlat,6,,1 -d rlon,6,,1 $input_file $output_file
    
    echo "Processed: $input_file -> $output_file"
}

export -f process_file

# Run the function on multiple cores using GNU parallel
find $source_dir -type f -name "*.nz" | parallel -j 12 process_file