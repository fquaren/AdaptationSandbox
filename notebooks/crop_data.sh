#!/bin/bash

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name crop
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 10G
#SBATCH --time 12:00:00

module load nco

# Directory containing input NetCDF files
source_dir="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/domain_adaptation/DA/1d-PS-RELHUM_2M-T_2M/"
# Directory to store cropped files
output_dir="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/domain_adaptation/DA/1d-PS-RELHUM_2M-T_2M_cropped/"

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
find "$source_dir" -type f -name "*.nz" | while read -r file; do
    process_file "$file"
done