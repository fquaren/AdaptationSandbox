#!/bin/bash
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name 19
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 10G
#SBATCH --time 12:00:00

module load cdo

# Remote server details
LOCAL_PATH="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/DA/1d-PS-RELHUM_2M-T_2M"

# File to store the list of selected files
FILE_LIST="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/DA/selected_files_2019.txt"

cd $LOCAL_PATH
while read file; do
    new_file="${file%.nz}_da.nz"
    cdo selname,T_2M,RELHUM_2M,PS "$file" "$new_file"  # Using CDO to extract the variable
    # Clean
    if [[ -f "$new_file" ]]; then
        rm "$file"
    else
        echo "Error: Failed to create $new_file, keeping original file $file."
    fi
done < $FILE_LIST