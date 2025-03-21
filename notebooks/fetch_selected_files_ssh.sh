#!/bin/bash
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name cscs
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 10G
#SBATCH --time 72:00:00

module load cdo

# Remote server details
REMOTE_USER="fquareng"
FRONTEND_HOST="ela.cscs.ch"
REMOTE_HOST="balfrin.cscs.ch"
REMOTE_PATH="/capstor/store/cscs/c2sm/scclim/climate_simulations/RUN_2km_cosmo6_climate/output/lm_f/1h_2D"
LOCAL_PATH="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/DA/T_2M"

# File to store the list of selected files
FILE_LIST="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/DA/selected_files.txt"

# Step 1: SSH into the remote server and find the selected files
ssh -T -J $REMOTE_USER@$FRONTEND_HOST $REMOTE_USER@$REMOTE_HOST <<EOF
cd /capstor/store/cscs/c2sm/scclim/climate_simulations/RUN_2km_cosmo6_climate/output/lm_f/1h_2D

# List all files, filter only those from the year 2011, and sort them
find lffd2011*.nz -type f | sort > ~/2011_files.txt

# Initialize shift counter
shift=0

# Declare an associative array to group files by date
declare -A files_by_date

# Use mapfile to load all files into an array for faster processing
mapfile -t files < ~/2011_files.txt

# Loop through files and categorize them by date using awk for fast parsing
for file in "${files[@]}"; do
    # Extract date part (YYYYMMDD) from the filename
    date_part=$(echo "$file" | awk '{print substr($0, 5, 8)}')  # From the 5th character, get 8 characters
    # Extract hour part (hh) from the filename
    hour_part=$(echo "$file" | awk '{print substr($0, 13, 2)}')  # From the 13th character, get 2 characters

    # Ensure date_part is not empty before proceeding
    if [[ -n "$date_part" && -n "$hour_part" ]]; then
        # Append the hour and filename to the associative array for the corresponding date
        files_by_date["$date_part"]+="$hour_part $file "
    else
        echo "Skipping invalid filename: $file"
    fi
done

# Sort dates
sorted_dates=($(for date in "${!files_by_date[@]}"; do echo "$date"; done | sort))

# Process each day and select the required hours
for date in "${sorted_dates[@]}"; do
    # Define the selected hours for the current shift
    selected_hours=($(( (shift) % 24 )) $(( (8 + shift) % 24 )) $(( (15 + shift) % 24 )))

    # Extract file list for the date (already sorted)
    IFS=$'\n' read -r -a files <<< "${files_by_date[$date]}"

    # Select files matching the desired hours
    selected_files=()
    for entry in "${files[@]}"; do
        # Extract hour and filename
        read -r hour filename <<< "$entry"

        # Check if the hour matches any of the selected hours
        if [[ " ${selected_hours[@]} " =~ " $hour " ]]; then
            selected_files+=("$filename")
        fi
    done

    # Print selected files (to be captured in selected_files.txt)
    if [[ ${#selected_files[@]} -eq 3 ]]; then
        printf "%s\n" "${selected_files[@]}" >> ~/selected_files.txt
    fi

    # Increment shift
    shift=$(( (shift + 1) % 24 ))
done
EOF

# Step 2: Copy the selected files from the remote server to local
rsync -avz -e "ssh -J $REMOTE_USER@$FRONTEND_HOST" $REMOTE_USER@$REMOTE_HOST:~/selected_files.txt $FILE_LIST

# Step 3: Copy the selected NetCDF files from remote to local
mkdir -p $LOCAL_PATH
rsync -avz -e "ssh -J $REMOTE_USER@$FRONTEND_HOST" --files-from=$FILE_LIST $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/ $LOCAL_PATH/

# Step 4: Extract the "T_2M" variable and save as new files
cd $LOCAL_PATH
while read file; do
    new_file="T_2M_${file%.nz}.nz"
    cdo selname,T_2M "$file" "$new_file"  # Using CDO to extract the variable
done < $FILE_LIST