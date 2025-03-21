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
FRONTEND_HOST_HOST="ela.cscs.ch"
REMOTE_HOST="balfrin.cscs.ch"
REMOTE_PATH="/capstor/store/cscs/c2sm/scclim/climate_simulations/RUN_2km_cosmo6_climate/output/lm_f/1h_2D"
LOCAL_PATH="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/DA/T_2M"

# File to store the list of selected files
FILE_LIST="/work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/data/DA/selected_files.txt"

# Step 1: SSH into the remote server and find the selected files
ssh $REMOTE_USER@$FRONTEND_HOST 
ssh $REMOTE_USER@$REMOTE_HOST << 'EOF' > /users/fquareng/selected_files.txt
cd /capstor/store/cscs/c2sm/scclim/climate_simulations/RUN_2km_cosmo6_climate/output/lm_f/1h_2D

# List all files, filter only those from the year 2011, and sort them
ls lffd2011*.nz | sort > /users/fquareng/all_files.txt

# Initialize shift counter
shift=0

# Declare an associative array to group files by date
declare -A files_by_date

# Loop through files and categorize them by date
while read file; do
    date_part=\$(echo "\$file" | grep -oE '2011[0-9]{4}')
    hour_part=\$(echo "\$file" | grep -oE '[0-9]{10}' | cut -c9-10)
    
    files_by_date[\$date_part]+="\$hour_part \$file\n"
done < /users/fquareng/all_files.txt

# Sort dates
sorted_dates=(\$(printf "%s\n" "\${!files_by_date[@]}" | sort))

# Process each day and select the required hours
for date in "\${sorted_dates[@]}"; do
    # Define the selected hours for the current shift
    selected_hours=(\$(( (0 + shift) % 24 )) \$(( (8 + shift) % 24 )) \$(( (15 + shift) % 24 )))

    # Extract file list for the date
    IFS=$'\n'
    files=(\$(echo -e "\${files_by_date[\$date]}" | sort -n))

    # Select files matching the desired hours
    selected_files=()
    for entry in "\${files[@]}"; do
        read -r hour filename <<< "\$entry"
        for h in "\${selected_hours[@]}"; do
            if [[ "\$hour" -eq "\$h" ]]; then
                selected_files+=("\$filename")
            fi
        done
    done

    # Print selected files (to be captured in selected_files.txt)
    if [[ "\${#selected_files[@]}" -eq 3 ]]; then
        printf "%s\n" "\${selected_files[@]}"
    fi

    # Increment shift
    shift=\$(( (shift + 1) % 24 ))
done
EOF

# Step 2: Copy the selected files from the remote server
rsync -avz -e "ssh -J $REMOTE_USER@$FRONTEND_HOST" $REMOTE_USER@$REMOTE_HOST:/users/fquareng/selected_files.txt $TMP_FILE_LIST

# Step 3: Copy the selected NetCDF files from remote to local
mkdir -p $LOCAL_PATH
# rsync -av --files-from=$TMP_FILE_LIST $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/ $LOCAL_PATH/
rsync -avz -e "ssh -J $REMOTE_USER@$FRONTEND_HOST" --files-from=$TMP_FILE_LIST $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/ $LOCAL_PATH/

# Step 4: Extract the "T_2M" variable and save as new files
cd $LOCAL_PATH
while read file; do
    new_file="$T_2M_{file%.nz}.nz"
    cdo selname,T_2M "$file" "$new_file"  # Using CDO to extract the variable
    # Alternative using NCO:
    # ncks -v T_2M "$file" "$new_file"
done < $TMP_FILE_LIST