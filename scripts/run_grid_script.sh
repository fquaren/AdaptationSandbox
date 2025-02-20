#!/bin/bash
#SBATCH --account tbeucler_downscaling 
#SBATCH --mail-type NONE 
#SBATCH --mail-user filippo.quarenghi@unil.ch
#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name python_%j
#SBATCH --output /scratch/fquareng/outputs/output_%j.out
#SBATCH --error /scratch/fquareng/job_errors/job_error_%j.log
#SBATCH --partition cpu
#SBATCH --cpus-per-task 8 
#SBATCH --mem 10G 
#SBATCH --time 00:30:00 

# Initialize the current shell to allow micromamba to activate environments
# dcsrsoft use 20240303
# module load miniforge3
# conda_init

# Activate the environment
source /users/fquareng/.bashrc
micromamba activate dwnscl

# Run the Python script
# /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/conda/envs/dwnscl/bin/
python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/AdaptationSandbox/notebooks/grid_12_data.py