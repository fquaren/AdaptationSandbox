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

# Activate the environment
source /users/fquareng/.bashrc
# micromamba activate dwnscl

# Run the Python script
# /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/conda/envs/dwnscl/bin/
micromamba run -n dwnscl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/AdaptationSandbox/notebooks/cluster_data_threshold.py
