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
eval "$(micromamba shell hook --shell=bash)"
module load micromamba/1.4.2

# Activate the environment
micromamba activate dwnscl
# Python
module load python

# Run the Python script
python3 /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/AdaptationSandbox/scripts/test.py