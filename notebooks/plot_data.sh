#!/bin/bash

#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name plot
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition cpu
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 200G
#SBATCH --time 12:00:00

source /users/fquareng/.bashrc
micromamba run -n dl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/AdaptationSandbox/notebooks/plot_cluster_data.py
