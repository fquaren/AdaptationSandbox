#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type NONE
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name UNet
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu-gh
#SBATCH --gres gpu:0
#SBATCH --gres-flags enforce-binding
#SBATCH --cpus-per-task 12
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 100G
#SBATCH --time 01:00:00

sleep 8000
