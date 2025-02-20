#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type ALL
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name test_training_script
#SBATCH --output outputs/train_%j
#SBATCH --error job_errors/train_%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 10G
#SBATCH --time 00:30:00

module load singularityce/4.1.0

export SINGULARITY_BINDPATH="/scratch,/dcsrsoft,/users,/work,/reference"

singularity run --nv /dcsrsoft/singularity/containers/pytorch/pytorch-ngc-24.05-2.4.sif

source /users/fquareng/.bashrc

micromamba run -n dwnscl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/AdaptationSandbox/notebooks/train_test.py