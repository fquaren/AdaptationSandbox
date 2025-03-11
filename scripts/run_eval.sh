#!/bin/bash -l
#SBATCH --account tbeucler_downscaling
#SBATCH --mail-type NONE
#SBATCH --mail-user filippo.quarenghi@unil.ch

#SBATCH --chdir /scratch/fquareng/
#SBATCH --job-name c6
#SBATCH --output outputs/%j
#SBATCH --error job_errors/%j

#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --gres-flags enforce-binding
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 8G
#SBATCH --time 00:10:00

module load singularityce/4.1.0
export SINGULARITY_BINDPATH="/scratch,/dcsrsoft,/users,/work,/reference"
singularity run --nv /dcsrsoft/singularity/containers/pytorch/pytorch-ngc-24.05-2.4.sif

source /users/fquareng/.bashrc

CLUSTER_ID=6
EXPERIMENT_ID="${CLUSTER_ID}_${SLURM_JOB_ID}_$(openssl rand -hex 4)"
echo "Experiment ID: $EXPERIMENT_ID"

micromamba run -n dwnscl python /work/FAC/FGSE/IDYST/tbeucler/downscaling/fquareng/AdaptationSandbox/src/eval.py
