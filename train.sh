#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 96G
#SBATCH --time 72:00:00
#SBATCH --account=vita
#SBATCH --gres gpu:2

srun singularity exec --bind /scratch/izar --nv ../pytorch_latest.sif \
  /bin/bash -c "source ../.venv/apollo/bin/activate && time CUDA_VISIBLE_DEVICES=0,1 python3 -u -m openpifpaf.train $(printf "%s " "$@")"
