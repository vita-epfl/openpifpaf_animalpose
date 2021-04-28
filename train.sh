#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 96G
#SBATCH --time 48:00:00
#SBATCH --account=vita
#SBATCH --gres gpu:2

module load gcc python cuda

srun /bin/bash -c "source ../.venv/animal/bin/activate && time CUDA_VISIBLE_DEVICES=0,1 python3 -m openpifpaf.train $(printf "%s " "$@")"
