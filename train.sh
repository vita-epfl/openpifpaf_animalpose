#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 96G
#SBATCH --time 48:00:00
#SBATCH --account=vita
#SBATCH --gres gpu:2

module load gcc python cuda
source ../.venv/animal/bin/activate
srun /bin/bash -c "time python3 -m torch.distributed.launch --nproc_per_node=4 -m openpifpaf.train $(printf "%s " "$@")"
