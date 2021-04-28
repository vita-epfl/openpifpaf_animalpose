#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 96G
#SBATCH --time 10:00:00
#SBATCH --account=vita
#SBATCH --gres gpu:1

module load gcc python cuda
source ../.venv/animal/bin/activate
srun /bin/bash -c "python3 -m openpifpaf_animalpose.voc_to_coco"
