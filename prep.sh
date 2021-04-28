#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 40
#SBATCH --mem 96G
#SBATCH --time 1:00:00
#SBATCH --account=vita

module load gcc python cuda
source ../.venv/animal/bin/activate
python -m openpifpaf_animalpose.voc_to_coco
