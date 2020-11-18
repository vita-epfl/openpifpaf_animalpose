#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 96G
#SBATCH --time 06:00:00
#SBATCH --account=vita
#SBATCH --gres gpu:1

pattern=$1
shift

srun singularity exec --bind /scratch/izar --nv ../pytorch_latest.sif \
  /bin/bash -c "source ../.venv/apollo/bin/activate && find outputs/ -name \"$pattern\" -exec python3 -m openpifpaf.eval --checkpoint {} $(printf "%s " "$@") \;"
