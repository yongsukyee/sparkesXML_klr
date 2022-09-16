#!/bin/bash -l

#SBATCH --time=2:00:00
#SBATCH --mem=60GB
###SBATCH --partition=h2
#SBATCH --array=1-50
#SBATCH --job-name=simplepulse-multi

module load python
module load texlive/2021
source ../env/bin/activate

files=(/datasets/work/mlaifsp-sparkes/work/sparkesX/multi/${SLURM_JOB_NAME%-multi}/*.sf)
[[ $files =~ * ]] && exit 1
# Train the model
python -u post_process.py ${files[SLURM_ARRAY_TASK_ID-1]}
