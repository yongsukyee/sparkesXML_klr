#!/bin/bash -l

#SBATCH --cpus-per-task=28
#SBATCH --time=3:00:00
#SBATCH --mem=60GB
###SBATCH --partition=h2

module load python
module load texlive/2021
source env/bin/activate

# Train the model
srun -u python -u runalgo_klr.py

