#!/bin/bash -l

#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem=60GB
#SBATCH --partition=m1tb,m4tb

module load python
module load texlive/2021
source ../env/bin/activate

# Train the model
srun -u python -u runalgo_klr.py $1

