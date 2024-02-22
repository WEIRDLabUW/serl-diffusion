#!/bin/bash

#SBATCH --job-name=jacob-experiment
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=violinhockey@gmail.com

#SBATCH --account=weirdlab
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --ntasks=1

#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

#SBATCH --chdir=/gscratch/weirdlab/jacob33/serl-diffusion
#SBATCH --export=all
#SBATCH --output=slurm/%j-out.txt   # where STDOUT goes
#SBATCH --error=slurm/%j-err.txt    # where STDERR goes

# Activate Anaconda work environment
source /mmfs1/home/jacob33/miniconda3/bin/activate
conda activate /gscratch/weirdlab/jacob33/environments/serl-diffusion

srun python train_script.py