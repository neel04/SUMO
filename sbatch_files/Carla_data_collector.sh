#!/bin/bash

#SBATCH --job-name Carla_data_collection   ## name that will show up in the queue
#SBATCH --exclusive
#SBATCH --output slurm-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --partition=gpu
#SBATCH -t 7-72:00:00  # time limit: (D-HH:MM)
#SBATCH --comment laion
#SBATCH --ntasks-per-node=1

export WANDB_API_KEY=618e11c734b0f6069af4735cde3d3d515930d678
# Running the data collection script
nvcc --version
nvidia-smi
singularity shell --writable-tmpfs -B /fsx/awesome:/home/awesome --nv /fsx/awesome/definition_files/new_carla_img.sif
cd /fsx/awesome/carla-roach/
# Conda init to initialize bash for conda 
conda init bash
bash
conda activate carla
./run/data_collect_bc.sh