#!/bin/bash

#SBATCH --job-name BDD100K_preprocessing   ## name that will show up in the queue
#SBATCH --output slurm-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --partition=compute-od-gpu
#SBATCH --time=99:09:42  ## time for analysis (day-hour:min:sec)
#SBATCH --mail-user neelgupta04@outlook.com  ## your email address
#SBATCH --mail-type BEGIN  ## slurm will email you when your job starts
#SBATCH --mail-type END  ## slurm will email you when your job ends
#SBATCH --mail-type FAIL  ## slurm will email you when your job fails

## Load modules
CUDA_VISIBLE_DEVICES=-1 #no GPUs
TF_CPP_MIN_LOG_LEVEL=0 #logging
grep MemTotal /proc/meminfo
lscpu
nvidia-smi
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=0
export JOBLIB_TEMP_FOLDER=/tmp
pip3 install joblib psutil
singularity exec -B /fsx/awesome/temp/:/fsx/awesome/temp/ tfio_modified.sif python3 ./scripts/Final_converter.py