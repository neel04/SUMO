#!/bin/bash
#SBATCH --job-name Comma2k19_download   ## name that will show up in the queue
#SBATCH --output slurm-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --partition=cpu16
#SBATCH --time=99:09:42  ## time for analysis (day-hour:min:sec)
#SBATCH --mail-user neelgupta04@outlook.com  ## your email address
singularity shell -B /fsx/awesome/:/home/awesome docker://p3terx/aria2-pro
cd /fsx/awesome/comma2k19_dataset  
echo "Downloading Comma2k19 dataset"
aria2c https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5ddb.torrent
echo "Download complete"