#!/bin/bash
#SBATCH --job-name SUMO_train   ## name that will show up in the queue
#SBATCH --exclusive
#SBATCH --output slurm-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --partition=gpu
#SBATCH -t 7-72:00:00  # time limit: (D-HH:MM)
#SBATCH --comment laion
#SBATCH --ntasks-per-node=1
#SBATCH --exclude gpu-st-p4d-24xlarge-[290]
#SBATCH --requeue # requeue the job if preempted

# Training on a slurm cluster
export DIST_PORT=23333  # You may use whatever you want
export PORT=$DIST_PORT
export NUM_GPUS=8
export PARTITION=gpu
export WANDB_API_KEY=618e11c734b0f6069af4735cde3d3d515930d678

cd /fsx/awesome/
echo "Starting training" 
echo $NUM_GPUS $PARTITION

srun -p "$PARTITION" --job-name=openpilot -n $NUM_GPUS --gres=gpu:$NUM_GPUS --comment laion \
singularity exec --nv -B /fsx/awesome:/fsx/awesome singularity_base_container.sif bash /fsx/awesome/sbatch_files/sumo_executor.sh 
echo "All Singularity Commands executed"