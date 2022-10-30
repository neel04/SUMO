#!/bin/bash

#SBATCH --job-name SUMO_training   ## name that will show up in the queue
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --output slurm-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --partition=gpu
#SBATCH -t 7-00:00  # time limit: (D-HH:MM)
#SBATCH --comment laion
#SBATCH --exclude gpu-st-p4d-24xlarge-[66,106,141,144,225,324,347-350]

## Load modules
TF_CPP_MIN_LOG_LEVEL=0 #logging
grep MemTotal /proc/meminfo
lscpu
nvidia-smi
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=0
export JOBLIB_TEMP_FOLDER=/tmp
singularity exec --nv --cleanenv -B /fsx/awesome:/home/awesome tensorflow_latest.sif \
python3 scripts/convnext_baseline.py --model_name='Semi_fp16' --batch_size=128 --epochs=10 \
--lr_base_512=3e-5
#python3 ./keras_cv_attention_models/train_script.py --seed 0 --batch_size 1 -s CoAtNet0 \
#-d 'bdd100k:1.0.0' --rescale_mode='tf ' --num_layers=0 --mixup_alpha=0 --cutmix_alpha=0 \
#--summary -m efficientnet.EfficientNetV1B0 --input_shape=224 --random_crop_min=0 --magnitude=1 --eval_central_crop=0