#!/bin/bash

# HOSTNAMES MASTER_ADDR MASTER_PORT COUNT_NODE are coming from the main script
# Script runs the distributed training of the model

## Distributed script arguments
export WANDB_API_KEY=618e11c734b0f6069af4735cde3d3d515930d678
export CUDA_MODULE_LOADING=LAZY

echo myuser=`whoami`
echo COUNT_NODE=$COUNT_NODE
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $HOSTNAMES
echo hostname = `hostname`
echo MASTER_ADDR= $MASTER_IP
echo MASTER_PORT= $MASTER_PORT

H=`hostname`
THEID=`echo -e $HOSTNAMES  | python3 -c "import sys;[sys.stdout.write(str(i)) for i,line in enumerate(next(sys.stdin).split(' ')) if line.strip() == '$H'.strip()]"`
echo THEID=$THEID

#source ~/.bashrc
#conda activate SUMO_dist
chmod +x /home/awesome/awesome/scripts/torch_convnext.py

## ---------
## FP16 DOES NOT WORK! DOWNSCALAE BATCH_SIZE EVERY F*CKING  TIME
## ---------
export OMP_NUM_THREADS=6
export NCCL_DEBUG=INFO

echo Warning: Not all GPUs are being used, and FP16 disabled
echo $THEID Starting

# Using DDP script here
singularity exec --nv --cleanenv -B /fsx/awesome:/home/awesome torch_cuda_11_7.sif \
\
torchrun --max_restarts=3 --nproc_per_node=6 --nnodes=$COUNT_NODE --node_rank=$THEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT  \
scripts/ddp_convnext.py --model_name='resnet50' --batch_size=480 --epochs=100 \
--num_workers 2 --lr=5e-3 --optimize='LAMB' --weight_decay=0.04 --group_name='ResNets'  \
--pretrained=imagenet --wandb_id=hoiu34 --resume=True #ConvNext_Baselines Effnet_baselines 6e-5
