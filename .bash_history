ls
python3 scripts/convnext_baseline.py 
python3 scripts/convnext_baseline.py 
pip3 install wandb
clear; python3 scripts/convnext_baseline.py 
pip3 install keras_cv_attention_models
clear; python3 scripts/convnext_baseline.py 
exit
clear; python3 scripts/convnext_baseline.py 
clear; python3 scripts/convnext_baseline.py 
clear; python3 scripts/convnext_baseline.py 
clear; python3 scripts/convnext_baseline.py 
clear; python3 scripts/convnext_baseline.py 
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext_baseline_small'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext_baseline_small'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext_baseline_small'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext_baseline_small'
clear
exit
exit
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext_baseline_small'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext_baseline_small'
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext_baseline_small' --batch_size=1024
exit
exit
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext_baseline_small' --batch_size=1024
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext_baseline_small' --batch_size=1024
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext_baseline_small' --batch_size=1024
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext_baseline_small' --batch_size=1024
exit
exit
exit
clear; python3 scripts/convnext_baseline.py  --model_name='Convnext_baseline_small' --batch_size=1024
exit
exit
accelerate launch --multi-gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=1024 --epochs=10 --lr=3e-5 --pretrained='imagenet'
accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=1024 --epochs=10 --lr=3e-5 --pretrained='imagenet'
clear; accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=1024 --epochs=10 --lr=3e-5 --pretrained='imagenet'
accelerate launch -h
clear; accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=512 --epochs=10 --lr=3e-5 --pretrained='imagenet'
clear; accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=512 --epochs=10 --lr=3e-5 --pretrained='imagenet'
exit
exit
clear; accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=512 --epochs=10 --lr=3e-5 --pretrained='imagenet'
exit
exit
2022-08-17 18:06:52.799374: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.801311: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.810669: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.812519: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.814489: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.816425: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.827577: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.827669: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.829490: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.831319: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.840794: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.842602: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.844451: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.846302: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.855787: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.857613: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.861447: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.866982: I tensorflow/stream_executor/cuda/cuda_driver.ls
clear
ls
nvidia-smi
2022-08-17 18:06:52.799374: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.801311: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.810669: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.812519: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.814489: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.816425: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.827577: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.827669: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.829490: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.831319: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.840794: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.842602: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.844451: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.846302: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.855787: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.857613: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.861447: I tensorflow/stream_executor/cuda/cuda_driver.cc:739] failed to allocate 2.2K (2304 bytes) from device: CUDA_ERROR_OUT_OF_MEMORY: out of memory
2022-08-17 18:06:52.866982: I tensorflow/stream_executor/cuda/cuda_driver.clear
accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=1024 --epochs=10 \
accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=128 --epochs=10 --lr=3e-5 --pretrained='imagenet'
nvidia-smi
exit
exit
accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=32 --epochs=10 --lr=3e-5 --pretrained='imagenet'
exit
exit
#==============================
# MODEL SETUP
#==============================
accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=1024 --epochs=10 --lr=3e-5 --pretrained='imagenet'
exit
exit
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=1024 --epochs=10 --lr=3e-5 --pretrained='imagenet'
nvitop
accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=512 --epochs=10 --lr=3e-5 --pretrained='imagenet'
accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=512 --epochs=10 --lr=3e-5 --pretrained='imagenet'
accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=512 --epochs=10 --lr=3e-5 --pretrained='imagenet'
accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=512 --epochs=10 --lr=3e-5 --pretrained='imagenet'
accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=512 --epochs=10 --lr=3e-5 --pretrained='imagenet'
nvitop
clear; accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=512 --epochs=10 --lr=3e-5 --pretrained='imagenet'
clear; accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=512 --epochs=10 --lr=3e-5 --pretrained='imagenet'
clear; accelerate launch --multi_gpu --mixed_precision=fp16 scripts/torch_convnext.py --model_name='convnext_small' --batch_size=512 --epochs=10 --lr=3e-5 --pretrained='imagenet'
sudo apt remove libnccl2 libnccl-dev
apt remove libnccl2 libnccl-dev
exit
exit
pwd
cd awesome
ls
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=8 scripts/torch_convnext.py --model_name='convnext_nano' --batch_size=80 --epochs=25 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='CNext_Nano'
exit
exit
accelerate env
exit
clear
ls
clear
accelerate launch --config_file='./child_config.yaml' scripts/torch_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=80 --epochs=25 --lr=7e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='No_Pretrained_Models_7e-5'
python3 scripts/torch_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=80 --epochs=25 --lr=7e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='No_Pretrained_Models_7e-5'
exit
ls
 hostname -I | awk '{print $1}'
accelerate config --config_file='/home/awesome/root_config.yaml'
accelerate launch --config_file='./root_config.yaml' scripts/torch_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=80 --epochs=25 --lr=7e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='No_Pretrained_Models_7e-5'
accelerate config --config_file='/home/awesome/root_config.yaml'
accelerate launch --config_file='./root_config.yaml' scripts/torch_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=80 --epochs=25 --lr=7e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='No_Pretrained_Models_7e-5'
accelerate launch --config_file='./root_config.yaml' --num_cpu_threads_per_process=48  scripts/torch_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=80 --epochs=25 --lr=7e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='No_Pretrained_Models_7e-5'
accelerate launch --config_file='./root_config.yaml' --num_cpu_threads_per_process=48
accelerate launch --config_file='./root_config.yaml' --num_cpu_threads_per_process=48 ./scripts/torch_convnext.py 
accelerate launch --debug --config_file='./root_config.yaml' --num_cpu_threads_per_process=48 ./scripts/torch_convnext.py 
accelerate launch --config_file='./root_config.yaml' --num_cpu_threads_per_process=48  scripts/torch_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=80 --epochs=25 --lr=7e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='No_Pretrained_Models_7e-5'
exit
accelerate launch --config_file='./child_config.yaml' scripts/torch_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=80 --epochs=25 --lr=7e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='No_Pretrained_Models_7e-5'
exit
accelerate launch --num_processes 16 --num_machines 2 --multi_gpu --mixed_precision "fp16" --machine_rank 0 --main_process_ip "172.31.232.116" --main_process_port 8888 scripts/torch_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=80 --epochs=25 \
accelerate launch --num_processes 16 --num_machines 2 --multi_gpu --mixed_precision "fp16" --machine_rank 0 --main_process_ip "172.31.232.116" --main_process_port 8888 scripts/torch_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=80 --epochs=25
ip a
ifconfig a
curl ifconfig.me
exit
clear
accelerate launch --num_processes 16 --num_machines 2 --multi_gpu --mixed_precision "fp16" --machine_rank 1 --main_process_ip "172.31.232.116" --main_process_port 8888 scripts/torch_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=80 --epochs=25 \
curl ifconfig.me
exit
python3 scripts/ddp_convnext.py 
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 scripts/ddp_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=40 --epochs=50 --num_workers 4 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='DDP_No_Pretrained'
exit
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 scripts/ddp_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=40 --epochs=50 --num_workers 4 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='DDP_No_Pretrained'
clear; torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 scripts/ddp_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=40 --epochs=50 --num_workers 4 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='DDP_No_Pretrained'
exit
clear; torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 scripts/ddp_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=40 --epochs=50 --num_workers 4 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='DDP_No_Pretrained'
exit
clear; torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 scripts/ddp_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=40 --epochs=50 --num_workers 4 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='DDP_No_Pretrained'
exit
clear; torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 scripts/ddp_convnext.py --model_name='convnext_xlarge_in22k' --batch_size=40 --epochs=50 --num_workers 4 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='DDP_No_Pretrained'
exit
accelerate launch --num_processes=8 \ --num_machines 1 --multi_gpu --mixed_precision fp16 --main_process_port 4444 scripts/torch_convnext.pyaccelerate launch --num_processes=8 \ --num_machines 1 --multi_gpu --mixed_precision fp16 --main_process_port 4444 scripts/torch_convnext.py
accelerate launch --num_processes=8 --num_machines 1 --multi_gpu --mixed_precision fp16 --main_process_port 4444 scripts/torch_convnext.py
clear
ls
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py
ip -a
ip a
ip
curl ifconfig.me
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py
exit
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py
exit
accelerate launch --config_file './child_config.yaml' scripts/torch_convnext.py
exit
accelerate launch --config_file './child_config.yaml' scripts/torch_convnext.py
accelerate launch --config_file './child_config.yaml' scripts/torch_convnex
accelerate launch --config_file './child_config.yaml' scripts/torch_convnext.py --model_name='convnext_base_in22k' --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
exit
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py --model_name='convnext_base_in22k--model_name='convnext_base_in22k' --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py --model_name='convnext_base_in22k' --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
exit
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py --model_name='convnext_base_in22k' --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
clear
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py --model_name='convnext_base_in22k' --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
exit
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py --model_name='convnext_base_in22k' --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
clear
accelerate launch --config_file './child_config.yaml' scripts/torch_convnext.py --model_name='convnext_base_in22k' --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
exit
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py --model_name='convnext_base_in22k' --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
clear
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py --model_name='convnext_base_in22k' --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
exit
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py --model_name='convnext_base_in22k' --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
exit
accelerate launch --config_file './child_config.yaml' scripts/torch_convnext.py --model_name='convnext_base_in22k' --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
clear
accelerate launch --config_file './child_config.yaml' scripts/torch_convnext.py --model_name='convnext_base_in22k' --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
exit
accelerate launch --config_file './root_config.yaml' scripts/torch_convnext.py --model_name='convnext_base_in22k' --num_workers 2 --lr=6e-5 --optimize='AdamW' --weight_decay=0.002 --group_name='ConvNext_Baselines'
exit
conda activate carla
exit
conda activate carla
conda inint
exit
conda activate carla
bash run/data_collect_b
bash run/data_collect_bc.sh 
ls
pwd
cd carla_gym/
cd utils/
ls
cat config_utils.py 
cd ..
ls
python3
bash run/data_collect_bc.sh 
pwd
ls
pwd
python3
ls
cat carla_gym/
cd carla_gym/
ls
cd ..
ls
cd carla_
cd carla_gym/
ls
cat __init__.py 
cd ..
clear
ls
./run/data_collect_bc.sh 
pwd
ls
python3
ls
ipython
clear
ls
pwd
export PYTHONPATH="$PYTHONPATH:/home/awesome/carla/carla-roach"
./run/data_collect_bc.sh 
echo PYTHONPATH="$PYTHONPATH:/home/awesome/carla/carla-roach"
echo $PYTHONPATH
vim ~/.bashrc
nano ~/.bashrc
exit
conda activate carla
exit
ls
pwd
cd carla/
ls
pwd
ls
bash carla-roach/run/data_collect_bc.sh 
pwd
ls
cd carla-roach/
ls
bash run/data_collect_bc.sh 
ls
pwd
conda activate carla
bash
conda init bash
bash
vim ~/.bashrc 
conda init bash
bash
exit
conda activate carla
./run/data_collect_bc.sh 
python3
ls
cd ..
ls
python34
python3
./run/data_collect_bc.sh 
./carla-roach/run/data_collect_bc.sh 
cd carla-roach/
./run/data_collect_bc.sh 
source ~/.bashrc
conda activate carla
./run/data_collect_bc.sh 
conda deactivate
conda activate carla
./run/data_collect_bc.sh 
exit
ls
cd carla-roach/
conda 
clear
ls
conda activate carla
conda init bash
bash
exit
conda activate varla
conda activate carla
pwd
ls
./carla-roach/run/data_collect_bc.sh 
pwd
cd carla-roach/
cd ..
python3
./carla-roach/run/data_collect_bc.sh 
cd carla-roach/
./run/data_collect_bc.sh 
easy_install ${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
exit
conda activate carla
conda init bash
bash
exit
conda activate carla
cd carla-roach/
ls
./run/data_collect_bc.sh 
easy_install /home/awesome/awesome/carla/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
exit
conda init bash
bash
easy_install /home/awesome/awesome/carla/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg
exit
ls
cd carla-roach/
ls
./run/data_collect_
./run/data_collect_bc.sh 
exit
python3
apt
apt-get install neofetch
sudo apt-get install neofetch
apt-get moo
lshw
lscpu
hostnamectl
cat /etc/os-release
sudo
exit
python3
sudo apt-get install libjpeg-turbo8
apt-get install libjpeg-turbo8
sudo -s
sudo -u
sudo
su
su apt-get install libjpeg-turbo8
su
apt-file search libjpeg.so.8
exit
conda activate carla
./run/data_collect_bc.sh~
./run/data_collect_bc.sh
python3
exit
[200~cd /fsx/awesome/carla-roach/
conda init bash
bash
conda deactivate
exit
cd /fsx/awesome/carla-roach/
conda init bash
bash
conda activate carla
./run/data_collect_bc.sh
exit
conda activate carla
./run/data_collect_bc.sh
exit
cd /fsx/awesome/carla-roach/
conda init bash
bash
exit
conda activate carla
cd carla-roach/
./run/data_collect_
./run/data_collect_bc.sh 
conda install carla
pip3 install carla
python3
exit
conda init bash
conda activate cartla
conda activate carla
conda init bash
bash
exit
cd carla-roach/
ls
./run/data_collect_bc.sh 
nvcc
nvidia-smi
exit
bash
exit
conda activate cartla
conda activate carla
conda init bash
bash
exit
bash
exit
conda activate carla
ls
cd carla-roach/
./run/data_collect_bc.sh 
exit
cd carla-roach/
./run/data_collect_bc.sh 
pip3 install h5py
./run/data_collect_bc.sh 
exit
cd carla-roach/
./run/data_collect_bc.sh 
exit
cd carla-roach/
./run/data_collect_
./run/data_collect_bc.sh 
exit
cd carla-roach/
./run/data_collect_bc.sh 
exit
conda activate carla
./run/data_collect_bc.sh 
singularity
exit
cd carla-roach/
./run/data_collect_bc.sh 
conda init bash
bash
exit
cd carla-roach/
./run/data_collect_bc.sh 
exit
