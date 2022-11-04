pwd;
echo Port @ $PORT
export WANDB_API_KEY=618e11c734b0f6069af4735cde3d3d515930d678
cd /fsx/awesome/Openpilot-Deepdive;  \
python3 main.py --tqdm True --batch_size 6 --name 3e-4_NoWD_ConvNext_Small_8 --lr 3e-4 \
--optimizer lamb --resume=True --n_workers 8 \
--model convnext_small_in22k 
echo FINSIHED