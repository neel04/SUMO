from glob import glob
import os
os.environ.setdefault('OMP_NUM_THREADS', '6')
os.environ["NCCL_DEBUG"] = "INFO"

import time
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio

import torch
import torch.distributed as dist
import timm
import wandb
import argparse

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import Accuracy
from timm.optim.optim_factory import create_optimizer_v2
from torchinfo import summary

# ======================================================================================================================
# Initialization
# ======================================================================================================================

# Disable all GPUS
tf.config.set_visible_devices([], 'GPU')
visible_devices = tf.config.get_visible_devices()
for device in visible_devices:
    assert device.device_type != 'GPU'

parser = argparse.ArgumentParser()

parser.add_argument('--model_name', type=str, default='convnext', help='model')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--pretrained', type=str, default='None', help='pretrained model path')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')

parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for batch_size=512, lr = lr_base_512 * 512 / batch_size")
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer')
parser.add_argument('--log_frequency', type=int, default=50, help='log frequency')
parser.add_argument('--val_frequency', type=int, default=1, help='eval frequency')

parser.add_argument('--input_shape', type=int, default=[256, 256], help='input shape')
parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
parser.add_argument('--group_name', type=str, default=None, help='group name for WandB')
parser.add_argument('--wandb_id', type=str, default='696969', help='WandB run ID')
parser.add_argument('--resume', type=bool, default=False, help='resume training')
parser.add_argument('--drop_rate', type=float, default=0.3, help='dropout rate')

# print all the arguments
args = parser.parse_args()
print(args)

args.group_name = args.model_name if args.group_name is None else args.group_name

# =================================================================
# Helpper functions
# =================================================================
def dist_fn(idx, x, rank, total_nodes):
    '''
    Function to distribute different batches of data to different GPUs
    '''
    rank += total_nodes 
    
    if idx % rank == 0:
        return True
    else:
        return False

class tfds_ds(torch.utils.data.IterableDataset):
    '''
    Creates a torch Dataset from a tfds.Dataset
    '''
    def __init__(self, subset, nodes, rank):
        self.subset = subset
        self.total_nodes = nodes
        self.rank = rank

        self.train_test_ratio = 5000 # number of shards to remove from training and add to test

        if self.subset == 'train':
            self.num_shards = dataset.info.splits[self.subset].num_shards - self.train_test_ratio
        else:
            self.num_shards = dataset.info.splits[self.subset].num_shards
            self.mini_train_bs = self.train_test_ratio // self.total_nodes

        self.mini_batch_size = self.num_shards // self.total_nodes
        self.len = (self.num_shards * 963) // args.batch_size // self.total_nodes # number of batches to consume

        assert self.subset is not None
 
        if self.subset == 'train':
            split_string = f'{self.subset}[{self.mini_batch_size * self.rank}shard:{(self.mini_batch_size * (self.rank + 1))}shard]'
        else:
            #split_string = f'{self.subset}[{self.mini_batch_size * self.rank}shard:{self.mini_batch_size * (self.rank + 1)}shard]+train[{47736-self.train_test_ratio}shard:]'
            initial_cutoff = 47736 - self.train_test_ratio
            split_string = f'{self.subset}[{self.mini_batch_size * self.rank}shard:{self.mini_batch_size * (self.rank + 1)}shard]+train[{(self.mini_train_bs * self.rank)+initial_cutoff}shard:{initial_cutoff+(self.mini_train_bs * (self.rank + 1))}shard]'

        print(f'Loading {self.subset} dataset from {split_string} | total shards: {self.num_shards}')
        self.ds = dataset.as_dataset(
            split=split_string,
            as_supervised=True, 
            ).unbatch().batch(args.batch_size, num_parallel_calls=args.num_workers
            ).apply(tf.data.experimental.ignore_errors()
            ).take(self.len - 100).prefetch(32)
        
        self.dataset = tfds.as_numpy(self.ds)

        del self.ds # free memory
    
    def to_ten(self, tensor):
        return torch.from_numpy(tensor)

    def __len__(self) -> torch.Tensor:
        return self.mini_batch_size * 963 # 963 is hardcoded prebatched tensor (hence why .unbatch)

    def __iter__(self):
        for image, label in self.dataset:
            yield self.to_ten(image), self.to_ten(label)

class wandb_logger():
    def __init__(self, args):
        self.wandb_args = {'id':args.wandb_id,'job_type':'train','entity': 'neel', 'name': args.model_name, 'config': args, 'magic': True, 'group':args.group_name, 'project': 'SUMO', 'resume': 'allow'}
        self.rank = None
    
    def setup(self, rank):
        if rank == 0:
            print(f'Initialization with W&B on rank {rank}')
            self.rank = rank
            wandb.init(**self.wandb_args)
    
    def log(self, obj, idx=None):
        if self.rank == 0 and obj.get('model_name') is not None: # means we want to log a table, not model metrics
            print(f'Logging table: {obj}\n')
            wandb.Table(rows=list(obj.values()), columns=list(obj.keys()))
        elif self.rank is not None and idx is not None: # check if wandb has been init
            return wandb.log(obj, step=idx)
        elif self.rank is not None:
            return wandb.log(obj)
        else:
            return None

    def save(self, path):
        if self.rank is not None:
            return wandb.save(path)
        else:
            return None

def save_model(model_to_save, model_save_path, logger):
    '''
    DDP save the model to the given path
    '''
    # Save the model to the given path
    torch.save(model_to_save.state_dict(), model_save_path)
    # log model to wandb
    logger.save(model_save_path)

#==============================
# MODEL SETUP
#==============================
input_context = tf.distribute.InputContext(
    input_pipeline_id=0,  # Worker id
    num_input_pipelines=16,  # Total number of workers
)
read_config = tfds.ReadConfig(
    input_context=input_context,
)

#dataset = tfds.load(name='dataset_bdd100k', data_dir='s3://s-laion/ssd-videos/', as_supervised=True, read_config=read_config)
# train = 47736 | test = 5303
#TODO: Fix test sharding
dataset = tfds.builder_from_directory("s3://s-laion/ssd-videos/dataset_bdd100k/1.0.0/")

def main():
    # SHS = 4
    # initializing WandDB
    logger = wandb_logger(args)
    # Init on rank 0
    logger.setup(rank)

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    
    chkp_list = glob(f'./checkpoints/{args.model_name}*.pth')
    # chkp contains a list of all checkpoints in the current directory
    # like: ['nfnet0_1.pt', 'nfnet0_2.pt', 'nfnet0_3.pt']
    # we want to load the last checkpoint, which is the one with the highest number
    if len(chkp_list) > 0 and args.resume is True:
        # load the last checkpoint
        chkp = max(chkp_list, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        print(f'\n{"$"*50}\nRESUMING from checkpoint\nList:{chkp_list}\nChosen:{chkp}\n{"$"*50}')
    else:
        chkp = ''

    # check if args.model_name is starts with efficientnet
    # in which case we use model_args with drop_rate
    if args.model_name.startswith('efficientnet'):
        model_args = {'model_name':args.model_name, 'pretrained':args.pretrained, 'num_classes':51, 'in_chans':6, 'checkpoint_path':chkp, 'drop_rate':args.drop_rate}
    else:
        model_args = {'model_name':args.model_name, 'pretrained':args.pretrained, 'num_classes':51, 'in_chans':6, 'checkpoint_path':chkp}

    #logger.log(model_args)
    model = DDP(
        timm.create_model(
            **model_args
        ).to(device_id), device_ids=[device_id])

    # log model summary
    model_summary = summary( model, input_shape=(args.batch_size, *args.input_shape, 6) )
    print(model_summary)

    optimizer = create_optimizer_v2(
        lr=args.lr, opt=args.optimizer, weight_decay=args.weight_decay, momentum=0.000001,
        model_or_params=model)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-1)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-04)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.00097, verbose=False)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.47, verbose=False)

    # Creating the datasets
    #nodes = os.environ['COUNT_NODE'] #set by SLURM script

    train_loader = tfds_ds('train', nodes, rank)
    val_loader = tfds_ds('test', nodes, rank)

    torch.backends.cudnn.benchmark = True

    # Boring stuff
    loss_function = torch.nn.CrossEntropyLoss().to(device_id) # loss function

    train_metric_accuracy = Accuracy(compute_on_cpu=True).to(device_id)
    val_metric_accuracy = Accuracy(compute_on_cpu=True).to(device_id)
    val_top_k = Accuracy(compute_on_cpu=True, top_k=5).to(device_id)

    # AMP
    scaler = GradScaler()

    # Training loop
    for epoch in range(args.epochs):
        print('Starting loop')

        # Training loop
        model.train()
        for idx, batch in enumerate(train_loader):
            #Timing each step
            start = time.time()
            optimizer.zero_grad()

            # obtaining the data
            inputs, targets = batch[0], batch[1]           
            inputs, targets = inputs.to(device_id, non_blocking=True), targets.to(device_id, non_blocking=True)
            # converting BHWC TO BCHW for PyTorch
            inputs = inputs.permute(0, 3, 1, 2).float()

            with autocast(dtype=torch.float16):
                logits = model(inputs)
                loss = loss_function(logits, targets)
            
            # scale loss
            scaler.scale(loss).backward()
            # unscale gradients
            scaler.unscale_(optimizer)
            # clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            
            # stepping through scheduler and AMP's scaler
            # check if the scheduler is ReduceLROnPlateau
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(loss)
            else:
                scheduler.step()
            
            scaler.step(optimizer)
            scaler.update()

            if idx % args.log_frequency == 0:
                accuracy = train_metric_accuracy(logits, torch.argmax(targets, dim=1))
                com_lr =  scheduler._last_lr[0] if isinstance(scheduler._last_lr, list) else scheduler.last_lr
                print('Epoch: {}, Step: {}, Loss: {} , Acc: {} | Time taken: {} | shapes: {},{} | LR: {}'.format(epoch, idx, loss.item(), accuracy, time.time() - start, inputs.shape, targets.shape, com_lr))
                logger.log({'acc': accuracy, 'Epoch': epoch, 'train_loss': loss.item(), 'time_per_n_step': time.time() - start, 'lr':com_lr}, idx=idx)
                
                del inputs, targets, logits, loss, accuracy # free memory
        
        print(f'------\nTraining Finished\n------')
        # Computing the overall training accuracy 
        total_train_accuracy = train_metric_accuracy.compute()
        
        print(f"\n{'='*50}\nTraining acc for epoch {epoch}: {total_train_accuracy}\n{'='*50}")
        logger.log({'epoch_end_train_acc': total_train_accuracy, 'Epoch': epoch})
        
        # Validation loop
        if epoch % args.val_frequency == 0:
            model.eval()
            for idx, batch in enumerate(val_loader):
                with torch.no_grad():
                    # Disabling gradient computation
                    inputs, targets = batch[0].to(device_id), batch[1].to(device_id)
                    # converting BHWC TO BCHW for PyTorch
                    inputs = inputs.permute(0, 3, 1, 2).float()

                    logits = model(inputs)

                    loss = loss_function(logits, targets)

                    val_metric_accuracy.update(logits, torch.argmax(targets, dim=1))
                    val_top_k.update(logits, torch.argmax(targets, dim=1))
                    
                    if idx % args.log_frequency == 0:
                        logger.log({'val_acc': val_metric_accuracy, f'Val_Top-{val_top_k.top_k}':val_top_k ,'Epoch': epoch, 'val_loss': loss.item()}, idx=idx)
                        print(f'val_acc: {val_metric_accuracy} | val_top-{val_top_k.top_k}: {val_top_k} | val_loss: {loss.item()}')
                        
                        del inputs, targets, logits, loss # free memory

        # Calculate validation metrics
        total_val_accuracy = val_metric_accuracy.compute()
        print(f"\n{'-'*50}\nValidation acc for epoch {epoch}: {total_val_accuracy}\n{'-'*50}")
        logger.log({'final_val_acc': total_val_accuracy, f'final_top_{val_top_k.top_k}_acc':val_top_k.compute() ,'Epoch': epoch})
        
        # Save checkpoint to Wandb
        if epoch % 2 == 0 and rank == 0:
            chkp_path = f'./checkpoints/{args.model_name}_{epoch}.pth'
            save_model(model, chkp_path, logger)

        # Reset metric for next epoch
        del total_train_accuracy, total_val_accuracy # free memory

        train_metric_accuracy.reset()
        val_metric_accuracy.reset()
        val_top_k.reset()

if __name__ == '__main__':
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    nodes = dist.get_world_size()

    print(f"Start running SUMO w/ DDP @ {rank} | World size: {nodes}")

    main()