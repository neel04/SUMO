import os
# Set OMP_NUM_THREADS to 6
os.environ['OMP_NUM_THREADS'] = '6'
import time
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_io as tfio
import torch
import timm
import argparse

from accelerate import Accelerator
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
parser.add_argument('--pretrained', type=str, default='imagenet', help='pretrained model path')
parser.add_argument('--epochs', type=int, default=2, help='number of epochs')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')

parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for batch_size=512, lr = lr_base_512 * 512 / batch_size")
parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer')
parser.add_argument('--log_frequency', type=int, default=50, help='log frequency')
parser.add_argument('--val_frequency', type=int, default=1, help='eval frequency')

parser.add_argument('--input_shape', type=int, default=[256, 256], help='input shape')
parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
parser.add_argument('--group_name', type=str, default=None, help='group name for WandB')

# print all the arguments
args = parser.parse_args()
print(args)

args.group_name = args.model_name if args.group_name is None else args.group_name

# Convert wandb_args a dict
# project="SUMO", entity="neel", name=args.model_name, config=args, magic=True
wandb_args = {"wandb":{'entity': 'neel', 'name': args.model_name, 'config': args, 'magic': True, 'group':args.group_name}}

accelerator = Accelerator(log_with='wandb')
device = accelerator.device

accelerator.init_trackers("SUMO", config=args, init_kwargs=wandb_args)

# Helpful funcs
def save_model(accelerator, model_to_save, model_save_path):
  state = accelerator.get_state_dict(model_to_save) # This will call the unwrap model as well
  accelerator.save(state, model_save_path)

class tfds_ds(torch.utils.data.IterableDataset):
    '''
    Creates a torch Dataset from a tfds.Dataset
    '''
    def __init__(self, subset, nodes, rank):
        self.subset = subset
        self.total_nodes = nodes
        self.rank = rank
        self.num_shards = dataset.info.splits[self.subset].num_shards
        self.len = (self.num_shards * 963) // args.batch_size // self.total_nodes # number of batches to consume

        assert self.subset is not None
 
        self.mini_batch_size = self.num_shards // self.total_nodes

        split_string = f'{self.subset}[{self.mini_batch_size * self.rank}shard:{self.mini_batch_size * (self.rank + 1)}shard]'

        self.ds = dataset.as_dataset(
            split=split_string,
            as_supervised=True, 
            ).unbatch().batch(args.batch_size, num_parallel_calls=args.num_workers
            ).apply(tf.data.experimental.ignore_errors()
            ).prefetch(32).take(self.len - 100) #).enumerate().filter(lambda i,x: dist_fn(i, x, self.rank, self.total_nodes)
        
        self.dataset = tfds.as_numpy(self.ds)

        del self.ds # free memory
    
    def to_ten(self, tensor):
        return torch.from_numpy(tensor)

    def __len__(self) -> torch.Tensor:
        return self.mini_batch_size * 963 # 963 is hardcoded prebatched tensor (hence why .unbatch)

    def __iter__(self):
        for image, label in self.dataset:
            yield self.to_ten(image), self.to_ten(label)

input_context = tf.distribute.InputContext(
    input_pipeline_id=0,  # Worker id
    num_input_pipelines=16,  # Total number of workers
)
read_config = tfds.ReadConfig(
    input_context=input_context,
)
#dataset = tfds.load(name='dataset_bdd100k', data_dir='s3://s-laion/ssd-videos/', as_supervised=True, read_config=read_config)
# train = 47736 | test = 5303
dataset = tfds.builder_from_directory("s3://s-laion/ssd-videos/dataset_bdd100k/1.0.0/")

def main():
    model = timm.create_model(args.model_name, pretrained=args.pretrained, num_classes=51, in_chans=6).to(device)
    # log model summary
    model_summary = summary( model, input_shape=(args.batch_size, *args.input_shape, 6) )
    accelerator.print(model_summary)

    optimizer = create_optimizer_v2(lr=args.lr, opt=args.optimizer, weight_decay=args.weight_decay, model_or_params=model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-1)
    loss_function = torch.nn.CrossEntropyLoss().to(device) # loss function

    # Creating the datasets
    nodes = accelerator.num_processes
    rank = accelerator.process_index

    accelerator.print(f'nodes: {nodes}, rank: {rank}')

    train_loader = tfds_ds('train', nodes, rank)
    val_loader = tfds_ds('test', nodes, rank)

    # Accelerate
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)

    train_metric_accuracy = Accuracy().to(device)
    val_metric_accuracy = Accuracy().to(device)
    val_top_k = Accuracy(top_k=5).to(device)
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
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            # converting BHWC TO BCHW for PyTorch
            inputs = inputs.permute(0, 3, 1, 2).float()
            logits = model(inputs)

            loss = loss_function(logits, targets)
            accelerator.backward(loss)

            optimizer.step()
            scheduler.step()

            if idx % args.log_frequency == 0:
                accuracy = train_metric_accuracy(logits, torch.argmax(targets, dim=1))
                accelerator.print('Epoch: {}, Step: {}, Loss: {} , Acc: {} | Time taken: {}'.format(epoch, idx, loss.item(), accuracy, time.time() - start))
                accelerator.log({'acc': accuracy, 'Epoch': epoch, 'train_loss': loss.item(), 'time_per_n_step': time.time() - start}, step=idx)

                del inputs, targets, logits, loss # free memory

        total_train_accuracy = train_metric_accuracy.compute()
        accelerator.print(f"\n{'='*50}\nTraining acc for epoch {epoch}: {total_train_accuracy}\n{'='*50}")
        accelerator.log({'epoch_end_train_acc': total_train_accuracy, 'Epoch': epoch})
        
        # Validation loop
        if epoch % args.val_frequency == 0:
            model.eval()
            for idx, batch in enumerate(val_loader):
                with torch.no_grad():
                    # Disabling gradient computation
                    inputs, targets = batch[0].to(device), batch[1].to(device)
                    # converting BHWC TO BCHW for PyTorch
                    inputs = inputs.permute(0, 3, 1, 2).float()

                    logits = model(inputs)

                    loss = loss_function(logits, targets)

                    val_metric_accuracy.update(logits, torch.argmax(targets, dim=1))
                    val_top_k.update(logits, torch.argmax(targets, dim=1))
                    
                    if idx % args.log_frequency == 0:
                        accelerator.log({'val_acc': val_metric_accuracy, f'Val_Top-{val_top_k.top_k}':val_top_k ,'Epoch': epoch, 'val_loss': loss.item()}, step=idx)
                        accelerator.print(f'val_acc: {val_metric_accuracy} | val_top-{val_top_k.top_k}: {val_top_k} | val_loss: {loss.item()}')

        # Calculate validation metrics
        total_val_accuracy = val_metric_accuracy.compute()
        accelerator.print(f"\n{'-'*50}\nValidation acc for epoch {epoch}: {total_val_accuracy}\n{'-'*50}")
        accelerator.log({'final_val_acc': total_val_accuracy, f'final_top_{val_top_k.top_k}_acc':val_top_k.compute() ,'Epoch': epoch})
        
        # Save checkpoint to Wandb
        if epoch % 2 == 0:
            chkp_path = f'./checkpoints/{args.model_name}_{epoch}.pth'
            save_model(accelerator, model, chkp_path)

        # Reset metric for next epoch
        train_metric_accuracy.reset()
        val_metric_accuracy.reset()
        val_top_k.reset()

if __name__ == '__main__':
    # Executing everything
    main()
    accelerator.end_training()