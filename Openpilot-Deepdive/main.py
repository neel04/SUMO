from glob import glob
import os
import time
import wandb
from tqdm import tqdm
from argparse import ArgumentParser

from timm.optim.optim_factory import create_optimizer_v2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from shampoo_opt.shampoo import Shampoo

if torch.__version__ == 'parrots':
    print(f'Using Parrots version {torch.__version__}')
    from pavi import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter

from data import PlanningDataset, SequencePlanningDataset, Comma2k19SequenceDataset
from model import PlaningNetwork, MultipleTrajectoryPredictionLoss, SequencePlanningNetwork
from utils import draw_trajectory_on_ax, get_val_metric, get_val_metric_keys

def seed_everything(seed=42):
    '''
    Here, we seed everything to ensure reproducibility
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed) # Numpy module

def get_hyperparameters(parser):
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--log_per_n_step', type=int, default=20)
    parser.add_argument('--val_per_n_epoch', type=int, default=1)

    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--model', type=str, default='convnext_tiny_in22k', help='Name of the model to use for TIMM')

    parser.add_argument('--M', type=int, default=5)
    parser.add_argument('--num_pts', type=int, default=33)
    parser.add_argument('--mtp_alpha', type=float, default=1.0)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--sync_bn', type=bool, default=True)
    parser.add_argument('--tqdm', type=bool, default=False)
    parser.add_argument('--optimize_per_n_step', type=int, default=40)
    parser.add_argument('--name', type=str, default='OPX_EffNet_baseline')
    parser.add_argument('--wandb_group', type=str, default='Comma2k19_runs')

    try:
        exp_name = os.environ["SLURM_JOB_ID"]
    except KeyError:
        exp_name = str(time.time())
    parser.add_argument('--exp_name', type=str, default=exp_name)

    return parser


def setup(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', init_method='tcp://localhost:%s' % os.environ['PORT'], rank=rank, world_size=world_size)
    print('[%.2f]' % time.time(), 'DDP Initialized at %s:%s' % ('localhost', os.environ['PORT']), rank, 'of', world_size, flush=True)


def get_dataloader(rank, world_size, batch_size, pin_memory=False, num_workers=0):
    train = Comma2k19SequenceDataset('data/comma2k19_train_non_overlap.txt', 'data/comma2k19/','train', use_memcache=False)
    val = Comma2k19SequenceDataset('data/comma2k19_val_non_overlap.txt', 'data/comma2k19/','demo', use_memcache=False)

    if torch.__version__ == 'parrots':
        dist_sampler_params = dict(num_replicas=world_size, rank=rank, shuffle=True)
    else:
        dist_sampler_params = dict(num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    train_sampler = DistributedSampler(train, **dist_sampler_params)
    val_sampler = DistributedSampler(val, **dist_sampler_params)

    loader_args = dict(num_workers=num_workers, persistent_workers=True if num_workers > 0 else False, prefetch_factor=2, pin_memory=pin_memory)
    train_loader = DataLoader(train, batch_size, sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val, batch_size=1, sampler=val_sampler, **loader_args)

    return train_loader, val_loader


def cleanup():
    dist.destroy_process_group()

class SequenceBaselineV1(nn.Module):
    def __init__(self, model_name, M, num_pts, mtp_alpha, lr, optimizer, optimize_per_n_step=40) -> None:
        super().__init__()
        self.M = M
        self.num_pts = num_pts
        self.mtp_alpha = mtp_alpha
        self.lr = lr
        self.optimizer = optimizer

        self.net = SequencePlanningNetwork(model_name, M, num_pts)

        self.optimize_per_n_step = optimize_per_n_step  # for the gru module

    @staticmethod
    def configure_optimizers(args, model):
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.01)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
        elif args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.002)
        elif args.optimizer == 'shampoo':
            optimizer = Shampoo(model.parameters(), lr=args.lr,momentum=0.99)
        else:
            try:
                optimizer = create_optimizer_v2(
                    lr=args.lr, opt=args.optimizer, weight_decay=0.00001, momentum=0.4,
                    model_or_params=model)
            except:
                raise NotImplementedError
        #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 20, 0.9)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-1)
        print('Using CosineAnnealing\nUsing optimizer:', optimizer)

        return optimizer, lr_scheduler

    def forward(self, x, epoch, hidden=None):
        if hidden is None:
            hidden = torch.zeros((2, x.size(0), 512)).to(self.device)
        return self.net(x, hidden, epoch)


def main(rank, world_size, args):
    if rank == 0:
        writer = SummaryWriter()
        # WandB setup
        wandb.init(project='SUMO', entity='neel', config=args, name=args.name, group=args.wandb_group, magic=True)
        print(f'Arguments parsed: {args}')
        wandb.run.save()
        wandb.save(os.path.join(wandb.run.dir, 'main.py'))
    
    train_dataloader, val_dataloader = get_dataloader(rank, world_size, args.batch_size, False, args.n_workers)
    
    model = SequenceBaselineV1(args.model, args.M, args.num_pts, args.mtp_alpha, args.lr, args.optimizer, args.optimize_per_n_step)
    
    use_sync_bn = args.sync_bn
    if use_sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda()
    optimizer, lr_scheduler = model.configure_optimizers(args, model)
    model: SequenceBaselineV1

    dist.barrier()
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
    loss = MultipleTrajectoryPredictionLoss(args.mtp_alpha, args.M, args.num_pts, distance_type='angle')

    if args.resume:
        # resuming mechanism, obtaining the latest checkpoint path
        # This is useful in a SLURM setting where we pre-empt multiple times
        # and want to resume from the latest checkpoint
        chkp_list = glob(f'/fsx/awesome/comma2k19_checkpoints/{args.model}*.pth')

        if len(chkp_list) > 0:
            # We want to obtain the file last modified in chkp_list
            chkp_file = max(chkp_list, key=os.path.getctime) # Get the latest file
            checkpoint = torch.load(chkp_file, map_location=f'cuda:{rank}')
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print(f'\n{"==="*25}\nLoaded checkpoint from {chkp_file}\n{"==="*25}')  

    num_steps = 0
    disable_tqdm = (not args.tqdm) or (rank != 0)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in tqdm(range(args.epochs), disable=disable_tqdm, position=0):
        train_dataloader.sampler.set_epoch(epoch)
        model.train()
        
        for batch_idx, data in enumerate(tqdm(train_dataloader, leave=False, disable=disable_tqdm, position=1)):
            seq_inputs, seq_labels = data['seq_input_img'].cuda(), data['seq_future_poses'].cuda()
            bs = seq_labels.size(0)
            seq_length = seq_labels.size(1)
            
            hidden = torch.zeros((2, bs, 512)).cuda()
            total_loss = 0
            for t in tqdm(range(seq_length), leave=False, disable=disable_tqdm, position=2):
                num_steps += 1
                inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
                
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    pred_cls, pred_trajectory, hidden = model(inputs, epoch, hidden)

                cls_loss, reg_loss = loss(pred_cls, pred_trajectory, labels)
                total_loss += (cls_loss + args.mtp_alpha * reg_loss.mean()) / model.module.optimize_per_n_step
            
                if rank == 0 and (num_steps + 1) % args.log_per_n_step == 0:
                    # Pushing metrics to WandB
                    wandb.log({
                        'train/epoch': epoch,
                        'loss/cls': cls_loss,
                        'loss/reg': reg_loss.mean(),
                        'loss/reg_x': reg_loss[0],
                        'loss/reg_y': reg_loss[1],
                        'loss/reg_z': reg_loss[2],
                        'param/lr': optimizer.param_groups[0]['lr'],
                    }, step=num_steps)

                if (t + 1) % model.module.optimize_per_n_step == 0:
                    hidden = hidden.clone().detach()
                    optimizer.zero_grad()
                    #total_loss.backward()
                    scaler.scale(total_loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # TODO: move to args
                    #optimizer.step()
                    scaler.step(optimizer)
                    if rank == 0:
                        wandb.log({'loss/total': total_loss}, step=num_steps)
                    scaler.update()
                    total_loss = 0

            if not isinstance(total_loss, int):
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # TODO: move to args
                optimizer.step()
                if rank == 0:
                    wandb.log({'loss/total': total_loss}, step=num_steps)

        lr_scheduler.step()
        if (epoch + 1) % args.val_per_n_epoch == 0:
            if rank == 0:
                # save mode
                ckpt_path = '/fsx/awesome/comma2k19_checkpoints/%s_%d.pth' % (args.model, epoch)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'loss': total_loss,
                }, ckpt_path)

                time.sleep(10) # wait for all processes to save the model
                wandb.save(ckpt_path)
                print('[Epoch %d] checkpoint saved at %s' % (epoch, ckpt_path))

            model.eval()
            # delete useless variables that take up GPU memory
            del seq_inputs, seq_labels, inputs, labels, pred_cls, pred_trajectory, hidden, total_loss, cls_loss, reg_loss
            torch.cuda.empty_cache()

            with torch.no_grad():
                saved_metric_epoch = get_val_metric_keys()
                for batch_idx, data in enumerate(tqdm(val_dataloader, leave=False, disable=disable_tqdm, position=1)):
                    seq_inputs, seq_labels = data['seq_input_img'].cuda(), data['seq_future_poses'].cuda()

                    bs = seq_labels.size(0)
                    seq_length = seq_labels.size(1)
                    
                    hidden = torch.zeros((2, bs, 512), device=seq_inputs.device)
                    for t in tqdm(range(seq_length), leave=False, disable=True, position=2):
                        inputs, labels = seq_inputs[:, t, :, :, :], seq_labels[:, t, :, :]
                        pred_cls, pred_trajectory, hidden = model(inputs, epoch, hidden)

                        metrics = get_val_metric(pred_cls, pred_trajectory.view(-1, args.M, args.num_pts, 3), labels)
                        
                        for k, v in metrics.items():
                            saved_metric_epoch[k].append(v.float().mean().item())
                
                dist.barrier()  # Wait for all processes
                # sync
                metric_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.float32, device='cuda')
                counter_single = torch.zeros((len(saved_metric_epoch), ), dtype=torch.int32, device='cuda')
                # From Python 3.6 onwards, the standard dict type maintains insertion order by default.
                # But, programmers should not rely on it.
                for i, k in enumerate(sorted(saved_metric_epoch.keys())):
                    metric_single[i] = np.mean(saved_metric_epoch[k])
                    counter_single[i] = len(saved_metric_epoch[k])

                metric_gather = [torch.zeros((len(saved_metric_epoch), ), dtype=torch.float32, device='cuda')[None] for _ in range(world_size)]
                counter_gather = [torch.zeros((len(saved_metric_epoch), ), dtype=torch.int32, device='cuda')[None] for _ in range(world_size)]
                dist.all_gather(metric_gather, metric_single[None])
                dist.all_gather(counter_gather, counter_single[None])

                if rank == 0:
                    metric_gather = torch.cat(metric_gather, dim=0)  # [world_size, num_metric_keys]
                    counter_gather = torch.cat(counter_gather, dim=0)  # [world_size, num_metric_keys]
                    metric_gather_weighted_mean = (metric_gather * counter_gather).sum(0) / counter_gather.sum(0)
                    for i, k in enumerate(sorted(saved_metric_epoch.keys())):
                        wandb.log({k: metric_gather_weighted_mean[i]}, step=num_steps)
                dist.barrier()

            model.train()

    cleanup()


if __name__ == "__main__":
    print('[%.2f]' % time.time(), 'starting job...', os.environ['SLURM_PROCID'], 'of', os.environ['SLURM_NTASKS'], flush=True)
    seed_everything(420) # Seeding
    parser = ArgumentParser()
    parser = get_hyperparameters(parser)
    args = parser.parse_args()

    setup(rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['SLURM_NTASKS']))
    main(rank=int(os.environ['SLURM_PROCID']), world_size=int(os.environ['SLURM_NTASKS']), args=args)