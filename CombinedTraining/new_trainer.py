import os, sys, math, time, random, datetime, functools
import lpips
import numpy as np
from pathlib import Path
from loguru import logger
from copy import deepcopy
from omegaconf import OmegaConf
from collections import OrderedDict
from einops import rearrange
from contextlib import nullcontext

from datapipe.datasets import create_dataset

from utils import util_net
from utils import util_common
from utils import util_image

import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.utils.data as udata
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.utils as vutils
from torch.nn.parallel import DistributedDataParallel as DDP

class TrainerBase:
    def __init__(self, configs):
        self.configs = configs

        # Setup distributed training: self.num_gpus, self.rank
        self.setup_dist()

        # Setup seed
        self.setup_seed()

        # Initialize logger
        self.init_logger()

    def setup_dist(self):
        self.num_gpus = torch.cuda.device_count()

        if self.num_gpus > 1:
            if mp.get_start_method(allow_none=True) is None:
                mp.set_start_method('spawn', force=True)
            self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
            self.rank = int(os.environ.get('RANK', 0))
            self.world_size = int(os.environ.get('WORLD_SIZE', 1))

            # Set GPU device
            torch.cuda.set_device(self.local_rank)

            # Initialize the process group
            dist.init_process_group(
                timeout=datetime.timedelta(seconds=3600),
                backend='nccl',
                init_method='env://',
            )
        else:
            # Single GPU or CPU
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1

    def setup_seed(self, seed=None, global_seeding=None):
        if seed is None:
            seed = self.configs.train.get('seed', 12345)
        if global_seeding is None:
            global_seeding = self.configs.train.global_seeding
            assert isinstance(global_seeding, bool)
        if not global_seeding:
            seed += self.rank
            torch.cuda.manual_seed(seed)
        else:
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def init_logger(self):
        if self.configs.resume:
            assert self.configs.resume.endswith(".pth")
            save_dir = Path(self.configs.resume).parents[1]
            project_id = save_dir.name
        else:
            project_id = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_dir = Path(self.configs.save_dir) / project_id
            if not save_dir.exists() and self.rank == 0:
                save_dir.mkdir(parents=True)

        # Text logging
        log_text_path = save_dir / 'training.log'
        if self.rank == 0:
            if log_text_path.exists():
                assert self.configs.resume
            self.logger = logger
            self.logger.remove()
            self.logger.add(log_text_path, format="{message}", mode='a', level='INFO')
            self.logger.add(sys.stdout, format="{message}")

        # Checkpoint saving
        self.ckpt_dir = save_dir / 'ckpts'
        if self.rank == 0 and not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir()

        # Logging the configurations
        if self.rank == 0:
            self.logger.info(OmegaConf.to_yaml(self.configs))

    def build_dataloader(self):
        def _wrap_loader(loader):
            while True: yield from loader

        # Make datasets
        datasets = {'train': create_dataset(self.configs.data.get('train', dict)), }
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            datasets['val'] = create_dataset(self.configs.data.get('val', dict))
        if self.rank == 0:
            for phase in datasets.keys():
                length = len(datasets[phase])
                self.logger.info('Number of images in {:s} data set: {:d}'.format(phase, length))

        # Make dataloaders
        if self.num_gpus > 1:
            sampler = udata.distributed.DistributedSampler(
                datasets['train'],
                num_replicas=self.world_size,
                rank=self.rank,
            )
        else:
            sampler = None
        dataloaders = {'train': _wrap_loader(udata.DataLoader(
            datasets['train'],
            batch_size=self.configs.train.batch[0] // max(self.num_gpus, 1),
            shuffle=False if self.num_gpus > 1 else True,
            drop_last=True,
            num_workers=min(self.configs.train.num_workers, 4),
            pin_memory=True,
            prefetch_factor=self.configs.train.get('prefetch_factor', 2),
            worker_init_fn=my_worker_init_fn,
            sampler=sampler,
        ))}
        if hasattr(self.configs.data, 'val') and self.rank == 0:
            dataloaders['val'] = udata.DataLoader(datasets['val'],
                                                  batch_size=self.configs.train.batch[1],
                                                  shuffle=False,
                                                  drop_last=False,
                                                  num_workers=0,
                                                  pin_memory=True,
                                                  )

        self.datasets = datasets
        self.dataloaders = dataloaders
        self.sampler = sampler

    def build_model(self):
        """
        Builds and initializes the model. Loads weights if a checkpoint is specified.
        Wraps the model with DistributedDataParallel if necessary.
        """
        params = self.configs.model.get('params', dict)
    
        # Use the original method to get the model object dynamically
        self.model = util_common.get_obj_from_str(self.configs.model.target)(**params).cuda()
    
        # Load checkpoint if available
        if self.configs.model.ckpt_path and os.path.exists(self.configs.model.ckpt_path):
            ckpt = torch.load(self.configs.model.ckpt_path, map_location=f'cuda:{self.local_rank}')
            self.logger.info(f"Loaded model checkpoint from {self.configs.model.ckpt_path}")
            self.model.load_state_dict(ckpt)
    
        # Wrap the model with DistributedDataParallel if using multiple GPUs
        if self.num_gpus > 1:
            self.model = DDP(self.model, device_ids=[self.local_rank])
    


    def save_ckpt(self, iteration):
        if self.rank == 0:
            ckpt_path = self.ckpt_dir / f"ckpt_{iteration}.pth"
            torch.save({'state_dict': self.model.state_dict()}, ckpt_path)

    def validation(self):
        pass


class TrainerDifIR(TrainerBase):
    def prepare_data(self, data, dtype=torch.float32, phase='train'):
        """
        Prepares data for model input. Handles paired LR-HR data or single LR data
        and performs necessary device transfers, type conversions, and padding if required.
        """
        device = torch.device(f'cuda:{self.local_rank}')
        dataset_type = self.configs.data.get(phase, {}).get('type', None)
    
        if dataset_type == 'paired':
            # Process LR-HR paired data
            lq = data['lq'].to(device).to(dtype=dtype)
            gt = data['gt'].to(device).to(dtype=dtype)
    
            # Access in_channels dynamically
            model_in_channels = (
                self.model.module.in_channels if hasattr(self.model, 'module') else self.model.in_channels
            )
    
            if lq.shape[1] < model_in_channels:
                self.logger.warning(f"[{phase.upper()} DATA] Padding lq from {lq.shape[1]} to {model_in_channels} channels.")
                padding = torch.zeros(
                    (lq.size(0), model_in_channels - lq.size(1), *lq.shape[2:]),
                    device=lq.device, dtype=lq.dtype
                )
                lq = torch.cat([lq, padding], dim=1)
    
            # Log tensor properties
            if self.rank == 0:
                self.logger.info(f"[{phase.upper()} DATA] Key: lq, Shape: {lq.shape}, Type: {lq.dtype}, Device: {lq.device}")
                self.logger.info(f"[{phase.upper()} DATA] Key: gt, Shape: {gt.shape}, Type: {gt.dtype}, Device: {gt.device}")
    
            return {'lq': lq, 'gt': gt}
    
        elif dataset_type == 'single':
            # Process single LR data
            lq = data['lq'].to(device).to(dtype=dtype)
    
            # Access in_channels dynamically
            model_in_channels = (
                self.model.module.in_channels if hasattr(self.model, 'module') else self.model.in_channels
            )
    
            if lq.shape[1] < model_in_channels:
                self.logger.warning(f"[{phase.upper()} DATA] Padding lq from {lq.shape[1]} to {model_in_channels} channels.")
                padding = torch.zeros(
                    (lq.size(0), model_in_channels - lq.size(1), *lq.shape[2:]),
                    device=lq.device, dtype=lq.dtype
                )
                lq = torch.cat([lq, padding], dim=1)
    
            # Log tensor properties
            if self.rank == 0:
                self.logger.info(f"[{phase.upper()} DATA] Single LR Shape: {lq.shape}, Type: {lq.dtype}, Device: {lq.device}")
    
            return {'lq': lq}
    
        else:
            self.logger.error(f"Unsupported dataset type '{dataset_type}' for phase '{phase}'.")
            raise ValueError(f"Unsupported dataset type '{dataset_type}' for phase '{phase}'.")


    def training_step(self, data):
        lq = data['lq']
        gt = data['gt']

        # Generate timesteps for the diffusion process
        batch_size = lq.size(0)
        timesteps = torch.randint(
            low=0,
            high=self.configs.diffusion.params.steps,
            size=(batch_size,),
            device=lq.device,
        )

        # Forward pass with timesteps
        pred = self.model(lq, timesteps)
        loss = F.mse_loss(pred, gt)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.rank == 0:
            self.logger.info(f"Loss: {loss.item():.4f}")
        return loss

    def validation(self):
        if self.rank == 0:
            self.model.eval()
            with torch.no_grad():
                for data in self.dataloaders['val']:
                    lq = data['lq']
                    gt = data['gt']

                    pred = self.model(lq)
                    psnr = util_image.batch_PSNR(pred, gt)

                    self.logger.info(f"Validation PSNR: {psnr:.2f}")
            self.model.train()

    def train(self):
        self.build_model()
        self.build_dataloader()

        # Initialize optimizer and scaler for mixed precision
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.configs.train.lr)
        scaler = amp.GradScaler(enabled=self.configs.train.use_amp)

        # Training loop
        for iteration in range(self.configs.train.iterations):
            # Fetch the next batch
            data = next(self.dataloaders['train'])
            data = self.prepare_data(data)

            # Forward and backward pass
            with amp.autocast(enabled=self.configs.train.use_amp):
                loss = self.training_step(data)

            # Backward and optimizer step
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

            # Logging
            if iteration % self.configs.train.log_freq[0] == 0 and self.rank == 0:
                self.logger.info(f"Iteration {iteration}/{self.configs.train.iterations}, Loss: {loss.item()}")

            # Validation
            if iteration % self.configs.train.val_freq == 0 and self.rank == 0:
                self.validation()

            # Save checkpoints
            if iteration % self.configs.train.save_freq == 0:
                self.save_ckpt(iteration)

        # Final validation and checkpoint save
        if self.rank == 0:
            self.validation()
            self.save_ckpt('final')

def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

if __name__ == '__main__':
    # Entry point for testing or debugging
    pass
