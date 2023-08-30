import logging
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "8888"

import torch.distributed as dist
from placement_utils.utils import *
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
from layout_placement.config.config import Config
from layout_placement.dataset.dataset import PlacementDataset
from layout_placement.trainer.trainer import Trainer

if __name__ == "__main__":
    logging.info(f"{torch.__version__}")
    logger = Logger()
    config = Config()

    if config.args.use_wandb and (config.args.local_rank == -1 or config.args.local_rank == 0):
        wandb.init(project='placement-model', name=config.args.run_description)

    logging.info(f"start training placement model!!!! local_rank={config.args.local_rank}")
    set_seed(config.args.seed)

    if config.args.distribute and config.args.local_rank != -1:
        assert torch.cuda.device_count() > config.args.local_rank
        torch.cuda.set_device(config.args.local_rank)
        device = torch.device("cuda", index=config.args.local_rank)
        world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=config.args.local_rank
        )
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1

    train_set = PlacementDataset(config, split="train")
    valid_set = PlacementDataset(config, split="val")

    # prepare data sampler
    if config.args.distribute and config.args.local_rank != -1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=config.args.local_rank)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=config.args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.args.batch_size,
        shuffle=False,
        num_workers=config.args.num_workers
    )

    if config.args.local_rank == 0 or config.args.local_rank == -1:
        logging.info(f"datasets loaded, train: {len(train_set)}, valid: {len(valid_set)}")

    trainer = Trainer(config, device, world_size)
    trainer.train(train_loader, valid_loader)