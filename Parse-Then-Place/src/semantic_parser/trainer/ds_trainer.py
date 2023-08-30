# coding=utf8

import logging
import os
import shutil
from typing import Callable, Dict, List

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers.optimization import Adafactor

import wandb
from semantic_parser.metrics import Metric
from utils import file_utils

logger = logging.getLogger(__name__)


class DSTrainer:

    TRAINER_NAME = 'deepspeed'

    def __init__(self,
                 project_name: str,
                 experiment_config: Dict,
                 args,
                 model,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 collate_fn: Callable,
                 metrics: List[Metric],
                 ckpt_metric: str = 'loss',
                 optimizer = None):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collate_fn = collate_fn
        self.output_dir = args.output_dir

        # Metrics
        self.metrics = metrics
        self.ckpt_metric = ckpt_metric
        self.is_ckpt_metric_positive = True
        for m in metrics:
            if m.METRIC_NAME == ckpt_metric:
                self.is_ckpt_metric_positive = m.is_positive

        self._normal_ckpt_path = os.path.join(self.output_dir, 'checkpoint')
        self._eval_interval = self.args.eval_delay

        logger.info("Setup experiment")
        self._setup_experiment(project_name, experiment_config)
        self._setup_model(optimizer)
        self._setup_dataloader()

    def _log_hyperparameters(self):
        micro_batch_size = self._ds_config['train_micro_batch_size_per_gpu']
        gradient_accumulation = self._ds_config.get('gradient_accumulation_steps', 1)
        real_batch_size = micro_batch_size * gradient_accumulation * self._world_size
        config = {
            'seed': self.args.seed,
            'epoch': self.args.num_epochs,
            'gradient_accumulation_steps': gradient_accumulation,
            'micro_batch_size': micro_batch_size,
            'batch_size': real_batch_size,
            'gradient_clip': self._ds_config.get('gradient_clipping', 0.0),
            'label_smoothing': self.args.label_smoothing_factor
        }
        if self.args.use_adafactor:
            config['optimizer'] = 'AdaFactor'
            config['learning_rate'] = self.args.adafactor_lr
        else:
            config.update({
                'optimizer': self._ds_config['optimizer']['type'],
                'learning_rate': self._ds_config['optimizer']['params']['lr'],
                'scheduler': self._ds_config['scheduler']['type'],
                'warmup_steps': self._ds_config['scheduler']['params']['warmup_num_steps']
            })
        return config

    def _setup_experiment(self, project_name: str, experiment_config: Dict):
        deepspeed.init_distributed(dist_backend=self.args.backend)
        self._local_rank = int(os.environ['LOCAL_RANK'])
        self._world_size = int(os.environ['WORLD_SIZE'])
        logger.info(f'World size: {self._world_size}, Local rank: {self._local_rank}, Main Process: {self._is_main_process}')
        self._ds_config = file_utils.read_json(self.args.deepscale_config)

        # setup
        if self._is_main_process:
            if not os.path.exists(self.output_dir):
                file_utils.makedirs(self.output_dir)
            hypara_config = self._log_hyperparameters()
            hypara_config.update({
                'trainer': self.TRAINER_NAME
            })
            hypara_config.update(experiment_config)
            wandb.init(project=project_name, config=hypara_config)
            shutil.copy(self.args.deepscale_config, os.path.join(self.output_dir, 'ds_config.json'))
        dist.barrier()

    @property
    def _is_main_process(self):
        return self._local_rank in [-1, 0]

    def _setup_model(self, optimizer = None):
        if optimizer is not None:
            logger.info("Overwrite optimizer specified in config.json")
            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(args=self.args, model=self.model,
                                                                           optimizer=optimizer)
        else:
            parameters = self.model.parameters()
            if self.args.use_adafactor:
                optimizer = Adafactor([{'params': parameters}], scale_parameter=False, relative_step=False,
                                    warmup_init=False, lr=self.args.adafactor_lr, weight_decay=1e-5)
            else:
                optimizer = None
            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(args=self.args, model=self.model,
                                                                           model_parameters=parameters,
                                                                           optimizer=optimizer)
        self.device = self.model_engine.local_rank

    def _setup_dataloader(self):
        micro_batch_size = self._ds_config['train_micro_batch_size_per_gpu']
        eval_micro_batch_size = self.args.eval_micro_batch_size
        self.train_sampler = DistributedSampler(dataset=self.train_dataset, num_replicas=self._world_size)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=micro_batch_size,
                                           collate_fn=self.collate_fn,
                                           sampler=self.train_sampler)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=eval_micro_batch_size,
                                         collate_fn=self.collate_fn)

    def _aggreate_metrics(self, result: Dict) -> None:
        for metric in self.metrics:
            metric.aggregate(result)

    def _collect_metrics(self, reset: bool = False) -> Dict:
        results = dict()
        for metric in self.metrics:
            results.update(metric.compute())
            if reset:
                metric.reset()
        return results

    def __call__(self, train_step, eval_step):
        ckpt_metric_best_value = -1e+8 if self.is_ckpt_metric_positive else np.inf
        global_step = 0
        for epoch in range(self.args.num_epochs):
            self.train_sampler.set_epoch(epoch)
            self.model_engine.train()
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss = train_step(self.model_engine, batch, self.device)
                self.model_engine.backward(loss)
                self.model_engine.step()

                if global_step % self.args.train_log_step == 0 and self._is_main_process:
                    wandb.log({'training_loss': loss}, step=global_step)
                    logger.info('\t'.join([
                        f'[{epoch}/{self.args.num_epochs}][{batch_idx}/{len(self.train_dataloader)}]',
                        f'Loss: {loss.item():.3f}'
                    ]))
                global_step += 1

            # Special checkpoint in Deepspeed
            if epoch == (self.args.num_epochs - 1) or (epoch + 1) % self.args.save_interval == 0:
                self.model_engine.save_checkpoint(save_dir=self._normal_ckpt_path,
                                              client_state={'epoch': epoch + 1})

            if self._is_main_process and (epoch == (self.args.num_epochs - 1) or (epoch + 1) % self._eval_interval == 0):
                self.model_engine.eval()
                with torch.no_grad():
                    for data in self.val_dataloader:
                        batch_metrics, _ = eval_step(self.model_engine, data, self.device)
                        self._aggreate_metrics(batch_metrics)

                metrics = self._collect_metrics(reset=True)
                wandb.log(metrics, step=global_step)
                logger.info('\t'.join([f"{k}: {v:.3f}" for k, v in metrics.items()]))

                is_best = False
                ckpt_metric_value = metrics[self.ckpt_metric]
                if self.is_ckpt_metric_positive and ckpt_metric_value > ckpt_metric_best_value:
                    is_best = True
                elif not self.is_ckpt_metric_positive and ckpt_metric_value < ckpt_metric_best_value:
                    is_best = True
                self.do_checkpointing(epoch, is_best)

            dist.barrier()

    def do_checkpointing(self, epoch, is_best):
        if is_best:
            with open(os.path.join(self.output_dir, 'best_epoch'), 'w') as f:
                f.write(f"{epoch}")
