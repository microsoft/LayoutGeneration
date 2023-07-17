import os
import shutil
from typing import Dict

import wandb
import torch
import deepspeed
from torch.utils.data import DataLoader, DistributedSampler

from utils import utils
from .basic_trainer import Trainer
from .utils import CheckpointMeasurement


class DSTrainer(Trainer):

    TRAINER_NAME = 'deepspeed'

    def _setup_experiment(self, task_name: str, task_config: Dict):
        deepspeed.init_distributed(dist_backend=self.args.backend)
        self._local_rank = int(os.environ['LOCAL_RANK'])
        self._world_size = int(os.environ['WORLD_SIZE'])
        print(f'World size: {self._world_size}, Local rank: {self._local_rank}, Main Process: {self._is_main_process}')
        self._ds_config = utils.load_arguments(self.args.deepscale_config)
        self._normal_ckpt_path = os.path.join(self.args.out_dir, 'checkpoint.pth.tar')

        # setup
        if self._is_main_process:
            task_config.update({
                'warmup_steps': self._ds_config['scheduler']['params']['warmup_num_steps']
            })
            super()._setup_experiment(task_name, task_config)
            shutil.copy(self.args.deepscale_config, os.path.join(self.args.out_dir, 'ds_config.json'))
        torch.distributed.barrier()

    @property
    def _is_main_process(self):
        return self._local_rank in [-1, 0]

    def _load_ckpt(self):
        _, client_state = self.model_engine.load_checkpoint(self.train_ckpt_path, load_optimizer_states=False,
                                                            load_lr_scheduler_states=False, load_module_only=True,
                                                            tag=self.args.ds_ckpt_tag)  # model engine
        print(client_state)

    def _setup_model(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(args=self.args, model=self.model,
                                                                       model_parameters=params,)
        self.device = self.model_engine.local_rank

    def _setup_dataloader(self):

        assert self._ds_config['train_micro_batch_size_per_gpu'] == self.args.batch_size
        assert self._ds_config['gradient_accumulation_steps'] == self.args.gradient_accumulation

        self.train_sampler = DistributedSampler(dataset=self.train_dataset, num_replicas=self._world_size)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.args.batch_size,
                                           collate_fn=self.collate_fn,
                                           sampler=self.train_sampler)
        print(len(self.train_dataloader), len(self.train_dataset))
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.args.eval_batch_size,
                                         collate_fn=self.collate_fn)

    def __call__(self, train_step, eval_step, eval_interval=1):
        self.ckpt_measurement.reset()
        gold_layouts = []
        global_step = 0
        for epoch in range(self.args.epoch):
            self.train_sampler.set_epoch(epoch)
            self.model_engine.train()
            for batch_idx, data in enumerate(self.train_dataloader):
                loss = train_step(self.model_engine, data, self.tokenizer, self.device)
                self.model_engine.backward(loss)
                self.model_engine.step()

                if global_step % self.args.train_log_step == 0 and self._is_main_process:
                    wandb.log({'training_loss': loss}, step=global_step)
                    print('\t'.join([
                        f'[{epoch}/{self.args.epoch}][{batch_idx}/{len(self.train_dataloader)}]',
                        f'Loss: {loss.item():.3f}'
                    ]))
                global_step += 1

            # Special checkpoint in Deepspeed
            self.model_engine.save_checkpoint(save_dir=self._normal_ckpt_path,
                                              client_state={'epoch': epoch + 1})

            if self._is_main_process and (epoch == (self.args.epoch - 1) or (epoch + 1) % eval_interval == 0):
                self.model_engine.eval()
                self.init_val_metrics()
                pred_layouts = list()
                with torch.no_grad():
                    for data in self.val_dataloader:
                        metrics, out = eval_step(self.model_engine, data, self.seq_processor, self.tokenizer, self.device)
                        self.aggregate_metrics(metrics)

                        pred_bboxes = out['pred_bboxes']
                        pred_labels = out['gold_labels'] if self.is_label_condition else out['pred_labels']
                        mask = out['mask']

                        pred_layouts.extend(self.collect_layouts(pred_bboxes, pred_labels, mask))

                        if epoch == 0:
                            gold_labels, gold_bboxes = out['gold_labels'], out['gold_bboxes']
                            gold_layouts.extend(self.collect_layouts(gold_bboxes, gold_labels, mask))

                self.logging_metrics(epoch, global_step)

                if self.is_debug:
                    self.save_val_output(gold_layouts, pred_layouts, data['out_str'], out['out_str'])

                # checkpoint measure
                if self.ckpt_measurement == CheckpointMeasurement.ACCURACY:
                    measure = self.val_bbox_acc
                else:
                    measure = self.ckpt_measurement.compute(gold_layouts, pred_layouts)
                is_best = self.ckpt_measurement.update(measure)
                wandb.log(
                    {f'checkpoint_measure-{self.ckpt_measurement.measurement}': measure}, step=global_step)
                print(f'Checkpoint Measure-{self.ckpt_measurement.measurement}: {measure:.3f}')
                self.do_checkpointing(epoch, is_best)

            torch.distributed.barrier()

    def do_checkpointing(self, epoch, is_best):
        if is_best:
            best_ckpt_path = os.path.join(self.args.out_dir, 'model_best.pth.tar')
            if os.path.exists(best_ckpt_path):
                # Only keep on best checkpoint (Save memory)
                shutil.rmtree(best_ckpt_path)
            shutil.copytree(self._normal_ckpt_path, best_ckpt_path)

        if epoch != self.args.epoch - 1:
            shutil.rmtree(self._normal_ckpt_path)
