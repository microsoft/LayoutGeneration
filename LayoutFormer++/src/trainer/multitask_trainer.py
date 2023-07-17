# coding=utf8

from typing import Dict, Callable
import os
import json
import numpy as np

import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, RandomSampler, BatchSampler, Sampler

from trainer.basic_trainer import Trainer
from trainer.ds_trainer import DSTrainer
from trainer.utils import CheckpointMeasurement
from model import LayoutTransformerTokenizer


class MultiTaskBatchSampler(BatchSampler):

    def __init__(self, sampler: Sampler[int], batch_size: int, drop_last: bool,
                 num_tasks: int, task_sample_weights: str = None) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self._num_tasks = num_tasks
        if task_sample_weights is not None:
            weights = np.array([float(w)
                               for w in task_sample_weights.split(',')])
            if len(weights) != self._num_tasks:
                raise ValueError(
                    "Task sample weights should be equivalent to num tasks")
            self._task_sample_probs = weights / weights.sum()
        else:
            self._task_sample_probs = np.ones(
                self._num_tasks) * (1 / self._num_tasks)
        print("Task Sample Probs: ", self._task_sample_probs)

    def _sample_task(self):
        task_id = np.random.choice(
            self._num_tasks, 1, p=self._task_sample_probs)[0]
        return task_id

    def __iter__(self):
        batch = []
        task_id = self._sample_task()
        for idx in self.sampler:
            _idx = task_id * len(self.sampler) + idx
            batch.append(_idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                task_id = self._sample_task()
        if len(batch) > 0 and not self.drop_last:
            yield batch


class MultiTaskTrainer(Trainer):

    def __init__(self, task_name: str, args, tokenizer, model, seq_processor, train_dataset: Dataset,
                 val_dataset: Dataset, optimizer, scheduler=None, is_label_condition=True, checkpoint_measure=None,
                 d2c_fn=None, is_debug=False, task_config: Dict = None, collate_fn: Callable = None,
                 single_task_per_batch: bool = False, single_task_per_batch_task_sample_weights: str = None,
                 save_vocab: bool = True):
        self._single_task_per_batch = single_task_per_batch
        self._single_task_per_batch_task_sample_weights = single_task_per_batch_task_sample_weights
        super().__init__(task_name, args, tokenizer, model, seq_processor, train_dataset, val_dataset,
                         optimizer, scheduler, is_label_condition, checkpoint_measure, d2c_fn, is_debug,
                         task_config, collate_fn)

        if save_vocab and isinstance(tokenizer, LayoutTransformerTokenizer):
            vocab_path = os.path.join(self.args.out_dir, "vocab.json")
            tokenizer.save_vocab(vocab_path)

    def _setup_dataloader(self):
        if torch.cuda.device_count() > 1:
            train_bsz = self.args.batch_size * torch.cuda.device_count()
            val_bsz = self.args.eval_batch_size * torch.cuda.device_count()
        else:
            train_bsz, val_bsz = self.args.batch_size, self.args.eval_batch_size

        if self._single_task_per_batch:
            print("Perform single task per batch")
            sampler = RandomSampler(self.train_dataset, generator=None)
            train_batch_sampler = MultiTaskBatchSampler(sampler, batch_size=train_bsz, drop_last=True,
                                                        num_tasks=self.train_dataset.num_tasks,
                                                        task_sample_weights=self._single_task_per_batch_task_sample_weights)
            self.train_dataloader = DataLoader(dataset=self.train_dataset, collate_fn=self.collate_fn,
                                               batch_sampler=train_batch_sampler)
        else:
            self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                               batch_size=train_bsz,
                                               collate_fn=self.collate_fn,
                                               drop_last=True, shuffle=True)

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=val_bsz,
                                         collate_fn=self.collate_fn,
                                         drop_last=True)

    TRAINER_NAME = 'basic_multitask'

    def __call__(self, train_step, eval_step, tasks, eval_interval: int = 1):

        task_best_miou = {tn: {'value': 0.0, 'epoch': 0} for tn in tasks}
        task_gold_layouts = {tn: list() for tn in tasks}
        task_measurement = CheckpointMeasurement(
            max_num_elements=self.args.max_num_elements,
            measurement=CheckpointMeasurement.MIOU)

        self.ckpt_measurement.reset()
        global_step = 0
        for epoch in range(self.args.epoch):
            self.model.train()
            for batch_idx, data in enumerate(self.train_dataloader):
                loss = train_step(self.model, data,
                                  self.tokenizer, self.device)
                loss = loss / self.gradient_accumulation
                loss.backward()

                # weights update
                if ((batch_idx + 1) % self.gradient_accumulation == 0) or (batch_idx + 1 == len(self.train_dataloader)):
                    if self.enable_clip_gradient:
                        torch.nn.utils.clip_grad_norm_(
                            parameters=self.model.parameters(), max_norm=self.clip_gradient)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()

                if global_step % self.args.train_log_step == 0:
                    wandb.log({'training_loss': loss}, step=global_step)
                    print('\t'.join([
                        f'[{epoch}/{self.args.epoch}][{batch_idx}/{len(self.train_dataloader)}]',
                        f'Loss: {loss.item():.3f}'
                    ]))
                global_step += 1

            normal_ckpt_path = os.path.join(
                self.args.out_dir, f'epoch_{epoch}_checkpoint.pth.tar')
            torch.save(self.model.state_dict(), normal_ckpt_path)

            if (epoch == (self.args.epoch - 1) or (epoch + 1) % eval_interval == 0):
                self.model.eval()
                with torch.no_grad():
                    task_miou, eval_pred = dict(), dict()
                    eval_task_loss, eval_loss, num_eval_steps = dict(), 0.0, 0
                    for task in tasks:
                        add_gold_layouts = len(task_gold_layouts[task]) == 0
                        print(f"Evaluate on {task}")
                        self.val_dataset.switch_task(task)
                        dataloader = DataLoader(dataset=self.val_dataset,
                                                batch_size=self.args.eval_batch_size,
                                                collate_fn=self.collate_fn)
                        eval_task_loss[task] = {
                            'loss': 0.0, 'count': 0
                        }
                        eval_pred[task] = list()

                        for data in tqdm(dataloader):
                            num_eval_steps += 1
                            eval_step_loss, eval_step_pred = eval_step(self.model, data,
                                                                       self.val_dataset.seq_processor,
                                                                       self.tokenizer, self.device)
                            eval_loss += eval_step_loss.item()

                            # task loss
                            eval_task_loss[task]['loss'] += eval_step_loss.item()
                            eval_task_loss[task]['count'] += 1

                            if eval_step_pred is None:
                                continue

                            # collect predicted layouts
                            eval_pred[task].extend(self.collect_layouts(eval_step_pred['pred_bboxes'], eval_step_pred['pred_labels'],
                                                                        eval_step_pred['pred_mask']))

                            # collect gold layouts
                            if add_gold_layouts:
                                task_gold_layouts[task].extend(self.collect_layouts(eval_step_pred['gold_bboxes'], eval_step_pred['gold_labels'],
                                                                                    eval_step_pred['gold_mask']))

                        # compute loss & mIoU
                        eval_task_loss[task] = eval_task_loss[task]['loss'] / \
                            eval_task_loss[task]['count']
                        if len(eval_pred[task]) > 0 and len(task_gold_layouts[task]) > 0:
                            task_miou[task] = task_measurement.compute(
                                task_gold_layouts[task], eval_pred[task])

                    # Measure loss & mIoU
                    if num_eval_steps == 0:
                        num_eval_steps = 1
                    eval_loss = eval_loss / num_eval_steps

                    # log
                    log_dict = {
                        'epoch': epoch,
                        'eval_loss': eval_loss,
                    }
                    log_dict.update(
                        {f'{tn}_loss': v for tn, v in eval_task_loss.items()})
                    log_dict.update(
                        {f'{tn}_miou': v for tn, v in task_miou.items()})
                    wandb.log(log_dict, step=global_step)
                    print('\t'.join([
                        f'[{epoch}/{self.args.epoch}]',
                        f'Eval Loss: {eval_loss:.3f}',
                    ]))
                    for task, task_loss in eval_task_loss.items():
                        print(f'{task} Loss: {task_loss:.3f}')
                        if task in task_miou:
                            print(f'{task} mIoU: {task_miou[task]:.3f}')

                    # save the best iou
                    self.update_best_miou(epoch, task_miou, task_best_miou)

                measure = eval_loss
                is_best = self.ckpt_measurement.update(measure)
                wandb.log(
                    {f'checkpoint_measure-{self.ckpt_measurement.measurement}': measure}, step=global_step)
                print(
                    f'Checkpoint Measure-{self.ckpt_measurement.measurement}: {measure:.3f}')
                self.do_checkpointing(epoch, is_best, task_best_miou)

    def update_best_miou(self, epoch, curr_miou, best_miou):
        for task, value in curr_miou.items():
            if value > best_miou[task]['value']:
                best_miou[task]['value'] = value
                best_miou[task]['epoch'] = epoch

    def do_checkpointing(self, epoch, is_best, task_best_miou):
        # Save checkpoint
        if is_best:
            with open(os.path.join(self.args.out_dir, 'best_epoch'), 'w') as f:
                f.write(f"{epoch}")
            with open(os.path.join(self.args.out_dir, 'best_miou.json'), 'w') as f:
                f.write(json.dumps(task_best_miou))


class DSMultiTaskTrainer(DSTrainer):
    TRAINER_NAME = 'deepspeed_multitask'

    def __init__(self, task_name: str, args, tokenizer, model, seq_processor, train_dataset: Dataset, val_dataset: Dataset,
                 optimizer, scheduler=None, is_label_condition=True, checkpoint_measure=None, d2c_fn=None, is_debug=False,
                 task_config: Dict = None, collate_fn: Callable = None, save_vocab: bool = True):
        super().__init__(task_name, args, tokenizer, model, seq_processor, train_dataset, val_dataset, optimizer,
                         scheduler, is_label_condition, checkpoint_measure, d2c_fn, is_debug, task_config, collate_fn)

        if self._is_main_process and save_vocab and isinstance(tokenizer, LayoutTransformerTokenizer):
            vocab_path = os.path.join(self.args.out_dir, "vocab.json")
            tokenizer.save_vocab(vocab_path)

    def __call__(self, train_step, eval_step, tasks, eval_interval: int = 1):

        task_gold_layouts = {tn: list() for tn in tasks}
        task_measurement = CheckpointMeasurement(
            max_num_elements=self.args.max_num_elements,
            measurement=CheckpointMeasurement.MIOU)

        self.ckpt_measurement.reset()
        global_step = 0
        for epoch in range(self.args.epoch):
            self.train_sampler.set_epoch(epoch)
            self.model_engine.train()
            for batch_idx, data in enumerate(self.train_dataloader):
                loss = train_step(self.model_engine, data,
                                  self.tokenizer, self.device)
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
            self.model_engine.save_checkpoint(save_dir=self._normal_ckpt_path, tag=f'epoch_{epoch}',
                                              client_state={'epoch': epoch})

            if self._is_main_process and (epoch == (self.args.epoch - 1) or (epoch + 1) % eval_interval == 0):
                self.model_engine.eval()
                with torch.no_grad():
                    task_miou, eval_pred = dict(), dict()
                    eval_task_loss, eval_loss, num_eval_steps = dict(), 0.0, 0
                    for task in tasks:
                        add_gold_layouts = len(task_gold_layouts[task]) == 0
                        print(f"Evaluate on {task}")
                        self.val_dataset.switch_task(task)
                        dataloader = DataLoader(dataset=self.val_dataset,
                                                batch_size=self.args.eval_batch_size,
                                                collate_fn=self.collate_fn)
                        eval_task_loss[task] = {
                            'loss': 0.0, 'count': 0
                        }
                        eval_pred[task] = list()

                        for data in tqdm(dataloader):
                            num_eval_steps += 1
                            eval_step_loss, eval_step_pred = eval_step(self.model_engine, data,
                                                                       self.val_dataset.seq_processor,
                                                                       self.tokenizer, self.device)
                            eval_loss += eval_step_loss.item()

                            # task loss
                            eval_task_loss[task]['loss'] += eval_step_loss.item()
                            eval_task_loss[task]['count'] += 1

                            if eval_step_pred is None:
                                continue

                            # collect predicted layouts
                            eval_pred[task].extend(self.collect_layouts(eval_step_pred['pred_bboxes'], eval_step_pred['pred_labels'],
                                                                        eval_step_pred['pred_mask']))

                            # collect gold layouts
                            if add_gold_layouts:
                                task_gold_layouts[task].extend(self.collect_layouts(eval_step_pred['gold_bboxes'], eval_step_pred['gold_labels'],
                                                                                    eval_step_pred['gold_mask']))

                        # compute loss & mIoU
                        eval_task_loss[task] = eval_task_loss[task]['loss'] / \
                            eval_task_loss[task]['count']
                        if len(eval_pred[task]) > 0 and len(task_gold_layouts[task]) > 0:
                            task_miou[task] = task_measurement.compute(
                                task_gold_layouts[task], eval_pred[task])

                    # Measure loss & mIoU
                    if num_eval_steps == 0:
                        num_eval_steps = 1
                    eval_loss = eval_loss / num_eval_steps

                    # log
                    log_dict = {
                        'epoch': epoch,
                        'eval_loss': eval_loss,
                    }
                    log_dict.update(
                        {f'{tn}_loss': v for tn, v in eval_task_loss.items()})
                    log_dict.update(
                        {f'{tn}_miou': v for tn, v in task_miou.items()})
                    wandb.log(log_dict, step=global_step)
                    print('\t'.join([
                        f'[{epoch}/{self.args.epoch}]',
                        f'Eval Loss: {eval_loss:.3f}',
                    ]))
                    for task, task_loss in eval_task_loss.items():
                        print(f'{task} Loss: {task_loss:.3f}')
                        if task in task_miou:
                            print(f'{task} mIoU: {task_miou[task]:.3f}')

                measure = eval_loss
                is_best = self.ckpt_measurement.update(measure)
                wandb.log(
                    {f'checkpoint_measure-{self.ckpt_measurement.measurement}': measure}, step=global_step)
                print(
                    f'Checkpoint Measure-{self.ckpt_measurement.measurement}: {measure:.3f}')
                self.do_checkpointing(epoch, is_best)

            torch.distributed.barrier()

    def do_checkpointing(self, epoch, is_best):
        # save every checkpoint
        if is_best:
            with open(os.path.join(self.args.out_dir, 'best_epoch'), 'w') as f:
                f.write(f"{epoch}")
