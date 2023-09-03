import os
import os.path as osp
import shutil
import pickle
from typing import Dict, Callable

import wandb
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from utils import utils
from trainer.utils import CheckpointMeasurement
from data.transforms import DiscretizeBoundingBox


def linear(a, b, x, min_x, max_x):
    return a + min(max((x - min_x) / (max_x - min_x), 0), 1) * (b - a)


class Trainer():

    TRAINER_NAME = 'ddp'

    def __init__(self,
                 task_name: str,
                 args,
                 model,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 optimizer,
                 scheduler=None,
                 checkpoint_measure=None,
                 d2c_fn=None,
                 is_debug=False,
                 is_label_condition=True,
                 task_config: Dict = None,
                 collate_fn: Callable = None):
        self.args = args
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.d2c_fn = d2c_fn or self.default_d2c_fn
        self.is_debug = is_debug
        self.is_label_condition = is_label_condition
        self.collate_fn = collate_fn or utils.collate_fn

        self.ckpt_measurement = CheckpointMeasurement(
            max_num_elements=self.args.max_num_elements,
            measurement=checkpoint_measure or CheckpointMeasurement.MIOU)

        # Hyper-parameters
        self.gradient_accumulation = self.args.gradient_accumulation
        # Some optimizers like AdaFactor has integrated gradient clipping, so we don't need to clip it.
        self.enable_clip_gradient = self.args.enable_clip_gradient
        self.clip_gradient = self.args.clip_gradient

        self._local_rank = 0
        self._world_size = torch.cuda.device_count()

        self._setup_experiment(task_name, task_config)
        self._setup_model()
        self._setup_dataloader()

    def _setup_experiment(self, task_name: str, task_config: Dict):
        dist.init_process_group(self.args.backend)
        self._local_rank = int(self.args.local_rank)
        self._world_size = int(os.environ['WORLD_SIZE'])
        print(f'World size: {self._world_size}, Local rank: {self._local_rank}, Main Process: {self._is_main_process}')

        # setup
        if self._is_main_process:
            utils.init_experiment(self.args, self.args.out_dir)
            hypara_config = utils.log_hyperparameters(self.args, self._world_size)
            hypara_config.update({
                'trainer': self.TRAINER_NAME
            })
            hypara_config.update(task_config)
            wandb.init(project=task_name, config=hypara_config)
        dist.barrier()

    def default_d2c_fn(self, bbox):
        discrete_fn = DiscretizeBoundingBox(
            num_x_grid=self.args.discrete_x_grid,
            num_y_grid=self.args.discrete_y_grid)
        return discrete_fn.continuize(bbox)

    def convert_bbox_format(self, bboxes):
        _bboxes = self.d2c_fn(bboxes)
        if self.args.bbox_format == "xywh":
            _bboxes = utils.convert_xywh_to_ltrb(bboxes)
        elif self.args.bbox_format == "ltwh":
            _bboxes = utils.convert_ltwh_to_ltrb(bboxes)
        return _bboxes

    def collect_layouts(self, bboxes, labels, mask):
        converted_bbox = self.convert_bbox_format(bboxes)
        layouts = list()
        for j in range(labels.size(0)):
            _mask = mask[j]
            box = converted_bbox[j][_mask].cpu().numpy()
            label = labels[j][_mask].cpu().numpy()
            layouts.append((box, label))
        return layouts

    def save_val_output(self, gold_layouts, pred_layouts, name):
        val_output = {
            'gold': gold_layouts,
            'pred': pred_layouts,
            'name': name
        }

        save_path = osp.join(self.args.out_dir, 'val_output.pkl')
        with open(save_path, 'wb')as f:
            pickle.dump(val_output, f)

    def do_checkpointing(self, epoch, is_best):
        # Save checkpoint
        normal_ckpt_path = osp.join(self.args.out_dir, 'checkpoint.pth.tar')
        torch.save(self.model.state_dict(), normal_ckpt_path)
        if is_best:
            best_ckpt_path = osp.join(self.args.out_dir, 'model_best.pth.tar')
            if os.path.exists(best_ckpt_path):
                # Only keep on best checkpoint (Save memory)
                os.remove(best_ckpt_path)
            shutil.copy(normal_ckpt_path, best_ckpt_path)

    @property
    def _is_main_process(self):
        return self._local_rank in [-1, 0]

    def _setup_model(self):
        self.device = torch.device("cuda:{}".format(self._local_rank))
        self.model = torch.nn.parallel.DistributedDataParallel(self.model.to(self.device), device_ids=[self._local_rank],
                                                               output_device=self._local_rank, find_unused_parameters=True)

    def _setup_dataloader(self):
        self.train_sampler = DistributedSampler(dataset=self.train_dataset)
        self.val_sampler = DistributedSampler(dataset=self.val_dataset)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.args.batch_size,
                                           collate_fn=self.collate_fn,
                                           sampler=self.train_sampler,
                                           drop_last=True)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.args.eval_batch_size,
                                         collate_fn=self.collate_fn,
                                         sampler=self.val_sampler,
                                         drop_last=True)

    def __call__(self, train_step, eval_step):

        self.ckpt_measurement.reset()
        global_step = 0
        gold_layouts = []
        for epoch in range(self.args.epoch):
            self.train_sampler.set_epoch(epoch)
            self.model.train()
            for batch_idx, data in enumerate(self.train_dataloader):
                loss_info = train_step(self.args, self.model, data, self.device)

                kl_beta = linear(0.0, 1.0, global_step, self.args.kl_start_step, self.args.kl_end_step)
                loss = loss_info['group_bounding_box'] + loss_info['label_in_one_group'] + \
                    loss_info['grouped_box'] + loss_info['grouped_label'] + \
                    kl_beta * loss_info['KL']

                loss = loss / self.gradient_accumulation
                loss.backward()

                # weights update
                if ((batch_idx + 1) % self.gradient_accumulation == 0) or (batch_idx + 1 == len(self.train_dataloader)):
                    if self.enable_clip_gradient:
                        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.clip_gradient)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step(loss)

                if global_step % self.args.train_log_step == 0 and self._is_main_process:
                    wandb.log(
                        {
                            'training_loss': loss,
                            'group_bounding_box_loss': loss_info['group_bounding_box'],
                            'label_in_one_group_loss': loss_info['label_in_one_group'],
                            'grouped_box_loss': loss_info['grouped_box'],
                            'grouped_label_loss': loss_info['grouped_label'],
                            'kl_loss': loss_info['KL']
                        },
                        step=global_step)

                    print('\t'.join([
                        f'[{epoch}/{self.args.epoch}][{batch_idx}/{len(self.train_dataloader)}]',
                        f'Loss: {loss.item():.3f}'
                    ]))
                global_step += 1

            if self._is_main_process:
                self.model.eval()
                pred_layouts = []
                fn_list = []
                with torch.no_grad():
                    for data in self.val_dataloader:
                        ori, rec, masks = eval_step(self.model, data, self.device)
                        pred_bboxes = rec['bboxes']
                        pred_labels = ori['labels'] if self.is_label_condition else rec['labels']
                        gold_labels = ori['labels']
                        gold_mask = masks['ori_box_mask']
                        pred_mask = masks['rec_box_mask']

                        pred_layouts.extend(self.collect_layouts(pred_bboxes, pred_labels, pred_mask))

                        if epoch == 0:
                            gold_bboxes = ori['bboxes']
                            gold_layouts.extend(self.collect_layouts(gold_bboxes, gold_labels, gold_mask))
                            fn_list += data['name']

                if self.is_debug:
                    self.save_val_output(gold_layouts, pred_layouts, fn_list)

                # checkpoint measure
                measure = self.ckpt_measurement.compute(gold_layouts, pred_layouts)
                is_best = self.ckpt_measurement.update(measure)
                wandb.log(
                    {f'checkpoint_measure-{self.ckpt_measurement.measurement}': measure}, step=global_step)
                print(f'Checkpoint Measure-{self.ckpt_measurement.measurement}: {measure:.3f}')
                self.do_checkpointing(epoch, is_best)
                print('finish checkpoint epoch {}'.format(epoch))

            dist.barrier()

    def clean_up(self):
        dist.destroy_process_group()
