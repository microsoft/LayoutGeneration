import os
import os.path as osp
import shutil
import pickle
from typing import Dict, Callable
from collections import OrderedDict as OD

import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data.transforms import DiscretizeBoundingBox
from utils import utils
from .utils import CheckpointMeasurement


class Trainer():

    TRAINER_NAME = 'basic'

    def __init__(self,
                 task_name: str,
                 args,
                 tokenizer,
                 model,
                 seq_processor,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 optimizer,
                 scheduler=None,
                 is_label_condition=True,
                 checkpoint_measure=None,
                 d2c_fn=None,
                 is_debug=False,
                 task_config: Dict = None,
                 collate_fn: Callable = None):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.seq_processor = seq_processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.d2c_fn = d2c_fn or self.default_d2c_fn
        self.is_label_condition = is_label_condition
        self.is_debug = is_debug
        self.collate_fn = collate_fn or utils.collate_fn

        self.ckpt_measurement = CheckpointMeasurement(
            max_num_elements=self.args.max_num_elements,
            measurement=checkpoint_measure or CheckpointMeasurement.MIOU)

        # Hyper-parameters
        self.load_train_ckpt = args.load_train_ckpt
        self.train_ckpt_path = args.train_ckpt_path

        self.gradient_accumulation = self.args.gradient_accumulation
        # Some optimizers like AdaFactor has integrated gradient clipping, so we don't need to clip it.
        self.enable_clip_gradient = self.args.enable_clip_gradient
        self.clip_gradient = self.args.clip_gradient

        self._local_rank = 0
        self._world_size = torch.cuda.device_count()

        self._setup_experiment(task_name, task_config)
        self._setup_model()
        if self.load_train_ckpt:
            print(f"Loading Checkpoint for training from: {self.train_ckpt_path}")
            self._load_ckpt()
        self._setup_dataloader()

    def _setup_experiment(self, task_name: str, task_config: Dict):
        utils.init_experiment(self.args, self.args.out_dir)
        hypara_config = utils.log_hyperparameters(self.args, self._world_size)
        hypara_config.update({
            'trainer': self.TRAINER_NAME
        })
        hypara_config.update(task_config)
        wandb.init(project=task_name, config=hypara_config)

    def _load_ckpt(self):
        state_dict = torch.load(self.train_ckpt_path, map_location=self.device)
        state = OD([(key.split("module.")[-1], state_dict[key]) for key in state_dict])
        self.model.load_state_dict(state, strict=False)

    def _setup_model(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device)

    def _setup_dataloader(self):
        if torch.cuda.device_count() > 1:
            train_bsz = self.args.batch_size * torch.cuda.device_count()
            val_bsz = self.args.eval_batch_size * torch.cuda.device_count()
        else:
            train_bsz, val_bsz = self.args.batch_size, self.args.eval_batch_size
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=train_bsz,
                                           collate_fn=self.collate_fn,
                                           drop_last=True, shuffle=True)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=val_bsz,
                                         collate_fn=self.collate_fn,
                                         drop_last=True)

    def default_d2c_fn(self, bbox):
        discrete_fn = DiscretizeBoundingBox(
            num_x_grid=self.args.discrete_x_grid,
            num_y_grid=self.args.discrete_y_grid)
        return discrete_fn.continuize(bbox)

    # TODO: define a Metrics class
    def init_val_metrics(self):
        self.val_num_bbox_correct = 0.0
        self.val_num_bbox = 0.0
        self.val_num_label_correct = 0.0
        self.val_num_examples = 0.0

        self.violation_num = 0.0
        self.rel_num = 0.0

    def aggregate_metrics(self, metrics):
        self.val_num_bbox_correct += metrics['num_bbox_correct']
        self.val_num_bbox += metrics['num_bbox']
        self.val_num_label_correct += metrics['num_label_correct']
        self.val_num_examples += metrics['num_examples']
        self.violation_num += metrics.get('violation_num', 0)
        self.rel_num += metrics.get('rel_num', 0)

    def logging_metrics(self, epoch, global_step):
        self.val_bbox_acc = self.val_num_bbox_correct / self.val_num_bbox
        self.val_label_acc = self.val_num_label_correct / self.val_num_examples
        self.violation_rate = self.violation_num / self.rel_num
        wandb.log(
            {
                'epoch': epoch,
                'val_acc': self.val_bbox_acc,
                'val_label_acc': self.val_label_acc,
                'violation_rate': self.violation_rate
            },
            step=global_step)
        print('\t'.join([
            f'[{epoch}/{self.args.epoch}]',
            f'Val Accuracy: {self.val_bbox_acc:.3f}',
            f'Val Label Accuracy: {self.val_label_acc:.3f}',
            f'Violation Rate: {self.violation_rate:.3f}',

        ]))

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

    def do_checkpointing(self, epoch, is_best):
        # Save checkpoint
        if not os.path.exists(self.args.out_dir):
            os.mkdir(self.args.out_dir)
        normal_ckpt_path = osp.join(self.args.out_dir, 'checkpoint.pth.tar')
        torch.save(self.model.state_dict(), normal_ckpt_path)
        if is_best:
            best_ckpt_path = osp.join(self.args.out_dir, 'model_best.pth.tar')
            if os.path.exists(best_ckpt_path):
                # Only keep on best checkpoint (Save memory)
                os.remove(best_ckpt_path)
            shutil.copy(normal_ckpt_path, best_ckpt_path)

    def save_val_output(self, gold_layouts, pred_layouts, ori_str, out_str):
        val_output = {
            'gold': gold_layouts,
            'pred': pred_layouts,
            'out_str': out_str,
            'ori_str': ori_str
        }
        save_path = osp.join(self.args.out_dir, 'val_output.pkl')
        with open(save_path, 'wb')as f:
            pickle.dump(val_output, f)

    def __call__(self, train_step, eval_step):
        self.ckpt_measurement.reset()
        gold_layouts = []
        global_step = 0
        for epoch in range(self.args.epoch):
            self.model.train()
            for batch_idx, data in enumerate(self.train_dataloader):
                loss = train_step(self.model, data, self.tokenizer, self.device)
                loss = loss / self.gradient_accumulation
                loss.backward()

                # weights update
                if ((batch_idx + 1) % self.gradient_accumulation == 0) or (batch_idx + 1 == len(self.train_dataloader)):
                    if self.enable_clip_gradient:
                        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.clip_gradient)
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

            self.model.eval()
            self.init_val_metrics()
            pred_layouts = list()

            with torch.no_grad():
                for data in self.val_dataloader:
                    metrics, out = eval_step(self.model, data, self.seq_processor, self.tokenizer, self.device)
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

    def clean_up(self):
        pass
