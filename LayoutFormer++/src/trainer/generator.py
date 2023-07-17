from typing import List
import pickle
import time
import math
from pathlib import Path
from typing import Callable
from collections import OrderedDict as OD

import torch
import deepspeed
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from evaluation import metrics
from utils import utils, os_utils, visualization
from data.transforms import DiscretizeBoundingBox


class Generator():

    def __init__(self,
                 args,
                 tokenizer,
                 model,
                 seq_processor,
                 test_dataset,
                 fid_model,
                 ckpt_path: str = None,
                 ds_ckpt_tag: str = None,  # checkpoint tag for deepspeed
                 d2c_fn=None,
                 is_label_condition=True,
                 saved_layouts=['input', 'gold', 'pred'],
                 save_entries=None,
                 collate_fn: Callable = None,):
        self.args = args
        self.tokenizer = tokenizer
        self.model = model
        self.seq_processor = seq_processor
        self.test_dataset = test_dataset
        self.d2c_fn = d2c_fn or self.default_d2c_fn
        self.fid = fid_model
        self.is_label_condition = is_label_condition
        self.saved_layouts = saved_layouts
        self.save_entries = save_entries or list()
        self.collate_fn = collate_fn or utils.collate_fn

        self.image_out_dir = Path(self.args.out_dir) / 'pics'
        self.metrics_out_dir = Path(self.args.out_dir) / 'metrics.pkl'
        self.results_out_dir = Path(self.args.out_dir) / 'results.pkl'
        self._setup_experiment()
        self.ds_ckpt_tag = ds_ckpt_tag
        self._setup_model(ckpt_path, ds_ckpt_tag)

    @property
    def only_visualized_pred(self):
        return "pred" in self.saved_layouts and len(self.saved_layouts) == 1

    def _setup_experiment(self):
        # Get Device
        if self.args.trainer == 'deepspeed':
            deepspeed.init_distributed(dist_backend=self.args.backend)
            self._is_distributed = True
            self._local_rank = int(self.args.local_rank)
        elif self.args.trainer == 'ddp':
            dist.init_process_group(self.args.backend)
            self._is_distributed = True
            self._local_rank = int(self.args.local_rank)
        else:
            self._is_distributed = False
            self._local_rank = 0
        self.device = torch.device("cuda:{}".format(self._local_rank))
        print(
            f"Local Rank: {self._local_rank}, Main Process: {self._is_main_process}")

        # Dataloder
        self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                          batch_size=self.args.eval_batch_size,
                                          collate_fn=self.collate_fn)

        if self._is_main_process:
            os_utils.makedirs(self.image_out_dir)

    def _setup_model(self, ckpt_path: str, ds_ckpt_tag: str = None):
        if self.args.trainer == 'ddp':
            self.model = DistributedDataParallel(self.model.to(self.device), device_ids=[self._local_rank],
                                                 output_device=self._local_rank)
        elif self.args.trainer == 'deepspeed':
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.model, _, _, _ = deepspeed.initialize(args=self.args, model=self.model,
                                                       model_parameters=params)
        else:
            self.model.to(self.device)

        # Load checkpoint
        if ckpt_path:
            if self.args.trainer == 'deepspeed':
                _, client_state = self.model.load_checkpoint(ckpt_path, tag=ds_ckpt_tag,
                                                             load_module_only=True,
                                                             load_optimizer_states=False,
                                                             load_lr_scheduler_states=False)  # model engine
                print(client_state)
            elif self.args.trainer == 'ddp':
                state_dict = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                state_dict = torch.load(ckpt_path, map_location=self.device)
                state = OD([(key.split("module.")[-1], state_dict[key])
                           for key in state_dict])
                self.model.load_state_dict(state, strict=False)
        if self._is_distributed:
            dist.barrier()

    @property
    def _is_main_process(self):
        return self._local_rank in [-1, 0]

    def default_d2c_fn(self, bbox):
        discrete_fn = DiscretizeBoundingBox(
            num_x_grid=self.args.discrete_x_grid,
            num_y_grid=self.args.discrete_y_grid)
        return discrete_fn.continuize(bbox)

    def init_eval_metrics(self):
        self.eval_num_bbox_correct = 0.0
        self.eval_num_bbox = 0.0
        self.eval_num_label_correct = 0.0
        self.eval_num_examples = 0.0
        self.alignment = []
        self.overlap = []
        self.new_overlap = []
        self.violation_num = 0.0
        self.rel_num = 0.0

    def aggregate_metrics(self, metric, layouts, gold_mask, pred_mask):
        self.eval_num_bbox_correct += metric['num_bbox_correct']
        self.eval_num_bbox += metric['num_bbox']
        self.eval_num_label_correct += metric['num_label_correct']
        self.eval_num_examples += metric['num_examples']

        self.violation_num += metric.get('violation_num', 0)
        self.rel_num += metric.get('rel_num', 0)

        gold_padding_mask = ~gold_mask
        pred_padding_mask = ~pred_mask
        self.fid.collect_features(layouts['pred_bboxes'].to(
            self.device), layouts['pred_labels'].to(self.device), pred_padding_mask.to(self.device))
        self.fid.collect_features(layouts['gold_bboxes'].to(self.device), layouts['gold_labels'].to(
            self.device), gold_padding_mask.to(self.device), real=True)

        self.alignment += metrics.compute_alignment(
            layouts['pred_bboxes'], pred_mask).tolist()
        self.overlap += metrics.compute_overlap(
            layouts['pred_bboxes'], pred_mask).tolist()
        if self.args.dataset == 'rico':
            self.new_overlap += metrics.compute_overlap_ignore_bg(
                layouts['pred_bboxes'], layouts['pred_labels'], pred_mask).tolist()

    def logging_metrics(self):
        self.eval_bbox_acc = self.eval_num_bbox_correct / self.eval_num_bbox
        self.eval_label_acc = self.eval_num_label_correct / self.eval_num_examples
        self.fid_score_eval = self.fid.compute_score()
        self.max_iou_eval = metrics.compute_maximum_iou(
            self.gold_layouts, self.pred_layouts)
        self.alignment_eval = metrics.average(self.alignment)
        self.overlap_eval = metrics.average(self.overlap)

        if self.rel_num > 0:
            self.violation_rate = self.violation_num / self.rel_num
        else:
            self.violation_rate = 0.0

        print('Generated layouts are saved at:', self.args.out_dir)
        print("bbox accuracy: {}".format(self.eval_bbox_acc))
        print("Label accuracy: {}".format(self.eval_label_acc))
        print("FID score: {}".format(self.fid_score_eval))
        print("Max IoU: {}".format(self.max_iou_eval))
        print("Alignment: {}".format(self.alignment_eval))
        print("Overlap: {}".format(self.overlap_eval))
        print("Violation Rate: {}".format(self.violation_rate))
        print("Start Time: {}".format(self.start_time))
        print("Finish Time: {}".format(self.finish_time))

        eval_metrics = {
            'bbox_acc': self.eval_bbox_acc,
            'label_acc': self.eval_label_acc,
            'fid': self.fid_score_eval,
            'mIoU': self.max_iou_eval,
            'alignment': self.alignment_eval,
            'overlap': self.overlap_eval,
            'violation_rate': self.violation_rate,
            'start_time': self.start_time,
            'finish_time': self.finish_time
        }
        if self.args.dataset == 'rico':
            self.new_overlap = list(
                filter(lambda x: not math.isnan(x), self.new_overlap))
            self.new_overlap_eval = metrics.average(self.new_overlap)
            print("New Overlap: {}".format(self.new_overlap_eval))
            eval_metrics['new_overlap'] = self.new_overlap_eval

        with open(self.metrics_out_dir, 'wb') as fb:
            pickle.dump(eval_metrics, fb)

        with open(self.results_out_dir, 'wb') as fb:
            pickle.dump(self.results, fb)

    def collect_layouts(self, bboxes, labels, mask, layouts):
        bboxes = self.d2c_fn(bboxes)
        if self.args.bbox_format == "xywh":
            bboxes = utils.convert_xywh_to_ltrb(bboxes)
        elif self.args.bbox_format == "ltwh":
            bboxes = utils.convert_ltwh_to_ltrb(bboxes)

        for j in range(labels.size(0)):
            _mask = mask[j]
            box = bboxes[j][_mask].cpu().numpy()
            label = labels[j][_mask].cpu().numpy()
            layouts.append((box, label))
        return layouts, bboxes

    def switch_task(self, task: str, saved_layouts: List[str]):
        self.saved_layouts = saved_layouts
        self.test_dataset.switch_task(task)
        self.seq_processor = self.test_dataset.seq_processor
        print(f"Switch to {task}")

        self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                          batch_size=self.args.eval_batch_size,
                                          collate_fn=self.collate_fn)
        self.image_out_dir = Path(self.args.out_dir) / task / self.ds_ckpt_tag / 'pics'
        self.metrics_out_dir = Path(self.args.out_dir) / task / self.ds_ckpt_tag / "metrics.pkl"
        self.results_out_dir = Path(self.args.out_dir) / task / self.ds_ckpt_tag / "results.pkl"
        if self._is_main_process:
            os_utils.makedirs(self.image_out_dir)

    def __call__(self, test_step, draw_colors, constraint_fn: Callable = None):
        if self._is_main_process:
            self.fid.model.to(self.device)

            self.model.eval()
            self.init_eval_metrics()
            self.results = []
            self.input_layouts, self.gold_layouts, self.pred_layouts = [], [], []
            with torch.no_grad():
                self.start_time = time.strftime('%Y.%m.%d %H:%M:%S')
                for batch_idx, data in enumerate(self.test_dataloader):
                    if constraint_fn:
                        metrics, out = test_step(
                            self.model, data, self.seq_processor, self.tokenizer, self.device, constraint_fn)
                    else:
                        metrics, out = test_step(
                            self.model, data, self.seq_processor, self.tokenizer, self.device)

                    gold_bboxes = out['gold_bboxes']
                    gold_labels = out['gold_labels']
                    pred_bboxes = out['pred_bboxes']
                    pred_labels = out['gold_labels'] if self.is_label_condition else out['pred_labels']
                    gold_mask = (
                        out['gold_mask'] if 'gold_mask' in out else out['mask']).cpu()
                    pred_mask = (
                        out['pred_mask'] if 'pred_mask' in out else out['mask']).cpu()

                    self.pred_layouts, pred_bboxes = self.collect_layouts(
                        pred_bboxes, pred_labels, pred_mask, self.pred_layouts)
                    pred_bboxes = pred_bboxes.cpu()
                    pred_labels = pred_labels.cpu()

                    self.gold_layouts, gold_bboxes = self.collect_layouts(
                        gold_bboxes, gold_labels, gold_mask, self.gold_layouts)
                    gold_bboxes = gold_bboxes.cpu()
                    gold_labels = gold_labels.cpu()

                    if 'input' in self.saved_layouts:
                        input_bboxes = out['input_bboxes']
                        input_labels = out['input_labels']
                        input_mask = (
                            out['input_mask'] if 'input_mask' in out else out['mask']).cpu()
                        self.input_layouts, input_bboxes = self.collect_layouts(
                            input_bboxes, input_labels, input_mask, self.input_layouts)
                        input_bboxes = input_bboxes.cpu()
                        input_labels = input_labels.cpu()

                    for j in range(len(pred_labels)):
                        if len(self.results) < self.args.num_save:
                            if self.only_visualized_pred:
                                img_bboxes = torch.stack([pred_bboxes[j]])
                                img_labels = torch.stack([pred_labels[j]])
                                img_masks = pred_mask[j].unsqueeze(0)
                                visualization.save_image(img_bboxes, img_labels, img_masks, draw_colors,
                                                         self.image_out_dir /
                                                         f'{len(self.results):04d}_gen.png',
                                                         canvas_size=(360, 240))
                            else:
                                if 'input' in self.saved_layouts:
                                    img_bboxes = torch.stack([input_bboxes[j]])
                                    img_labels = torch.stack([input_labels[j]])
                                    img_masks = input_mask[j].unsqueeze(0)
                                    visualization.save_image(img_bboxes, img_labels, img_masks, draw_colors,
                                                             self.image_out_dir /
                                                             f'{len(self.results):04d}_input.png',
                                                             canvas_size=(360, 240))

                                img_bboxes = torch.stack([pred_bboxes[j]])
                                img_labels = torch.stack([pred_labels[j]])
                                img_masks = pred_mask[j].unsqueeze(0)
                                visualization.save_image(img_bboxes, img_labels, img_masks, draw_colors,
                                                         self.image_out_dir /
                                                         f'{len(self.results):04d}_gen.png',
                                                         canvas_size=(360, 240))

                                img_bboxes = torch.stack([gold_bboxes[j]])
                                img_labels = torch.stack([gold_labels[j]])
                                img_masks = gold_mask[j].unsqueeze(0)
                                visualization.save_image(img_bboxes, img_labels, img_masks, draw_colors,
                                                         self.image_out_dir /
                                                         f'{len(self.results):04d}_ori.png',
                                                         canvas_size=(360, 240))

                        self.results.append(
                            {
                                'fn': data['name'][j],
                                'pred': (pred_bboxes[j], pred_labels[j],),
                                'gold': (gold_bboxes[j], gold_labels[j],),
                            }
                        )
                        if 'input' in self.saved_layouts:
                            self.results[-1].update(
                                {'input': (input_bboxes[j], input_labels[j],)})
                        if self.save_entries:
                            for entry in self.save_entries:
                                if entry in out:
                                    self.results[-1].update(
                                        {entry: out[entry][j]})

                    layouts = {
                        'pred_bboxes': pred_bboxes,
                        'pred_labels': pred_labels,
                        'gold_bboxes': gold_bboxes,
                        'gold_labels': gold_labels
                    }
                    self.aggregate_metrics(
                        metrics, layouts, gold_mask, pred_mask)
                    print(f'finish [{batch_idx}/{len(self.test_dataloader)}]')

                self.finish_time = time.strftime('%Y.%m.%d %H:%M:%S')

            self.logging_metrics()

        if self._is_distributed:
            dist.barrier()

    def clean_up(self):
        if self.args.trainer == 'ddp':
            dist.destroy_process_group()
