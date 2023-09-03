import pickle
from pathlib import Path
from typing import Callable
from collections import OrderedDict as OD
import math

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
                 model,
                 test_dataset,
                 fid_model,
                 ckpt_path: str = None,
                 d2c_fn=None,
                 is_label_condition=True,
                 saved_layouts=['gold', 'pred', 'group'],
                 save_entries=None,
                 collate_fn: Callable = None):
        self.args = args
        self.model = model
        self.test_dataset = test_dataset
        self.d2c_fn = d2c_fn or self.default_d2c_fn
        self.fid = fid_model
        self.is_label_condition = is_label_condition
        self.saved_layouts = saved_layouts
        self.save_entries = save_entries or list()
        self.collate_fn = collate_fn or utils.collate_fn

        self.image_out_dir = Path(self.args.out_dir) / 'pics'
        self._setup_experiment()
        self._setup_model(ckpt_path)

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
        print(f"Local Rank: {self._local_rank}, Main Process: {self._is_main_process}")

        # Dataloder
        self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                          batch_size=self.args.eval_batch_size,
                                          collate_fn=self.collate_fn)

        if self._is_main_process:
            os_utils.makedirs(self.image_out_dir)

    def _setup_model(self, ckpt_path: str):
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
                self.model.load_checkpoint(ckpt_path)  # model engine
            elif self.args.trainer == 'ddp':
                state_dict = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            else:
                state_dict = torch.load(ckpt_path, map_location=self.device)
                state = OD([(key.split("module.")[-1], state_dict[key]) for key in state_dict])
                self.model.load_state_dict(state)
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
        self.alignment = []
        self.overlap = []
        self.new_overlap = []

    def aggregate_metrics(self, layouts, gold_mask, pred_mask):

        padding_mask = ~pred_mask
        self.fid.collect_features(layouts['pred_bboxes'].to(self.device), layouts['pred_labels'].to(self.device), padding_mask.to(self.device))
        padding_mask = ~gold_mask
        self.fid.collect_features(layouts['gold_bboxes'].to(self.device), layouts['gold_labels'].to(self.device), padding_mask.to(self.device), real=True)

        self.alignment += metrics.compute_alignment(layouts['pred_bboxes'], pred_mask).tolist()
        self.overlap += metrics.compute_overlap(layouts['pred_bboxes'], pred_mask).tolist()

        if self.args.dataset == 'rico':
            self.new_overlap += metrics.compute_overlap_ignore_bg(
                layouts['pred_bboxes'], layouts['pred_labels'], pred_mask).tolist()

    def logging_metrics(self):
        self.fid_score_eval = self.fid.compute_score()
        self.max_iou_eval = metrics.compute_maximum_iou(self.gold_layouts, self.pred_layouts)
        self.alignment_eval = metrics.average(self.alignment)
        self.overlap_eval = metrics.average(self.overlap)

        print('Generated layouts are saved at:', self.args.out_dir)
        print("FID score: {}".format(self.fid_score_eval))
        print("Max IoU: {}".format(self.max_iou_eval))
        print("Alignment: {}".format(self.alignment_eval))
        print("Overlap: {}".format(self.overlap_eval))

        eval_metrics = {
            'fid': self.fid_score_eval,
            'mIoU': self.max_iou_eval,
            'alignment': self.alignment_eval,
            'overlap': self.overlap_eval
        }

        if self.args.dataset == 'rico':
            self.new_overlap = list(
                filter(lambda x: not math.isnan(x), self.new_overlap))
            self.new_overlap_eval = metrics.average(self.new_overlap)
            print("New Overlap: {}".format(self.new_overlap_eval))
            eval_metrics['new_overlap'] = self.new_overlap_eval

        with open(self.args.out_dir+'/metrics.pkl', 'wb') as fb:
            pickle.dump(eval_metrics, fb)

        with open(self.args.out_dir+'/results.pkl', 'wb') as fb:
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

    def __call__(self, test_step, draw_colors):
        if self._is_main_process:
            self.fid.model.to(self.device)

            self.model.eval()
            self.init_eval_metrics()
            self.results = []
            self.group_layouts, self.gold_layouts, self.pred_layouts = [], [], []
            with torch.no_grad():
                for batch_idx, data in enumerate(self.test_dataloader):
                    ori, out, masks = test_step(self.args, self.model, data, self.device)

                    gold_bboxes = ori['bboxes']
                    gold_labels = ori['labels']
                    pred_bboxes = out['bboxes']
                    pred_labels = ori['labels'] if self.is_label_condition else out['labels']
                    pred_group_box = out['group_bounding_box']
                    pred_group_label = out['label_in_one_group']
                    gold_mask = masks['ori_box_mask']
                    pred_mask = masks['gen_box_mask']
                    group_mask = masks['gen_group_bounding_box_mask']

                    self.pred_layouts, pred_bboxes = self.collect_layouts(pred_bboxes, pred_labels, pred_mask, self.pred_layouts)
                    pred_bboxes = pred_bboxes.cpu()
                    pred_labels = pred_labels.cpu()

                    self.gold_layouts, gold_bboxes = self.collect_layouts(gold_bboxes, gold_labels, gold_mask, self.gold_layouts)
                    gold_bboxes = gold_bboxes.cpu()
                    gold_labels = gold_labels.cpu()
                    gold_labels[~gold_mask] = 0

                    self.group_layouts, pred_group_box = self.collect_layouts(pred_group_box, pred_group_label, group_mask, self.group_layouts)
                    pred_group_box = pred_group_box.cpu()
                    pred_group_label = pred_group_label.cpu()

                    for j in range(len(gold_labels)):
                        if len(self.results) < self.args.num_save:
                            # img_bboxes = torch.stack([gold_bboxes[j]])
                            # img_labels = torch.stack([gold_labels[j]])
                            # img_masks = gold_mask[j].unsqueeze(0).repeat(1, 1)
                            # visualization.save_image(img_bboxes, img_labels, img_masks, draw_colors,
                            #                          self.image_out_dir / f'{len(self.results)}_ori.png',
                            #                          canvas_size=(360, 240))

                            img_bboxes = torch.stack([pred_bboxes[j]])
                            img_labels = torch.stack([pred_labels[j]])
                            img_masks = pred_mask[j].unsqueeze(0).repeat(1, 1)
                            visualization.save_image(img_bboxes, img_labels, img_masks, draw_colors,
                                                     self.image_out_dir / f'{len(self.results)}_gen.png',
                                                     canvas_size=(360, 240))

                            img_bboxes = torch.stack([pred_group_box[j]])
                            img_labels = torch.stack([group_mask[j]])
                            img_masks = group_mask[j].unsqueeze(0).repeat(1, 1)
                            visualization.save_image(img_bboxes, img_labels, img_masks, draw_colors,
                                                     self.image_out_dir / f'{len(self.results)}_group.png',
                                                     canvas_size=(360, 240))
                        self.results.append(
                            {
                                'fn': data['name'][j],
                                'pred': (pred_bboxes[j], pred_labels[j],),
                                'gold': (gold_bboxes[j], gold_labels[j],),
                                'pred_group': (pred_group_box[j], pred_group_label[j]),
                            }
                        )

                        if self.save_entries:
                            for entry in self.save_entries:
                                if entry in out:
                                    self.results[-1].update({entry: out[entry][j]})

                    layouts = {
                        'pred_bboxes': pred_bboxes,
                        'pred_labels': pred_labels,
                        'gold_bboxes': gold_bboxes,
                        'gold_labels': gold_labels
                    }
                    self.aggregate_metrics(layouts, gold_mask, pred_mask)
                    print(f'finish [{batch_idx}/{len(self.test_dataloader)}]')

            self.logging_metrics()

        if self._is_distributed:
            dist.barrier()

    def clean_up(self):
        if self.args.trainer == 'ddp':
            dist.destroy_process_group()
