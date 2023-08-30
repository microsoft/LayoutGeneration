# coding=utf8

import logging
import os
import os.path as osp
from typing import Callable, Dict, List

import deepspeed
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from semantic_parser.metrics import Metric
from utils import file_utils

logger = logging.getLogger(__name__)


class Generator:

    MODEL_BIN_NAME = "pytorch_model.bin"

    def __init__(self, args, model, dataset: Dataset, collate_fn: Callable,
                 metrics: List[Metric], save_prefix: str = 'test') -> None:
        self.args = args
        self.model = model
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.output_dir = args.output_dir
        self.metrics = metrics
        self.ckpt_path = self.output_dir
        self.ds_ckpt_tag = self.args.ds_ckpt_tag

        self.metrics_save_path = osp.join(args.prediction_dir, f'{save_prefix}_metrics.json')
        self.prediction_save_path = osp.join(args.prediction_dir, f'{save_prefix}_predictions.json')
        self._setup()

    @property
    def _is_main_process(self):
        return self._local_rank in {-1, 0}

    def _setup(self):
        self.use_deepspeed = True
        self._local_rank = int(os.environ['LOCAL_RANK'])
        self.device = torch.device("cuda:{}".format(self._local_rank))
        eval_micro_batch_size = self.args.eval_micro_batch_size
        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=eval_micro_batch_size,
                                     collate_fn=self.collate_fn)

        if os.path.exists(os.path.join(self.ckpt_path, self.MODEL_BIN_NAME)):
            self.use_deepspeed = False
            state = torch.load(os.path.join(self.ckpt_path, self.MODEL_BIN_NAME))
            self.model.load_state_dict(state, strict=False)
            self.model.to(self.device)
        else:
            logger.info("Inference using deepspeed")
            deepspeed.init_distributed(dist_backend=self.args.backend)
            logger.info(f"Local Rank: {self._local_rank}, Main Process: {self._is_main_process}")

            # load with deepspeed
            params = filter(lambda p: p.requires_grad, self.model.parameters())
            self.model, _, _, _ = deepspeed.initialize(args=self.args, model=self.model,
                                                       model_parameters=params)
            if self.ckpt_path is not None:
                _, client_state = self.model.load_checkpoint(self.ckpt_path, tag=self.ds_ckpt_tag,
                                                            load_module_only=True,
                                                            load_optimizer_states=False,
                                                            load_lr_scheduler_states=False) # model engine
                logger.info(client_state)
            dist.barrier()

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

    def __call__(self, eval_step: Callable):
        if self._is_main_process:
            self.model.eval()
            predictions = list()
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.dataloader):
                    batch_metrics, batch_predictions = eval_step(self.model, batch, self.device)
                    self._aggreate_metrics(batch_metrics)
                    predictions.extend(batch_predictions)
                metrics = self._collect_metrics(reset=True)
                for key, value in metrics.items():
                    logger.info(f"{key}: {value:.3f}")

                file_utils.write_json(self.metrics_save_path, metrics)
                file_utils.write_json(self.prediction_save_path, predictions)

        if self.use_deepspeed:
            dist.barrier()
