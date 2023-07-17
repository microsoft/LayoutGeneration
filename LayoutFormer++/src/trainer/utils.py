from typing import List, Tuple

import torch
import numpy as np

from evaluation import metrics


class CheckpointMeasurement:

    MIOU = 'mIoU'  # layout mIoU
    ALIGNMENT = 'alignment'  # layout alignment (negative)
    OVERLAP = 'overlap'  # layout overlap (negative)
    ACCURACY = 'acc'  # bbox accuracy
    EVAL_LOSS = 'eval_loss'

    def __init__(self, max_num_elements: int, measurement: str = 'mIoU') -> None:
        self._max_num_elements = max_num_elements
        self._measurement = measurement
        self._init_best_value()

    def _init_best_value(self) -> None:
        if self._measurement == self.MIOU or self._measurement == self.ACCURACY:
            self._is_positive = True
        elif self._measurement == self.ALIGNMENT or self._measurement == self.OVERLAP or self._measurement == self.EVAL_LOSS:
            self._is_positive = False
        else:
            raise NotImplementedError(f"No measurement: {self._measurement}")
        self.reset()

    def reset(self) -> None:
        if self._is_positive:
            self._best_value = -1e+8
        else:
            self._best_value = np.inf

    @property
    def measurement(self):
        return self._measurement

    @property
    def best_value(self):
        return self._best_value

    @property
    def is_positive(self):
        return self._is_positive

    @property
    def max_num_elements(self):
        return self._max_num_elements

    def compute(self, gold_layouts: List[Tuple], pred_layouts: List[Tuple]) -> float:
        if self._measurement == self.MIOU:
            value = metrics.compute_maximum_iou(gold_layouts, pred_layouts)
        else:
            pred_bbox, pred_bbox_mask = list(), list()
            for bbox, _ in pred_layouts:
                diff = self.max_num_elements - len(bbox)
                if diff > 0:
                    mask = np.arange(self.max_num_elements) < len(bbox)
                    pred_bbox.append(np.concatenate(
                        [bbox, np.zeros((diff, 4))], axis=0))
                elif diff < 0:
                    pred_bbox.append(bbox[:self.max_num_elements, :])
                    mask = np.ones(self.max_num_elements, dtype=bool)
                else:
                    pred_bbox.append(bbox)
                    mask = np.ones(self.max_num_elements, dtype=bool)
                pred_bbox_mask.append(mask)
            pred_bbox = torch.from_numpy(np.stack(pred_bbox, axis=0))
            pred_bbox_mask = torch.from_numpy(np.stack(pred_bbox_mask, axis=0))

            if self._measurement == self.ALIGNMENT:
                value = metrics.compute_alignment(
                    pred_bbox, pred_bbox_mask).mean().item()
            else:
                # overlap
                value = metrics.compute_overlap(
                    pred_bbox, pred_bbox_mask).mean().item()
        return np.nan_to_num(value)

    def update(self, measure_value: float) -> bool:
        if self._is_positive:
            if measure_value > self._best_value:
                self._best_value = measure_value
                return True
            else:
                return False
        else:
            if measure_value < self._best_value:
                self._best_value = measure_value
                return True
            else:
                return False
