import torch

from utils import (
    compute_alignment,
    compute_maximum_iou,
    compute_overlap,
    convert_ltwh_to_ltrb,
    read_pt,
)


class Ranker:
    lambda_1 = 0.2
    lambda_2 = 0.2
    lambda_3 = 0.6

    def __init__(self, val_path=None):
        self.val_path = val_path
        if self.val_path:
            self.val_data = read_pt(val_path)
            self.val_labels = [vd["labels"] for vd in self.val_data]
            self.val_bboxes = [vd["bboxes"] for vd in self.val_data]

    def __call__(self, predictions: list):
        metrics = []
        for pred_labels, pred_bboxes in predictions:
            metric = []
            _pred_labels = pred_labels.unsqueeze(0)
            _pred_bboxes = convert_ltwh_to_ltrb(pred_bboxes).unsqueeze(0)
            _pred_padding_mask = torch.ones_like(_pred_labels).bool()
            metric.append(compute_alignment(_pred_bboxes, _pred_padding_mask))
            metric.append(compute_overlap(_pred_bboxes, _pred_padding_mask))
            if self.val_path:
                metric.append(
                    compute_maximum_iou(
                        pred_labels,
                        pred_bboxes,
                        self.val_labels,
                        self.val_bboxes,
                    )
                )
            metrics.append(metric)

        metrics = torch.tensor(metrics)
        min_vals, _ = torch.min(metrics, 0, keepdim=True)
        max_vals, _ = torch.max(metrics, 0, keepdim=True)
        scaled_metrics = (metrics - min_vals) / (max_vals - min_vals)
        if self.val_path:
            quality = (
                scaled_metrics[:, 0] * self.lambda_1
                + scaled_metrics[:, 1] * self.lambda_2
                + (1 - scaled_metrics[:, 2]) * self.lambda_3
            )
        else:
            quality = (
                scaled_metrics[:, 0] * self.lambda_1
                + scaled_metrics[:, 1] * self.lambda_2
            )
        _predictions = sorted(zip(predictions, quality), key=lambda x: x[1])
        ranked_predictions = [item[0] for item in _predictions]
        return ranked_predictions
