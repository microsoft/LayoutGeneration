from typing import Dict, List, Tuple
import numpy as np
import random
import copy

import torch
import torchvision.transforms as T

from data import transforms
from tasks.refinement import T5LayoutSequence
from utils import utils


class T5CompletionLayoutSequence(T5LayoutSequence):

    def parse_seq(self, _, output_str) -> Tuple[List[int], List[Tuple]]:
        return super().parse_seq(output_str)


class T5CompletionDataset:

    def __init__(self, cargs, tokenizer, seq_processor, sort_by_pos: bool = True,
                 shuffle_before_sort_by_label: bool = False,
                 sort_by_pos_before_sort_by_label: bool = False, *args, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.seq_processor = seq_processor
        self.index2label = seq_processor.index2label
        self.sort_by_pos = sort_by_pos
        transform_functions = [
            transforms.LexicographicSort(),
            transforms.DiscretizeBoundingBox(
                num_x_grid=cargs.discrete_x_grid,
                num_y_grid=cargs.discrete_y_grid)
        ]
        self.transform = T.Compose(transform_functions)
        self.shuffle_before_sort_by_label = shuffle_before_sort_by_label
        self.sort_by_pos_before_sort_by_label = sort_by_pos_before_sort_by_label
        print("Completion: ", transform_functions, "Sort by Pos: ", sort_by_pos,
              ", Shuffle: ", shuffle_before_sort_by_label,
              ", Sort by Pos before: ", sort_by_pos_before_sort_by_label)
        super().__init__(*args, **kwargs)

    def _sort_by_labels(self, labels, bboxes):
        _labels = labels.tolist()
        idx2label = [[idx, self.index2label[_labels[idx]]]
                     for idx in range(len(labels))]
        idx2label_sorted = sorted(idx2label, key=lambda x: x[1])
        idx_sorted = [d[0] for d in idx2label_sorted]
        return labels[idx_sorted], bboxes[idx_sorted]

    def _shuffle(self, labels, bboxes):
        ele_num = len(labels)
        shuffle_idx = np.arange(ele_num)
        random.shuffle(shuffle_idx)
        return labels[shuffle_idx], bboxes[shuffle_idx]

    def process(self, data) -> Dict:
        nd = self.transform(copy.deepcopy(data))

        # build layout sequence
        labels = nd['labels']
        bboxes = nd['discrete_gold_bboxes']
        gold_bboxes = nd['discrete_gold_bboxes']

        in_labels = labels[:1]
        in_bboxes = bboxes[:1, :]
        in_str = self.seq_processor.build_seq(in_labels, in_bboxes).lower().strip()
        if self.sort_by_pos:
            sorted_labels, sorted_bbox = labels, gold_bboxes
        else:
            if self.shuffle_before_sort_by_label:
                sorted_labels, sorted_bbox = self._shuffle(labels, gold_bboxes)
                sorted_labels, sorted_bbox = self._sort_by_labels(sorted_labels, sorted_bbox)
            else:
                sorted_labels, sorted_bbox = self._sort_by_labels(labels, gold_bboxes)
        out_str = self.seq_processor.build_seq(sorted_labels, sorted_bbox).lower().strip()
        return {
            'in_str': in_str,
            'out_str': out_str,
            'input_labels': in_labels,
            'input_bboxes': in_bboxes,
            'out_labels': sorted_labels,
            'out_bboxes': sorted_bbox,
            'gold_labels': sorted_labels,
            'gold_bboxes': sorted_bbox,
            'name': data['name']
        }


def completion_inference(model, data, seq_processor, tokenizer, device, top_k=10, temperature=0.7) -> Tuple[Dict, Dict]:
    gold_labels, mask = utils.to_dense_batch(data['gold_labels'])
    gold_bbox, _ = utils.to_dense_batch(data['gold_bboxes'])
    _input_labels, _input_mask = utils.to_dense_batch(data['input_labels'])
    _input_bbox, _ = utils.to_dense_batch(data['input_bboxes'])

    mask = mask.to(device).detach()
    _input_mask = _input_mask.to(device).detach()

    in_tokenization = tokenizer(data['in_str'], add_eos=True, add_bos=False)
    in_ids = in_tokenization['input_ids'].to(device)
    in_mask = in_tokenization['mask'].to(device)
    in_padding_mask = ~in_mask

    task_ids = None
    if 'task_id' in data:
        task_ids = torch.tensor(data['task_id']).long().to(device)

    decode_max_length = 120
    output_sequences = model(in_ids, in_padding_mask, max_length=decode_max_length, do_sample=True, top_k=top_k,
                             temperature=temperature, task_ids=task_ids)['output']
    out_str = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

    in_str = data['in_str']
    pred_bbox, pred_labels = list(), list()
    input_labels, input_bbox, input_mask = list(), list(), list()
    for idx, ostr in enumerate(out_str):
        out_str[idx] = ostr.strip()

        # combine input elements and output elements
        _pred_labels, _pred_bbox = seq_processor.parse_seq(in_str[idx], out_str[idx])

        if _pred_labels is None:
            # something wrong in the model's predictions
            continue

        _pred_labels = torch.tensor(_pred_labels).long()
        _pred_bbox = torch.tensor(_pred_bbox).long()
        pred_labels.append(_pred_labels)
        pred_bbox.append(_pred_bbox)

        input_labels.append(_input_labels[idx])
        input_bbox.append(_input_bbox[idx])
        input_mask.append(_input_mask[idx])

    input_labels = torch.stack(input_labels, dim=0).to(device).detach()
    input_bbox = torch.stack(input_bbox, dim=0).to(device).detach()
    input_mask = torch.stack(input_mask, dim=0).to(device).detach()

    # Compute Metrics
    # label
    pred_labels, pred_mask = utils.to_dense_batch(pred_labels)
    pred_labels = pred_labels.to(device)

    # bbox
    pred_bbox, _ = utils.to_dense_batch(pred_bbox)

    metric = {
        'num_bbox_correct': 0,  # num_bbox_correct,
        'num_bbox': mask.unsqueeze(dim=-1).repeat(1, 1, 4).sum(),  # num_bbox,
        'num_label_correct': 0,  # num_label_correct,
        'num_examples': mask.size(0),
    }
    out = {
        'pred_labels': pred_labels,
        'pred_bboxes': pred_bbox,
        'gold_labels': gold_labels,
        'gold_bboxes': gold_bbox,
        'input_labels': input_labels.to(device).detach(),
        'input_bboxes': input_bbox.to(device).detach(),
        'gold_mask': mask,
        'pred_mask': pred_mask,
        'input_mask': input_mask,
        'pred_str': out_str,
        'input_str': data['in_str'],
        'gold_str': data['out_str']
    }
    return metric, out
