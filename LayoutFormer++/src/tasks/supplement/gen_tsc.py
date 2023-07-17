from typing import Tuple, List, Dict
import random
import copy
import numpy as np

import torch
import torchvision.transforms as T

from data import transforms
from tasks.refinement import T5LayoutSequence
from utils import utils


class T5LayoutSequenceForGenTSC(T5LayoutSequence):

    SEP_ELE_TYPE_TOKEN = '<sep_c_t>'

    def __init__(self, tokenizer, index2label, label2index,
                 add_sep_token: bool = False,
                 label_size_add_unk_token: bool = False) -> None:
        super().__init__(tokenizer, index2label, label2index, add_sep_token)
        self.label_size_add_unk_token = label_size_add_unk_token
        if self.label_size_add_unk_token:
            print("Add unk token")

    def build_type_size_seq(self, labels, bboxes) -> str:
        tokens = list()
        for idx in range(len(labels)):
            ele_label = labels[idx]
            ele_bbox = bboxes[idx].tolist()
            tokens.append(self.index2label[int(ele_label)])
            if self.label_size_add_unk_token:
                tokens.extend([self.tokenizer.unk_token, self.tokenizer.unk_token])
            tokens.extend(map(str, ele_bbox[2:]))
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.SEP_TOKEN)
        return ' '.join(tokens)

    def parse_seq(self, _, output_str) -> Tuple[List[int], List[Tuple]]:
        return super().parse_seq(output_str)


class T5GenTSCDataset:

    def __init__(self, cargs, tokenizer, seq_processor, sort_by_pos: bool = False,
                 shuffle_before_sort_by_label: bool = False,
                 sort_by_pos_before_sort_by_label: bool = False, *args, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.seq_processor = seq_processor
        self.index2label = seq_processor.index2label
        self.sort_by_pos = sort_by_pos
        self.shuffle_before_sort_by_label = shuffle_before_sort_by_label
        self.sort_by_pos_before_sort_by_label = sort_by_pos_before_sort_by_label
        transform_functions = [
            transforms.ShuffleElements(),
            transforms.DiscretizeBoundingBox(
                num_x_grid=cargs.discrete_x_grid,
                num_y_grid=cargs.discrete_y_grid)
        ]
        self.transform = T.Compose(transform_functions)
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

        N = len(labels)

        in_complete_labels = labels[:max(int(0.5*N), 1)]
        in_complete_bboxes = bboxes[:max(int(0.5*N), 1), :]
        sorted_in_complete_labels, sorted_in_complete_bbox = self._sort_by_labels(in_complete_labels, in_complete_bboxes)
        in_complete_str = self.seq_processor.build_seq(sorted_in_complete_labels, sorted_in_complete_bbox).lower().strip()

        in_type_labels = labels[max(int(0.5*N), 1):]
        in_type_bboxes = bboxes[max(int(0.5*N), 1):, :]
        sorted_in_type_labels, sorted_in_type_bboxes = self._sort_by_labels(in_type_labels, in_type_bboxes)
        in_type_str = self.seq_processor.build_type_size_seq(sorted_in_type_labels, sorted_in_type_bboxes).lower().strip()

        in_str = in_complete_str + ' ' + self.seq_processor.SEP_ELE_TYPE_TOKEN + ' ' + in_type_str

        out_labels = torch.cat((sorted_in_complete_labels, sorted_in_type_labels), 0)
        out_bboxes = torch.cat((sorted_in_complete_bbox, sorted_in_type_bboxes), 0)
        # out_labels, out_bboxes = self._sort_by_labels(labels, gold_bboxes)
        out_str = self.seq_processor.build_seq(out_labels, out_bboxes).lower().strip()

        return {
            'in_str': in_str,
            'out_str': out_str,
            'in_complete_labels': in_complete_labels,
            'in_complete_bboxes': in_complete_bboxes,
            'out_labels': out_labels,
            'out_bboxes': out_bboxes,
            'gold_labels': out_labels,
            'gold_bboxes': out_bboxes,
            'name': data['name']
        }


def gen_tsc_inference(model, data, seq_processor, tokenizer, device, top_k=10, temperature=0.7):
    gold_labels, mask = utils.to_dense_batch(data['gold_labels'])
    gold_bbox, _ = utils.to_dense_batch(data['gold_bboxes'])
    _input_labels, _input_mask = utils.to_dense_batch(data['in_complete_labels'])
    _input_bbox, _ = utils.to_dense_batch(data['in_complete_bboxes'])

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
