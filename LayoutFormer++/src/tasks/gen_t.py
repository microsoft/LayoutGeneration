import copy
from typing import Dict, Tuple

import torch
import torchvision.transforms as T
import torch.nn.functional as F

from tasks.refinement import T5LayoutSequence
from data import transforms
from utils import utils
from evaluation import metrics
from model.layout_transformer.constrained_decoding import TransformerSortByDictLabelSizeConstraint


class T5LayoutSequenceForGenT(T5LayoutSequence):

    GEN_T = 'gen_t'
    GEN_TS = 'gen_ts'

    def __init__(self, task, tokenizer, index2label, label2index, add_sep_token: bool = False,
                 gen_ts_add_unk_token: bool = False,
                 gen_t_add_unk_token: bool = False) -> None:
        super().__init__(tokenizer, index2label, label2index, add_sep_token)
        self.task = task
        self.label_size_add_unk_token = gen_ts_add_unk_token
        if self.label_size_add_unk_token and self.task == self.GEN_TS:
            print("Label Size Constrained Add unk token")

        self.gen_t_add_unk_token = gen_t_add_unk_token
        if self.gen_t_add_unk_token and self.task == self.GEN_T:
            print("Label Constrained Add unk token")

    def build_input_seq(self, labels, bboxes):
        tokens = list()
        for idx in range(len(labels)):
            ele_label = labels[idx]
            ele_bbox = bboxes[idx].tolist()
            tokens.append(self.index2label[int(ele_label)])
            if self.task == self.GEN_TS:
                # label & size
                if self.label_size_add_unk_token:
                    tokens.extend([self.tokenizer.unk_token, self.tokenizer.unk_token])
                tokens.extend(map(str, ele_bbox[2:]))
            else:
                # label
                if self.gen_t_add_unk_token:
                    tokens.extend([self.tokenizer.unk_token] * 4)
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.SEP_TOKEN)
        re_seq = ' '.join(tokens)
        return re_seq


class T5GenTDataset:

    def __init__(self, cargs, tokenizer, seq_processor,
                 sort_by_pos: bool = True,
                 shuffle_before_sort_by_label: bool = False,
                 sort_by_pos_before_sort_by_label: bool = False, *args, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.seq_processor = seq_processor
        self.index2label = self.seq_processor.index2label
        self.sort_by_pos = sort_by_pos

        transform_functions = list()
        if sort_by_pos:
            transform_functions.append(transforms.LexicographicSort())
        else:
            # sort by label
            if shuffle_before_sort_by_label:
                transform_functions.append(transforms.ShuffleElements())
            elif sort_by_pos_before_sort_by_label:
                transform_functions.append(transforms.LexicographicSort())
            transform_functions.append(transforms.LabelDictSort(seq_processor.index2label))
        transform_functions.append(
            transforms.DiscretizeBoundingBox(
                num_x_grid=cargs.discrete_x_grid,
                num_y_grid=cargs.discrete_y_grid)
        )
        print("Constrained Generation: ", transform_functions)
        self.transform = T.Compose(transform_functions)
        super().__init__(*args, **kwargs)

    def _sort_by_labels(self, labels, bboxes):
        _labels = labels.tolist()
        idx2label = [[idx, self.index2label[_labels[idx]]]
                     for idx in range(len(labels))]
        idx2label_sorted = sorted(idx2label, key=lambda x: x[1])
        idx_sorted = [d[0] for d in idx2label_sorted]
        return labels[idx_sorted], bboxes[idx_sorted]

    def process(self, data):
        '''
        for each layout
        discretize and transform to seq
        '''
        # print("data before transform : {}".format(data))
        nd = self.transform(copy.deepcopy(data))
        labels = nd['labels']
        bboxes = nd['discrete_bboxes']

        if self.sort_by_pos:
            # sort by dictionary
            sorted_labels, sorted_bboxes = self._sort_by_labels(labels, bboxes)
        else:
            sorted_labels, sorted_bboxes = labels, bboxes
        in_str = self.seq_processor.build_input_seq(sorted_labels, sorted_bboxes).lower().strip()
        gold_bboxes = nd['discrete_gold_bboxes']
        out_str = self.seq_processor.build_seq(labels, gold_bboxes).lower().strip()
        return {
            'in_str': in_str,
            'out_str': out_str,
            'gold_labels': labels,
            'gold_bboxes': gold_bboxes,
            'input_labels': labels,
            'input_bboxes': bboxes,
            'name': data['name']
        }


def gen_t_inference(model, data, seq_processor, tokenizer, device,
                    constraint_fn=None) -> Tuple[Dict, Dict]:
    gold_labels, mask = utils.to_dense_batch(data['gold_labels'])
    gold_bbox, _ = utils.to_dense_batch(data['gold_bboxes'])
    input_labels, _ = utils.to_dense_batch(data['input_labels'])
    input_bbox, _ = utils.to_dense_batch(data['input_bboxes'])
    padding_mask = ~mask
    max_len = mask.sum(dim=-1).max()

    in_tokenization = tokenizer(data['in_str'], add_eos=True, add_bos=False)
    in_ids = in_tokenization['input_ids'].to(device)
    in_mask = in_tokenization['mask'].to(device)
    in_padding_mask = ~in_mask

    if constraint_fn is not None:
        if isinstance(constraint_fn, TransformerSortByDictLabelSizeConstraint):
            constraint_fn.prepare(data['gold_labels'], data['gold_bboxes'])
        else:
            constraint_fn.prepare(data['gold_labels'])

    task_ids = None
    if 'task_id' in data:
        task_ids = torch.tensor(data['task_id']).long().to(device)

    decode_max_length = 120
    # output_sequences = model(in_ids, in_padding_mask,
    #                          max_length=decode_max_length,
    #                          generation_constraint_fn=constraint_fn,
    #                          task_ids=task_ids)['output']

    output_sequences = model(in_ids, in_padding_mask,
                             max_length=decode_max_length,
                             constrained_decoding=True,
                             generation_constraint_fn=constraint_fn,
                             task_ids=task_ids)['output']

    out_str = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

    pred_bbox, pred_labels = list(), list()
    for idx, ostr in enumerate(out_str):
        out_str[idx] = ostr.strip()
        _pred_labels, _pred_bbox = seq_processor.parse_seq(out_str[idx])

        if _pred_labels is None:
            # something wrong in the model's predictions
            pred_bbox.append(in_ids.new_zeros(max_len, 4).long())
            pred_labels.append(in_ids.new_zeros(max_len).long())
            continue

        diff = max_len - len(_pred_labels)
        _pred_labels = torch.tensor(_pred_labels).long().to(device)
        _pred_bbox = torch.tensor(_pred_bbox).long().to(device)
        if diff > 0:
            _pred_labels = F.pad(_pred_labels, (0, diff,),
                                 'constant', seq_processor.error_label_id)
            _pred_bbox = F.pad(_pred_bbox, (0, 0, 0, diff,), 'constant', 0)
        elif diff < 0:
            _pred_labels = _pred_labels[:max_len]
            _pred_bbox = _pred_bbox[:max_len, :]
        pred_labels.append(_pred_labels)
        pred_bbox.append(_pred_bbox)

    # Compute Metrics
    mask = mask.to(device).detach()

    # label
    pred_labels = torch.stack(pred_labels, dim=0).to(device).detach()
    gold_labels[padding_mask] = -100
    gold_labels = gold_labels.to(device).detach()
    num_label_correct, num_examples = metrics.calculate_label_accuracy(pred_labels, gold_labels, mask,
                                                                       element_wise=False)
    gold_labels[padding_mask] = 0

    # bbox
    pred_bbox = torch.stack(pred_bbox, dim=0).to(device).detach()
    gold_bbox[padding_mask] = -100
    gold_bbox = gold_bbox.to(device).detach()
    num_bbox_correct, num_bbox = metrics.calculate_bbox_accuracy(
        gold_bbox, pred_bbox, mask)
    gold_bbox[padding_mask] = 0

    metric = {
        'num_bbox_correct': num_bbox_correct,
        'num_bbox': num_bbox,
        'num_label_correct': num_label_correct,
        'num_examples': num_examples,
    }
    out = {
        'pred_labels': pred_labels,
        'pred_bboxes': pred_bbox,
        'gold_labels': gold_labels,
        'gold_bboxes': gold_bbox,
        'input_labels': input_labels.to(device).detach(),
        'input_bboxes': input_bbox.to(device).detach(),
        'mask': mask,
        'pred_str': out_str,
        'input_str': data['in_str'],
        'gold_str': data['out_str']
    }
    return metric, out
