import copy
from typing import Dict, Tuple

import torch
import torchvision.transforms as T

from data import transforms
from utils import utils


class T5UGenDataset:

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
        print("ugen Generation: ", transform_functions,
              "Sort by Pos: ", sort_by_pos)
        self.transform = T.Compose(transform_functions)
        super().__init__(*args, **kwargs)

    def process(self, data) -> Dict:
        nd = self.transform(copy.deepcopy(data))

        # build layout sequence
        labels = nd['labels']
        bboxes = nd['discrete_gold_bboxes']
        gold_bboxes = nd['discrete_gold_bboxes']

        in_labels = labels[:0]
        in_bboxes = bboxes[:0]
        in_str = self.seq_processor.build_seq(in_labels, in_bboxes).lower().strip()
        out_labels = labels
        out_bboxes = bboxes
        out_str = self.seq_processor.build_seq(out_labels, out_bboxes).lower().strip()
        return {
            'in_str': in_str,
            'out_str': out_str,
            'input_labels': in_labels,
            'input_bboxes': in_bboxes,
            'out_labels': out_labels,
            'out_bboxes': out_bboxes,
            'gold_labels': labels,
            'gold_bboxes': gold_bboxes,
            'name': data['name']
        }


def ugen_inference(model, data, seq_processor, tokenizer, device, top_k=10, temperature=0.7) -> Tuple[Dict, Dict]:
    gold_labels, mask = utils.to_dense_batch(data['gold_labels'])
    gold_labels = gold_labels.to(device)
    mask = mask.to(device)
    padding_mask = ~mask
    gold_labels[padding_mask] = 0

    mask = mask.to(device).detach()

    gold_bbox, _ = utils.to_dense_batch(data['gold_bboxes'])
    gold_bbox = gold_bbox.to(device)
    gold_bbox[padding_mask] = 0

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
    for idx, ostr in enumerate(out_str):
        out_str[idx] = ostr.strip()

        # combine input elements and output elements
        _pred_labels, _pred_bbox = seq_processor.parse_seq(in_str[idx], out_str[idx])

        if _pred_labels is None or len(_pred_labels) == 0:
            # something wrong in the model's predictions
            continue

        _pred_labels = torch.tensor(_pred_labels).long()
        _pred_bbox = torch.tensor(_pred_bbox).long()
        pred_labels.append(_pred_labels)
        pred_bbox.append(_pred_bbox)

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
        'gold_mask': mask,
        'pred_mask': pred_mask,
        'pred_str': out_str,
        'input_str': data['in_str'],
        'gold_str': data['out_str']
    }
    return metric, out
