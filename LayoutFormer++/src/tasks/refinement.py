from typing import Dict, List, Tuple
import re
import copy

import torch
import torchvision.transforms as T
import torch.nn.functional as F

from data import transforms, LayoutSequence
from utils import utils
from evaluation import metrics


class T5LayoutSequence(LayoutSequence):

    SEP_TOKEN = '|'

    def __init__(self, tokenizer, index2label, label2index, add_sep_token: bool = False) -> None:
        super().__init__(tokenizer)
        self.index2label = index2label
        self._error_label_id = 0

        self.label2index = dict()
        for key, value in label2index.items():
            new_key = key.lower().strip()
            if new_key not in self.label2index:
                self.label2index[new_key] = value

        self.add_sep_token = add_sep_token
        if self.add_sep_token:
            print("Add SEP token")

    @property
    def error_label_id(self):
        return self._error_label_id

    def build_seq(self, labels, bboxes) -> str:
        tokens = list()
        for idx in range(len(labels)):
            ele_label = labels[idx]
            ele_bbox = bboxes[idx].tolist()
            tokens.append(self.index2label[int(ele_label)])
            tokens.extend(map(str, ele_bbox))
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.SEP_TOKEN)
        return ' '.join(tokens)

    def parse_seq(self, output_str) -> Tuple[List[int], List[Tuple]]:
        labels, bbox = list(), list()
        if self.add_sep_token:
            _output_str = output_str + " |"
            for match in re.findall(r'(([\w\-\/\s]+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s\|)', _output_str):
                g_label = match[1].strip()
                g_bbox = [int(match[2+i]) for i in range(4)]
                labels.append(self.label2index.get(g_label, self.error_label_id))
                bbox.append(g_bbox)
        else:
            tokens = output_str.split()
            num_tokens = len(tokens)
            idx = 0
            while idx < num_tokens:
                curr_labels, curr_bbox = list(), list()
                while idx < num_tokens and not re.match(r'^\d+$', tokens[idx]):
                    curr_labels.append(tokens[idx])
                    idx += 1
                while idx < num_tokens and re.match(r'^\d+$', tokens[idx]):
                    curr_bbox.append(int(tokens[idx]))
                    idx += 1
                if len(curr_labels) > 0 and len(curr_bbox) == 4:
                    # If the label is incorrect, then assign it a special label index
                    labels.append(self.label2index.get(" ".join(curr_labels).strip(), self.error_label_id))
                    bbox.append(curr_bbox)
                else:
                    return None, None
        if len(labels) == 0:
            return None, None
        return labels, bbox


class T5RefinementDataset:
    def __init__(self, cargs, tokenizer, seq_processor, task_prompt: str = None,
                 sort_by_pos: bool = True, shuffle_before_sort_by_label: bool = False,
                 sort_by_pos_before_sort_by_label: bool = False, *args, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.seq_processor = seq_processor
        transform_functions = [
            transforms.AddGaussianNoise(
                mean=cargs.gaussian_noise_mean,
                std=cargs.gaussian_noise_std,
                bernoulli_beta=cargs.train_bernoulli_beta)
        ]
        if sort_by_pos:
            transform_functions.append(transforms.LexicographicSort())
        else:
            if shuffle_before_sort_by_label:
                print("Shuffle elements before sort by label")
                transform_functions.append(transforms.ShuffleElements())
            elif sort_by_pos_before_sort_by_label:
                print("Sort by pos before sort by label")
                transform_functions.append(transforms.LexicographicSort())
            transform_functions.append(transforms.LabelDictSort(self.seq_processor.index2label))
        transform_functions.append(
            transforms.DiscretizeBoundingBox(
                num_x_grid=cargs.discrete_x_grid,
                num_y_grid=cargs.discrete_y_grid)
        )
        print("Refinement: ", transform_functions)
        self.transform = T.Compose(transform_functions)
        self.task_prompt = task_prompt
        print(f"Using task prompt: {task_prompt}")
        super().__init__(*args, **kwargs)

    def process(self, data) -> Dict:
        nd = self.transform(copy.deepcopy(data))

        # build layout sequence
        labels = nd['labels']
        bboxes = nd['discrete_bboxes']
        gold_bboxes = nd['discrete_gold_bboxes']
        in_str = self.seq_processor.build_seq(labels, bboxes).lower().strip()
        out_str = self.seq_processor.build_seq(labels, gold_bboxes).lower().strip()
        if self.task_prompt is not None:
            in_str = "{} {}".format(self.task_prompt, in_str)
        return {
            'in_str': in_str,
            'out_str': out_str,
            'gold_labels': labels,
            'gold_bboxes': gold_bboxes,
            'input_labels': labels,
            'input_bboxes': bboxes,
            'name': data['name']
        }

    def filter_by_length(self, data: List[Dict]) -> List[Dict]:
        filtered_data = list()
        for item in data:
            in_str = self.process(item)['out_str']
            in_tokenization = self.tokenizer(in_str, return_tensors='pt', truncation=False)
            in_ids = in_tokenization.input_ids
            if in_ids.size(1) <= 512:
                filtered_data.append(item)
        print(f"Remove {len(data) - len(filtered_data)} examples")
        return filtered_data


def refinement_inference(model, data, seq_processor, tokenizer, device,
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

    if constraint_fn:
        constraint_fn.prepare(data['gold_labels'])

    task_ids = None
    if 'task_id' in data:
        task_ids = torch.tensor(data['task_id']).long().to(device)

    decode_max_length = in_ids.size(-1) + 1
    output_sequences = model(in_ids, in_padding_mask,
                             max_length=decode_max_length,
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
