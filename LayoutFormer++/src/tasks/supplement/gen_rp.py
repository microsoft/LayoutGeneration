from typing import Dict
import random
import copy
from itertools import combinations

import torch
import torchvision.transforms as T
import torch.nn.functional as F

from data import transforms
from tasks.refinement import T5LayoutSequence
from tasks.gen_r import RelationTypes, AddCanvasElement, detect_size_relation, detect_loc_relation, get_label_with_index
from model.utils import TransformerSortByDictRelationConstraint
from utils import utils
from evaluation import metrics


class T5LayoutSequenceForGenRP(T5LayoutSequence):

    REL_BEG_TOKEN = '<sep_labels_relations>'
    REL_SEP_TOKEN = '<sep_relations>'
    REL_ELE_SEP_TOKEN = '<sep_ele_rela_ele>'

    def __init__(self, tokenizer, index2label, label2index,
                 add_sep_token: bool = False, discrete_x_grid: int = 128,
                 discrete_y_grid: int = 128, gen_r_add_unk_token: bool = False,
                 gen_r_compact: bool = False) -> None:
        super().__init__(tokenizer, index2label, label2index, add_sep_token)
        self.index2type = RelationTypes.index2type()
        self.discrete_fn = transforms.DiscretizeBoundingBox(discrete_x_grid, discrete_y_grid)
        self.add_unk_token = gen_r_add_unk_token
        print("relation constrained: Add unk token in position & size", self.add_unk_token)
        self.compact_mode = gen_r_compact
        print("relation constrained: make the seq compact", self.compact_mode)

    def build_input_seq(self, labels, bboxes, relations) -> str:
        tokens = list()
        # labels
        for idx in range(len(labels)):
            ele_label = labels[idx]
            ele_bbox = bboxes[idx].tolist()
            tokens.append(self.index2label[int(ele_label)])
            tokens.extend(map(str, ele_bbox[:2]))
            if self.add_unk_token:
                tokens.extend([self.tokenizer.unk_token, self.tokenizer.unk_token])
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(self.SEP_TOKEN)

        # relations
        tokens.append(self.REL_BEG_TOKEN)

        for idx in range(len(relations)):
            label_i = relations[idx][2]
            index_i = relations[idx][3]
            if label_i != 0:
                tokens.append('label_{} index_{}'.format(int(label_i), index_i))
            else:
                tokens.append('label_0')  # canavs
            if not self.compact_mode:
                tokens.append(self.REL_ELE_SEP_TOKEN)
            tokens.append("relation_{}".format(int(relations[idx][4])))
            if not self.compact_mode:
                tokens.append(self.REL_ELE_SEP_TOKEN)
            label_j = relations[idx][0]
            index_j = relations[idx][1]
            if label_j != 0:
                tokens.append('label_{} index_{}'.format(int(label_j), index_j))
            else:
                tokens.append('label_0')
            if idx < len(relations) - 1:
                tokens.append(self.REL_SEP_TOKEN)

        re_seq = ' '.join(tokens)
        return re_seq


class AddSizeRelation():
    def __init__(self, seed=1024, ratio=0.1):
        self.ratio = ratio
        self.generator = random.Random()
        if seed is not None:
            self.generator.seed(seed)
        self.type2index = RelationTypes.type2index()

    def __call__(self, data):

        data['labels_with_canvas_index'] = get_label_with_index(data['labels_with_canvas'])
        N = len(data['labels_with_canvas'])

        rel_all = list(combinations(range(N), 2))
        # size = min(int(len(rel_all)                     * self.ratio), 10)
        size = int(len(rel_all) * self.ratio)
        rel_sample = set(self.generator.sample(rel_all, size))

        relations = []
        for i, j in combinations(range(N), 2):
            bi, bj = data['bboxes_with_canvas'][i], data['bboxes_with_canvas'][j]
            canvas = data['labels_with_canvas'][i] == 0
            if ((i, j) in rel_sample) and (not canvas):
                rel_size = detect_size_relation(bi, bj)
                relations.append([
                    data['labels_with_canvas'][i],
                    data['labels_with_canvas_index'][i],
                    data['labels_with_canvas'][j],
                    data['labels_with_canvas_index'][j],
                    self.type2index[rel_size]
                ])

        data['relations'] = torch.as_tensor(relations).long()

        return data


class T5GenRPDataset:
    def __init__(self, cargs, tokenizer, seq_processor, sort_by_pos: bool = True,
                 shuffle_before_sort_by_label: bool = False,
                 sort_by_pos_before_sort_by_label: bool = False, *args, **kwargs) -> None:
        self.args = cargs
        self.tokenizer = tokenizer
        self.seq_processor = seq_processor
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
        if cargs.gen_r_discrete_before_induce_relations:
            print("Relation Constrained: discrete before relation inductions")
            transform_functions.append(
                transforms.DiscretizeBoundingBox(
                    num_x_grid=cargs.discrete_x_grid,
                    num_y_grid=cargs.discrete_y_grid)
            )
            transform_functions.append(AddCanvasElement(use_discrete=True, discrete_fn=transform_functions[-1]))
            transform_functions.append(AddSizeRelation())
        else:
            transform_functions.append(AddCanvasElement())
            transform_functions.append(AddSizeRelation())
            transform_functions.append(
                transforms.DiscretizeBoundingBox(
                    num_x_grid=cargs.discrete_x_grid,
                    num_y_grid=cargs.discrete_y_grid)
            )
        print("Relation Constrained Generation: ", transform_functions)
        self.transform = T.Compose(transform_functions)
        super().__init__(*args, **kwargs)

    def process(self, data) -> Dict:
        nd = self.transform(copy.deepcopy(data))

        # build layout sequence
        labels = nd['labels']
        gold_bboxes = nd['discrete_gold_bboxes']
        relations = nd['relations']
        in_str = self.seq_processor.build_input_seq(labels, gold_bboxes, relations).lower().strip()
        out_str = self.seq_processor.build_seq(labels, gold_bboxes).lower().strip()

        return {
            'gold_labels': labels,
            'gold_bboxes': gold_bboxes,
            'real_gold_bboxes': nd['gold_bboxes'],
            'in_str': in_str,
            'out_str': out_str,
            'relations': relations,
            'name': data['name']
            }


def format_gen_r_pred_layout(result):
    '''
    1. add canvas element
    2. generate label with index
    '''
    pre_bboxes = torch.tensor(result['pred'][0], dtype=torch.float)
    pre_labels = torch.tensor(result['pred'][1], dtype=torch.long)

    # add canvas element
    x = torch.tensor([[0., 0., 1., 1.]], dtype=torch.float)
    y = torch.tensor([0], dtype=torch.long)
    pre_bboxes_with_canvas = torch.cat([x, pre_bboxes], dim=0)
    pre_labels_with_canvas = torch.cat([y, pre_labels], dim=0)

    # 2. generate label with index
    label_index_map = dict()
    for idx, lid in enumerate(pre_labels_with_canvas):
        _lid = lid.item()
        if _lid not in label_index_map:
            label_index_map[_lid] = idx

    pred_layout = {
        'bboxes_with_canvas': pre_bboxes_with_canvas,
        'labels_with_canvas': pre_labels_with_canvas,
        'labels_with_canvas_index': label_index_map
    }

    return pred_layout


def compute_rel_violation(pred_layout, relations, type2index):
    num_violations = 0
    for r in relations:
        try:
            label_j, index_j, label_i, index_i, rel_type_idx = r.tolist()
            ele_j_idx = pred_layout['labels_with_canvas_index'][label_j] + index_j - 1
            box_j = pred_layout['bboxes_with_canvas'][ele_j_idx]

            ele_i_idx = pred_layout['labels_with_canvas_index'][label_i] + index_i - 1
            box_i = pred_layout['bboxes_with_canvas'][ele_i_idx]

            is_canvas = label_j == 0
            rel_size = int(type2index[detect_size_relation(box_j, box_i)])
            rel_loc = int(
                type2index[detect_loc_relation(box_j, box_i, is_canvas)])

            if (rel_size == rel_type_idx) or (rel_loc == rel_type_idx):
                pass
            else:
                num_violations += 1
        except:
            num_violations += 1
    return num_violations


def gen_rp_inference(model, data, seq_processor, tokenizer, device, top_k=10, temperature=0.7, constraint_fn=None):
    gold_labels, mask = utils.to_dense_batch(data['gold_labels'])
    gold_bbox, _ = utils.to_dense_batch(data['gold_bboxes'])
    padding_mask = ~mask
    max_len = mask.sum(dim=-1).max()
    mask = mask.to(device).detach()

    in_tokenization = tokenizer(data['in_str'], add_eos=True, add_bos=False)
    in_ids = in_tokenization['input_ids'].to(device)
    in_mask = in_tokenization['mask'].to(device)
    in_padding_mask = ~in_mask

    if constraint_fn is not None:
        if isinstance(constraint_fn, TransformerSortByDictRelationConstraint):
            constraint_fn.prepare(data['gold_labels'], data['relations'])
        else:
            constraint_fn.prepare(data['gold_labels'])

    task_ids = None
    if 'task_id' in data:
        task_ids = torch.tensor(data['task_id']).long().to(device)

    decode_max_length = 150
    output_sequences = model(in_ids, in_padding_mask,
                             max_length=decode_max_length,
                             # generation_constraint_fn=constraint_fn,
                             do_sample=True,
                             top_k=10,
                             temperature=0.7,
                             task_ids=task_ids)['output']
    # output_sequences = model(in_ids, in_padding_mask, max_length=decode_max_length,
    #                          generation_constraint_fn=constraint_fn, task_ids=task_ids)['output']
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
    num_bbox_correct, num_bbox = metrics.calculate_bbox_accuracy(gold_bbox, pred_bbox, mask)
    gold_bbox[padding_mask] = 0

    # calculate violation
    num_relations, num_violations = 0, 0
    batch_violate = list()
    for j in range(len(gold_labels)):
        if len(data['in_str'][j].split('<sep_labels_relations>')) == 1:
            continue
        result = {
            'pred': (seq_processor.discrete_fn.continuize(pred_bbox[j].cpu()),
                     pred_labels[j].cpu()),
            'in_str': data['in_str'][j]
        }
        pred_layout = format_gen_r_pred_layout(result)
        v = compute_rel_violation(pred_layout, data['relations'][j],
                                  RelationTypes.type2index())
        batch_violate.append(v)
        num_violations += v
        num_relations += len(data['relations'][j])
    metric = {
        'num_bbox_correct': num_bbox_correct,
        'num_bbox': num_bbox,
        'num_label_correct': num_label_correct,
        'num_examples': num_examples,
        'violation_num': num_violations,
        'rel_num': num_relations
    }
    out = {
        'pred_labels': pred_labels,
        'pred_bboxes': pred_bbox,
        'gold_labels': gold_labels,
        'gold_bboxes': gold_bbox,
        'mask': mask,
        'input_str': data['in_str'],
        'gold_str': data['out_str'],
        'out_str': out_str,
        'pred_str': out_str,
        'name': data['name'],
        'relations': data['relations'],
        'violate_num': batch_violate
    }
    return metric, out
