import multiprocessing as mp
import os
import re
from collections import Counter
from collections import OrderedDict as OD
from itertools import chain

import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance
from scipy.optimize import linear_sum_assignment

from evaluation.layout_net import LayoutNet
from layout_placement.placement_utils.utils import CANVAS_SIZE, LABEL, LABEL2ID

from .DOCSIM import get_layout_sim


class LayoutFID():
    def __init__(self, max_num_elements: int,
                 num_labels: int, net_path: str, device: str = 'cpu'):

        self.model = LayoutNet(num_labels, max_num_elements).to(device)

        # load pre-trained LayoutNet
        state_dict = torch.load(net_path, map_location=device)
        # remove "module" prefix if necessary
        state = OD([(key.split("module.")[-1], state_dict[key]) for key in state_dict])

        self.model.load_state_dict(state)
        self.model.requires_grad_(False)
        self.model.eval()

        self.real_features = []
        self.fake_features = []

    def collect_features(self, bbox, label, padding_mask, real=False):
        if real and type(self.real_features) != list:
            return

        feats = self.model.extract_features(bbox.detach(), label, padding_mask)
        features = self.real_features if real else self.fake_features
        features.append(feats.cpu().numpy())

    def compute_score(self):
        feats_1 = np.concatenate(self.fake_features)
        self.fake_features = []

        if type(self.real_features) == list:
            feats_2 = np.concatenate(self.real_features)
            self.real_features = feats_2
        else:
            feats_2 = self.real_features

        mu_1 = np.mean(feats_1, axis=0)
        sigma_1 = np.cov(feats_1, rowvar=False)
        mu_2 = np.mean(feats_2, axis=0)
        sigma_2 = np.cov(feats_2, rowvar=False)

        return calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)


class ConstraintAcc:
    LEFT_THRES = 0.45
    RIGHT_THRES = 0.55
    TOP_THRES = 0.45
    BOTTOM_THRES = 0.55
    SMALL_THRES = 0.1
    LARGE_THRES = 0.5

    def __init__(self, predictions, dataset) -> None:
        self.predictions = predictions
        self.dataset = dataset
        self.num_total_type = 0
        self.num_correct_type = 0
        self.num_total_ps = 0
        self.num_correct_ps = 0
        self.num_total_hierarchy = 0
        self.num_correct_hierarchy = 0

    def _type_acc(self, layout_seq, constraint):
        _pattern = '(' + '|'.join(LABEL[self.dataset]) + ')'
        pred_types = re.findall(_pattern, layout_seq)
        gold_types = re.findall(_pattern, constraint)
        self.num_total_type += len(gold_types)
        for _type in gold_types:
            if _type in pred_types:
                self.num_correct_type += 1
                pred_types.remove(_type)

    def _ps_acc(self, layout_seq, constraint):
        constraint_pattern = '(' + '|'.join(LABEL[self.dataset]) + ') ' + '(?!undefined undefined)([^ ]+) ([^ ]+)'
        element_pattern = '(' + '|'.join(LABEL[self.dataset]) + ') ' + '(\d+) (\d+) (\d+) (\d+)'
        ps_constraints = re.findall(constraint_pattern, constraint)
        elements = re.findall(element_pattern, layout_seq)

        if len(ps_constraints) > 0:
            self.num_total_ps += len(ps_constraints)
            if len(elements) > 0:
                scaled_width = CANVAS_SIZE[self.dataset][0]
                scaled_height = max([int(ele[2])+int(ele[4]) for ele in elements])
                match_graph = np.zeros((len(ps_constraints), len(elements)))
                for i in range(len(ps_constraints)):
                    for j in range(len(elements)):
                        if ps_constraints[i][0] == elements[j][0] and \
                            self._is_satisfy_ps(ps_constraints[i][1], ps_constraints[i][2], elements[j][1:], scaled_width, scaled_height):
                            match_graph[i][j] = 1
                row_ind, col_ind = linear_sum_assignment(-match_graph)
                max_weight = match_graph[row_ind, col_ind].sum()
                self.num_correct_ps += max_weight

    def _hierarchy_acc(self, layout_seq, constraint):
        hierarchy_pattern = '\[.*?\]'
        element_pattern = '(' + '|'.join(LABEL[self.dataset]) + ')'
        gold_hierarchy = re.findall(hierarchy_pattern, constraint)
        pred_hierarchy = re.findall(hierarchy_pattern, layout_seq)
        if len(gold_hierarchy) > 0:
            self.num_total_hierarchy += len(gold_hierarchy)
            gold_hierarchy = [Counter(re.findall(element_pattern, item)) for item in gold_hierarchy]
            pred_hierarchy = [Counter(re.findall(element_pattern, item)) for item in pred_hierarchy]
            for i in range(len(gold_hierarchy)):
                for j in range(len(pred_hierarchy)):
                    if gold_hierarchy[i].items() <= pred_hierarchy[j].items():
                        self.num_correct_hierarchy += 1
                        pred_hierarchy.pop(j)
                        break

    def _is_satisfy_ps(self, pc, sc, position, scaled_width, scaled_height):
        position = list(map(int, position))
        if pc == 'left':
            if position[0] + position[2] / 2 > scaled_width * self.LEFT_THRES:
                return False
        if pc == 'right':
            if position[0] + position[2] / 2 < scaled_width * self.RIGHT_THRES:
                return False
        if pc == 'top':
            if position[1] + position[3] / 2 > scaled_height * self.TOP_THRES:
                return False
        if pc == 'bottom':
            if position[1] + position[3] / 2 < scaled_height * self.BOTTOM_THRES:
                return False
        if sc == 'small':
            if position[2] * position[3] > scaled_height * scaled_width * self.SMALL_THRES:
                return False
        if sc == 'large':
            if position[2] * position[3] < scaled_height * scaled_width * self.LARGE_THRES:
                return False
        return True

    def __call__(self):
        for prediction in self.predictions:
            constraint = prediction['constraint']
            pred_layout_seqs = prediction['prediction']
            for pred_layout_seq in pred_layout_seqs:
                self._type_acc(pred_layout_seq, constraint)
                self._ps_acc(pred_layout_seq, constraint)
                self._hierarchy_acc(pred_layout_seq, constraint)
        metrics = {'type_acc': 0.0, 'ps_acc': 0.0, 'hier_acc': 0.0}
        if self.num_total_type != 0: metrics['type_acc'] = self.num_correct_type / self.num_total_type
        if self.num_total_ps != 0: metrics['ps_acc'] = self.num_correct_ps / self.num_total_ps
        if self.num_total_hierarchy != 0: metrics['hier_acc'] = self.num_correct_hierarchy / self.num_total_hierarchy
        return metrics


class UniqueMatch:
    SIMILARITY_THRES = 0.7

    def __init__(self, gold_label, gold_pos, pred_label, pred_pos, num_return_sequences):
        gold_label = [item.tolist() for item in gold_label]
        gold_pos = [item.tolist() for item in gold_pos]
        pred_label = [item.tolist() for item in pred_label]
        pred_pos = [item.tolist() for item in pred_pos]

        self.num_return_sequences = num_return_sequences
        self.groups = len(pred_label) // num_return_sequences
        self.gold_label = gold_label
        self.gold_pos = gold_pos
        self.pred_label = [pred_label[group_id * num_return_sequences: (group_id + 1) * num_return_sequences] for group_id in range(self.groups)]
        self.pred_pos = [pred_pos[group_id * num_return_sequences: (group_id + 1) * num_return_sequences] for group_id in range(self.groups)]

    def _retrive_from_training_data(self, args):
        _pred_label, _pred_pos, _gold_label, _gold_pos = args
        _retrived_list = []

        for i in range(len(_pred_label)):
            if len(_pred_label[i]) == 0: continue
            scores = []
            self_sim = get_layout_sim(_pred_pos[i], _pred_label[i], _pred_pos[i], _pred_label[i])
            for j in range(len(_gold_label)):
                sim = get_layout_sim(_pred_pos[i], _pred_label[i], _gold_pos[j], _gold_label[j])
                scores.append(sim)
            scores = torch.tensor(scores)
            retrived_id = torch.argmax(scores)
            if scores[retrived_id] > self_sim * self.SIMILARITY_THRES: _retrived_list.append(retrived_id.item())
        return list(set(_retrived_list))

    def __call__(self):
        args = zip(self.pred_label, self.pred_pos, [self.gold_label] * self.groups, [self.gold_pos] * self.groups)
        with mp.Pool() as p:
            retrived_list = p.map(self._retrive_from_training_data, args)
        retrived_list = list(chain.from_iterable(retrived_list))
        return {'UM': len(retrived_list) / self.groups / self.num_return_sequences,
                'UM_between_group': len(set(retrived_list)) / self.groups / self.num_return_sequences,}


def compute_iou(box_1, box_2):
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, torch.Tensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = box_1.T
    l2, t2, r2, b2 = box_2.T
    w1 = lib.where((r1 - l1) == 0, 1e-7, r1 - l1)
    h1 = lib.where((b1 - t1) == 0, 1e-7, b1 - t1)
    w2 = lib.where((r2 - l2) == 0, 1e-7, r2 - l2)
    h2 = lib.where((b2 - t2) == 0, 1e-7, b2 - t2)

    a1, a2 = w1 * h1, w2 * h2

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max),
                   lib.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    return iou


def __compute_maximum_iou_for_layout(layout_1, layout_2):
    score = 0.
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        n = len(_bi)
        ii, jj = np.meshgrid(range(n), range(n))
        ii, jj = ii.flatten(), jj.flatten()
        iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, n)
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    return score / N


def __compute_maximum_iou(layouts_1_and_2):
    layouts_1, layouts_2 = layouts_1_and_2
    N, M = len(layouts_1), len(layouts_2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray([
        __compute_maximum_iou_for_layout(layouts_1[i], layouts_2[j])
        for i, j in zip(ii, jj)
    ]).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]


def __get_cond2layouts(layout_list):
    out = dict()
    for bs, ls in layout_list:
        cond_key = str(sorted(ls.tolist()))
        if cond_key not in out.keys():
            out[cond_key] = [(bs, ls)]
        else:
            out[cond_key].append((bs, ls))
    return out


def compute_maximum_iou(layouts_1, layouts_2, n_jobs=None):
    c2bl_1 = __get_cond2layouts(layouts_1)
    keys_1 = set(c2bl_1.keys())
    c2bl_2 = __get_cond2layouts(layouts_2)
    keys_2 = set(c2bl_2.keys())
    keys = list(keys_1.intersection(keys_2))
    args = [(c2bl_1[key], c2bl_2[key]) for key in keys]
    with mp.Pool(n_jobs) as p:
        scores = p.map(__compute_maximum_iou, args)
    scores = np.asarray(list(chain.from_iterable(scores)))
    return scores.mean().item()


def compute_overlap_ignore_bg(dataset_name, bbox, label, mask):
    if dataset_name == 'web':
        mask = torch.where(label == 6,  False, mask)  # List Item
    elif dataset_name == 'rico':
        mask = torch.where(label == 3,  False, mask)  # List Item
        mask = torch.where(label == 8,  False, mask)  # Card
        mask = torch.where(label == 10, False, mask)  # Background Image
        mask = torch.where(label == 16, False, mask)  # Modal
    return compute_overlap(bbox, mask)


def compute_overlap(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.3 Overlapping Loss

    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0)
    bbox = bbox.permute(2, 0, 1)

    l1, t1, r1, b1 = bbox.unsqueeze(-1)
    l2, t2, r2, b2 = bbox.unsqueeze(-2)
    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max),
                     torch.zeros_like(a1[0]))

    diag_mask = torch.eye(a1.size(1), dtype=torch.bool,
                          device=a1.device)
    ai = ai.masked_fill(diag_mask, 0)

    ar = ai / a1
    ar=torch.from_numpy(np.nan_to_num(ar.numpy()))
    score=torch.from_numpy(np.nan_to_num((ar.sum(dim=(1, 2)) / mask.float().sum(-1)).numpy()))
    return (score).mean().item()


def compute_alignment(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.4 Alignment Loss

    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = bbox
    xc = (xr + xl) / 2
    yc = (yt + yb) / 2
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)

    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.), 0.)

    X = -torch.log(1 - X)
    score=torch.from_numpy(np.nan_to_num((X.sum(-1)/mask.float().sum(-1)))).numpy()
    return (score).mean().item()


def decapulate(bbox):
    if len(bbox.size()) == 2:
        x1, y1, x2, y2 = bbox.T
    else:
        x1, y1, x2, y2 = bbox.permute(2, 0, 1)
    return x1, y1, x2, y2


def convert_ltwh_to_ltrb(bbox):
    l, t, w, h = decapulate(bbox)
    r = l + w
    b = t + h
    return torch.stack([l, t, r, b], axis=-1)


def create_fid_model(dataset, device='cpu'):
    if dataset == 'web':
        fid_net_path = os.path.join(os.getcwd(), './evaluation/net/fid_web_finetune.pth.tar')
        fid_model = LayoutFID(max_num_elements=76, num_labels=12,
                              net_path=fid_net_path, device=device)
    elif dataset == 'rico':
        fid_net_path = os.path.join(os.getcwd(), './evaluation/net/fid_rico.pth.tar')
        fid_model = LayoutFID(max_num_elements=20, num_labels=25,
                              net_path=fid_net_path, device=device)
    return fid_model


def collect_layouts(bboxes, labels, mask, layouts):
    for j in range(labels.size(0)):
        _mask = mask[j]
        box = bboxes[j][_mask].cpu().numpy()
        label = labels[j][_mask].cpu().numpy()
        layouts.append((box, label))
    return layouts, bboxes


def compute_metrics(predictions, golds, training_golds, dataset_name):
    num_return_sequences = len(predictions[0])
    pred_layout_seq = list(chain.from_iterable(predictions))
    gold_layout_seq = [d['plain_layout_seq'] for d in training_golds]   # for mIoU and FID
    gold_pos, gold_label, gold_pos_t, gold_label_t, gold_padding_mask = _collect_attribute(gold_layout_seq, dataset_name)
    pred_pos, pred_label, pred_pos_t, pred_label_t, pred_padding_mask = _collect_attribute(pred_layout_seq, dataset_name)
    predictions = [{
        'line_id': idx,
        'region_id': golds[idx]['region_id'],
        'text': golds[idx]['text'],
        'constraint': golds[idx]['execution'],
        'gold_layout_seq': golds[idx]['plain_layout_seq'],
        'prediction': predictions[idx]
    } for idx in range(len(predictions))]

    metrics = {}
    gold_layouts = []
    pred_layouts = []
    metrics['overlap_real'] = compute_overlap_ignore_bg(dataset_name, gold_pos_t, gold_label_t, gold_padding_mask)
    metrics['alignment_real'] = compute_alignment(gold_pos_t, gold_padding_mask)
    metrics['overlap_pred'] = compute_overlap_ignore_bg(dataset_name, pred_pos_t, pred_label_t, pred_padding_mask)
    metrics['alignment_pred'] = compute_alignment(pred_pos_t, pred_padding_mask)
    gold_layouts, _ = collect_layouts(gold_pos_t, gold_label_t, gold_padding_mask, gold_layouts)
    pred_layouts, _ = collect_layouts(pred_pos_t, pred_label_t, pred_padding_mask, pred_layouts)
    metrics['mIoU'] = compute_maximum_iou(pred_layouts, gold_layouts)
    fid_model = create_fid_model(dataset_name)
    fid_model.collect_features(pred_pos_t.to('cpu'), pred_label_t.to('cpu'), (~pred_padding_mask).to('cpu'))
    fid_model.collect_features(gold_pos_t.to('cpu'), gold_label_t.to('cpu'), (~gold_padding_mask).to('cpu'), real=True)
    metrics['FID'] = fid_model.compute_score()

    constraint_acc = ConstraintAcc(predictions, dataset_name)
    metrics.update(constraint_acc())
    unique_match = UniqueMatch(gold_label, gold_pos, pred_label, pred_pos, num_return_sequences)
    metrics.update(unique_match())
    return predictions, metrics


def _collect_attribute(seq_list, dataset):
    label, pos = [], []
    for layout_seq in seq_list:
        _label, _pos = _findall_elements(layout_seq, dataset)
        label.append(_label)
        pos.append(_pos)
    pos_t = torch.zeros(len(label), 200, 4)
    label_t = torch.zeros(len(label), 200).long()
    padding_mask_t = torch.zeros(len(label), 200).bool()
    for i, (_label, _pos) in enumerate(zip(label, pos)):
        if len(_label) == 0: continue
        pos_t[i][0: len(_label)] = convert_ltwh_to_ltrb(_pos)
        label_t[i][0: len(_label)] = _label
        padding_mask_t[i][0: len(_label)] = 1
    return pos, label, pos_t, label_t, padding_mask_t


def _findall_elements(s, dataset):
    labels = LABEL[dataset]
    canvas_width, canvas_height = CANVAS_SIZE[dataset]
    element_pattern = '(' + '|'.join(labels) + ')' + r' (\d+) (\d+) (\d+) (\d+)'
    label2id = LABEL2ID[dataset]

    elements = re.findall(element_pattern, s)
    label = torch.tensor(
        [label2id[element[0]] for element in elements]
    )
    position = torch.tensor([
        [int(element[1]) / canvas_width,
        int(element[2]) / canvas_height,
        int(element[3]) / canvas_width,
        int(element[4]) / canvas_height]
        for element in elements
    ])
    return label, position
