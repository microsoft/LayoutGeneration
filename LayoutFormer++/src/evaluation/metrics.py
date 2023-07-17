from typing import List
import numpy as np
import multiprocessing as mp
from itertools import chain
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict as OD

import torch
from pytorch_fid.fid_score import calculate_frechet_distance

from evaluation.utils.layoutnet import LayoutNet


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


def compute_overlap_ignore_bg(bbox, label, mask):
    mask = torch.where(label == 4,  False, mask)  # List Item
    mask = torch.where(label == 9,  False, mask)  # Card
    mask = torch.where(label == 11, False, mask)  # Background Image
    mask = torch.where(label == 17, False, mask)  # Modal

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

    ar = torch.nan_to_num(ai / a1)

    return ar.sum(dim=(1, 2)) / mask.float().sum(-1)


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

    return X.sum(-1) / mask.float().sum(-1)


def check_labels(input_labels: List, pred_labels: List):
    if input_labels is None or pred_labels is None:
        return False
    if len(input_labels) != len(pred_labels):
        return False
    is_correct = True
    for il, pl in zip(input_labels, pred_labels):
        if il != pl:
            is_correct = False
            break
    return is_correct


def calculate_label_accuracy(gold, pred, mask, element_wise: bool = False):
    """
    Args:
        gold (torch.Tensor): (batch_size, num_elements,)
        pred (torch.Tensor): (batch_size, num_elements,)
        mask (torch.Tensor): (batch_size, num_elements,),
        element_wise (bool): default False
    """
    diff = gold - pred
    diff[diff != 0] = 1
    diff = 1 - diff
    if element_wise:
        total = mask.sum()
        num_correct = diff.sum()
    else:
        # Only when all elements' labels are correct, return 1
        total = mask.size(0)
        num_correct = (diff.sum(-1) == mask.sum(-1)).int().sum()
    return num_correct, total


def calculate_bbox_accuracy(gold, pred, mask,
                            element_wise: bool = False):
    """
    Args:
        gold ([type]): (batch_size, num_elements, 4)
        pred ([type]): (batch_size, num_elements, 4)
        mask ([type]): (batch_size, num_elements,)
        element_wise: bool
    """
    gold = gold[mask]
    pred = pred[mask]

    diff = gold - pred
    diff[diff != 0] = 1
    diff = 1 - diff
    if element_wise:
        diff = diff.sum(dim=-1)
        diff[diff != 4] = 0
        diff[diff == 4] = 1
        num_correct = diff.sum()
        total = mask.sum()
    else:
        # coordinate wise
        num_correct = diff.sum()
        total = mask.unsqueeze(dim=-1).repeat(1, 1, 4).sum()
    return num_correct, total


def average(scores):
    return sum(scores) / len(scores)
