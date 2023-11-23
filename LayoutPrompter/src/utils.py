import json
import os
import re
from collections import Counter
from typing import List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ID2LABEL = {
    "publaynet": {1: "text", 2: "title", 3: "list", 4: "table", 5: "figure"},
    "rico": {
        1: "text",
        2: "image",
        3: "icon",
        4: "list item",
        5: "text button",
        6: "toolbar",
        7: "web view",
        8: "input",
        9: "card",
        10: "advertisement",
        11: "background image",
        12: "drawer",
        13: "radio button",
        14: "checkbox",
        15: "multi-tab",
        16: "pager indicator",
        17: "modal",
        18: "on/off switch",
        19: "slider",
        20: "map view",
        21: "button bar",
        22: "video",
        23: "bottom navigation",
        24: "number stepper",
        25: "date picker",
    },
    "posterlayout": {1: "text", 2: "logo", 3: "underlay"},
    "webui": {
        0: "text",
        1: "link",
        2: "button",
        3: "title",
        4: "description",
        5: "image",
        6: "background",
        7: "logo",
        8: "icon",
        9: "input",
    },
}


CANVAS_SIZE = {
    "rico": (90, 160),
    "publaynet": (120, 160),
    "posterlayout": (102, 150),
    "webui": (120, 120),
}


RAW_DATA_PATH = lambda x: os.path.join(os.path.dirname(__file__), f"../dataset/{x}/raw")


LAYOUT_DOMAIN = {
    "rico": "android",
    "publaynet": "document",
    "posterlayout": "poster",
    "webui": "web",
}


def clean_text(text: str, remove_summary: bool = False):
    if remove_summary:
        text = re.sub(r"#.*?#", "", text)
    text = text.replace("[#]", " ")
    text = text.replace("#", " ")
    text = text.replace("\n", " ")
    text = text.replace(",", ", ")
    text = text.replace(".", ". ").strip()
    text = re.sub(r"[ ]+", " ", text)
    return text


def read_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def read_pt(filename):
    with open(filename, "rb") as f:
        return torch.load(f)


def write_pt(filename, obj):
    with open(filename, "wb") as f:
        torch.save(obj, f)


def convert_ltwh_to_ltrb(bbox):
    if len(bbox.size()) == 1:
        l, t, w, h = bbox
        r = l + w
        b = t + h
        return l, t, r, b
    l, t, w, h = decapulate(bbox)
    r = l + w
    b = t + h
    return torch.stack([l, t, r, b], axis=-1)


def decapulate(bbox):
    if len(bbox.size()) == 2:
        x1, y1, x2, y2 = bbox.T
    else:
        x1, y1, x2, y2 = bbox.permute(2, 0, 1)
    return x1, y1, x2, y2


def detect_size_relation(b1, b2):
    REL_SIZE_ALPHA = 0.1
    a1, a2 = b1[2] * b1[3], b2[2] * b2[3]
    a1_sm = (1 - REL_SIZE_ALPHA) * a1
    a1_lg = (1 + REL_SIZE_ALPHA) * a1

    if a2 <= a1_sm:
        return "smaller"

    if a1_sm < a2 and a2 < a1_lg:
        return "equal"

    if a1_lg <= a2:
        return "larger"

    raise RuntimeError(b1, b2)


def detect_loc_relation(b1, b2, canvas=False):
    if canvas:
        yc = b2[1] + b2[3] / 2
        y_sm, y_lg = 1.0 / 3, 2.0 / 3

        if yc <= y_sm:
            return "top"

        if y_sm < yc and yc < y_lg:
            return "center"

        if y_lg <= yc:
            return "bottom"

    else:
        l1, t1, r1, b1 = convert_ltwh_to_ltrb(b1)
        l2, t2, r2, b2 = convert_ltwh_to_ltrb(b2)

        if b2 <= t1:
            return "top"

        if b1 <= t2:
            return "bottom"

        if t1 < b2 and t2 < b1:
            if r2 <= l1:
                return "left"

            if r1 <= l2:
                return "right"

            if l1 < r2 and l2 < r1:
                return "center"

    raise RuntimeError(b1, b2, canvas)


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
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max), torch.zeros_like(a1[0]))

    diag_mask = torch.eye(a1.size(1), dtype=torch.bool, device=a1.device)
    ai = ai.masked_fill(diag_mask, 0)

    ar = ai / a1
    ar = torch.from_numpy(np.nan_to_num(ar.numpy()))
    score = torch.from_numpy(
        np.nan_to_num((ar.sum(dim=(1, 2)) / mask.float().sum(-1)).numpy())
    )
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
    X[:, :, idx, idx] = 1.0
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.0
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.0), 0.0)

    X = -torch.log(1 - X)
    score = torch.from_numpy(np.nan_to_num((X.sum(-1) / mask.float().sum(-1)))).numpy()
    return (score).mean().item()


def compute_maximum_iou(
    labels_1: torch.Tensor,
    bboxes_1: torch.Tensor,
    labels_2: List[torch.Tensor],
    bboxes_2: List[torch.Tensor],
    labels_weight: float = 0.2,
    bboxes_weight: float = 0.8,
):
    scores = []
    for i in range(len(labels_2)):
        score = labels_bboxes_similarity(
            labels_1, bboxes_1, labels_2[i], bboxes_2[i], labels_weight, bboxes_weight
        )
        scores.append(score)
    return torch.tensor(scores).max().item()


def labels_similarity(labels_1, labels_2):
    def _intersection(labels_1, labels_2):
        cnt = 0
        x = Counter(labels_1)
        y = Counter(labels_2)
        for k in x:
            if k in y:
                cnt += 2 * min(x[k], y[k])
        return cnt

    def _union(labels_1, labels_2):
        return len(labels_1) + len(labels_2)

    if isinstance(labels_1, torch.Tensor):
        labels_1 = labels_1.tolist()
    if isinstance(labels_2, torch.Tensor):
        labels_2 = labels_2.tolist()
    return _intersection(labels_1, labels_2) / _union(labels_1, labels_2)


def bboxes_similarity(labels_1, bboxes_1, labels_2, bboxes_2, times=2):
    """
    bboxes_1: M x 4
    bboxes_2: N x 4
    distance: M x N
    """
    distance = torch.cdist(bboxes_1, bboxes_2) * times
    distance = torch.pow(0.5, distance)
    mask = labels_1.unsqueeze(-1) == labels_2.unsqueeze(0)
    distance = distance * mask
    row_ind, col_ind = linear_sum_assignment(-distance)
    return distance[row_ind, col_ind].sum().item() / len(row_ind)


def labels_bboxes_similarity(
    labels_1, bboxes_1, labels_2, bboxes_2, labels_weight, bboxes_weight
):
    labels_sim = labels_similarity(labels_1, labels_2)
    bboxes_sim = bboxes_similarity(labels_1, bboxes_1, labels_2, bboxes_2)
    return labels_weight * labels_sim + bboxes_weight * bboxes_sim
