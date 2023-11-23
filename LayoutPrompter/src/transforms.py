import copy
import math
import random
from itertools import combinations, product

import clip
import cv2
import numpy as np
import torch

from utils import decapulate, detect_loc_relation, detect_size_relation


class ShuffleElements:
    def __call__(self, data):
        if "gold_bboxes" not in data.keys():
            data["gold_bboxes"] = copy.deepcopy(data["bboxes"])

        ele_num = len(data["labels"])
        shuffle_idx = np.arange(ele_num)
        random.shuffle(shuffle_idx)
        data["bboxes"] = data["bboxes"][shuffle_idx]
        data["gold_bboxes"] = data["gold_bboxes"][shuffle_idx]
        data["labels"] = data["labels"][shuffle_idx]
        return data


class LabelDictSort:
    """
    sort elements in one layout by their label
    """

    def __init__(self, index2label=None):
        self.index2label = index2label

    def __call__(self, data):
        # NOTE: for refinement
        if "gold_bboxes" not in data.keys():
            data["gold_bboxes"] = copy.deepcopy(data["bboxes"])

        labels = data["labels"].tolist()
        idx2label = [[idx, self.index2label[labels[idx]]] for idx in range(len(labels))]
        idx2label_sorted = sorted(idx2label, key=lambda x: x[1])
        idx_sorted = [d[0] for d in idx2label_sorted]
        data["bboxes"], data["labels"] = (
            data["bboxes"][idx_sorted],
            data["labels"][idx_sorted],
        )
        data["gold_bboxes"] = data["gold_bboxes"][idx_sorted]
        return data


class LexicographicSort:
    """
    sort elements in one layout by their top and left postion
    """

    def __call__(self, data):
        if "gold_bboxes" not in data.keys():
            data["gold_bboxes"] = copy.deepcopy(data["bboxes"])
        l, t, _, _ = data["bboxes"].t()
        _zip = zip(*sorted(enumerate(zip(t, l)), key=lambda c: c[1:]))
        idx = list(list(_zip)[0])
        data["ori_bboxes"], data["ori_labels"] = data["gold_bboxes"], data["labels"]
        data["bboxes"], data["labels"] = data["bboxes"][idx], data["labels"][idx]
        data["gold_bboxes"] = data["gold_bboxes"][idx]
        return data


class AddGaussianNoise:
    """
    Add Gaussian Noise to bounding box
    """

    def __init__(
        self, mean=0.0, std=1.0, normalized: bool = True, bernoulli_beta: float = 1.0
    ):
        self.std = std
        self.mean = mean
        self.normalized = normalized
        # adding noise to every element by default
        self.bernoulli_beta = bernoulli_beta
        print(
            "Noise: mean={0}, std={1}, beta={2}".format(
                self.mean, self.std, self.bernoulli_beta
            )
        )

    def __call__(self, data):
        # Gold Label
        if "gold_bboxes" not in data.keys():
            data["gold_bboxes"] = copy.deepcopy(data["bboxes"])

        num_elemnts = data["bboxes"].size(0)
        beta = data["bboxes"].new_ones(num_elemnts) * self.bernoulli_beta
        element_with_noise = torch.bernoulli(beta).unsqueeze(dim=-1)

        if self.normalized:
            data["bboxes"] = (
                data["bboxes"]
                + torch.randn(data["bboxes"].size()) * self.std
                + self.mean
            )
        else:
            canvas_width, canvas_height = data["canvas_size"][0], data["canvas_size"][1]
            ele_x, ele_y = (
                data["bboxes"][:, 0] * canvas_width,
                data["bboxes"][:, 1] * canvas_height,
            )
            ele_w, ele_h = (
                data["bboxes"][:, 2] * canvas_width,
                data["bboxes"][:, 3] * canvas_height,
            )
            data["bboxes"] = torch.stack([ele_x, ele_y, ele_w, ele_h], dim=1)
            data["bboxes"] = (
                data["bboxes"]
                + torch.randn(data["bboxes"].size()) * self.std
                + self.mean
            )
            data["bboxes"][:, 0] /= canvas_width
            data["bboxes"][:, 1] /= canvas_height
            data["bboxes"][:, 2] /= canvas_width
            data["bboxes"][:, 3] /= canvas_height
        data["bboxes"][data["bboxes"] < 0] = 0.0
        data["bboxes"][data["bboxes"] > 1] = 1.0
        data["bboxes"] = data["bboxes"] * element_with_noise + data["gold_bboxes"] * (
            1 - element_with_noise
        )
        return data

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1}, beta={2})".format(
            self.mean, self.std, self.bernoulli_beta
        )


class DiscretizeBoundingBox:
    def __init__(self, num_x_grid: int, num_y_grid: int) -> None:
        self.num_x_grid = num_x_grid
        self.num_y_grid = num_y_grid
        self.max_x = self.num_x_grid
        self.max_y = self.num_y_grid

    def discretize(self, bbox):
        """
        Args:
            continuous_bbox torch.Tensor: N * 4
        Returns:
            discrete_bbox torch.LongTensor: N * 4
        """
        cliped_boxes = torch.clip(bbox, min=0.0, max=1.0)
        x1, y1, x2, y2 = decapulate(cliped_boxes)
        discrete_x1 = torch.floor(x1 * self.max_x)
        discrete_y1 = torch.floor(y1 * self.max_y)
        discrete_x2 = torch.floor(x2 * self.max_x)
        discrete_y2 = torch.floor(y2 * self.max_y)
        return torch.stack(
            [discrete_x1, discrete_y1, discrete_x2, discrete_y2], dim=-1
        ).long()

    def continuize(self, bbox):
        """
        Args:
            discrete_bbox torch.LongTensor: N * 4

        Returns:
            continuous_bbox torch.Tensor: N * 4
        """
        x1, y1, x2, y2 = decapulate(bbox)
        cx1, cx2 = x1 / self.max_x, x2 / self.max_x
        cy1, cy2 = y1 / self.max_y, y2 / self.max_y
        return torch.stack([cx1, cy1, cx2, cy2], dim=-1).float()

    def continuize_num(self, num: int) -> float:
        return num / self.max_x

    def discretize_num(self, num: float) -> int:
        return int(math.floor(num * self.max_y))

    def __call__(self, data):
        if "gold_bboxes" not in data.keys():
            data["gold_bboxes"] = copy.deepcopy(data["bboxes"])
        data["discrete_bboxes"] = self.discretize(data["bboxes"])
        data["discrete_gold_bboxes"] = self.discretize(data["gold_bboxes"])
        if "content_bboxes" in data.keys():
            data["discrete_content_bboxes"] = self.discretize(data["content_bboxes"])
        return data


class AddCanvasElement:
    def __init__(self, use_discrete=False, discrete_fn=None):
        self.x = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float)
        self.y = torch.tensor([0], dtype=torch.long)
        self.use_discrete = use_discrete
        self.discrete_fn = discrete_fn

    def __call__(self, data):
        if self.use_discrete:
            data["bboxes_with_canvas"] = torch.cat(
                [self.x, self.discrete_fn.continuize(data["discrete_gold_bboxes"])],
                dim=0,
            )
        else:
            data["bboxes_with_canvas"] = torch.cat([self.x, data["bboxes"]], dim=0)
        data["labels_with_canvas"] = torch.cat([self.y, data["labels"]], dim=0)
        return data


class AddRelation:
    def __init__(self, seed=1024, ratio=0.1):
        self.ratio = ratio
        self.generator = random.Random()
        if seed is not None:
            self.generator.seed(seed)
        self.type2index = RelationTypes.type2index()

    def __call__(self, data):
        data["labels_with_canvas_index"] = [0] + list(
            range(len(data["labels_with_canvas"]) - 1)
        )
        N = len(data["labels_with_canvas"])

        rel_all = list(product(range(2), combinations(range(N), 2)))
        # size = min(int(len(rel_all)                     * self.ratio), 10)
        size = int(len(rel_all) * self.ratio)
        rel_sample = set(self.generator.sample(rel_all, size))

        relations = []
        for i, j in combinations(range(N), 2):
            bi, bj = data["bboxes_with_canvas"][i], data["bboxes_with_canvas"][j]
            canvas = data["labels_with_canvas"][i] == 0

            if ((0, (i, j)) in rel_sample) and (not canvas):
                rel_size = detect_size_relation(bi, bj)
                relations.append(
                    [
                        data["labels_with_canvas"][i],
                        data["labels_with_canvas_index"][i],
                        data["labels_with_canvas"][j],
                        data["labels_with_canvas_index"][j],
                        self.type2index[rel_size],
                    ]
                )

            if (1, (i, j)) in rel_sample:
                rel_loc = detect_loc_relation(bi, bj, canvas)
                relations.append(
                    [
                        data["labels_with_canvas"][i],
                        data["labels_with_canvas_index"][i],
                        data["labels_with_canvas"][j],
                        data["labels_with_canvas_index"][j],
                        self.type2index[rel_loc],
                    ]
                )

        data["relations"] = torch.as_tensor(relations).long()

        return data


class RelationTypes:
    types = ["smaller", "equal", "larger", "top", "center", "bottom", "left", "right"]
    _type2index = None
    _index2type = None

    @classmethod
    def type2index(self):
        if self._type2index is None:
            self._type2index = dict()
            for idx, type in enumerate(self.types):
                self._type2index[type] = idx
        return self._type2index

    @classmethod
    def index2type(self):
        if self._index2type is None:
            self._index2type = dict()
            for idx, type in enumerate(self.types):
                self._index2type[idx] = type
        return self._index2type


class SaliencyMapToBBoxes:
    def __init__(
        self,
        threshold: int,
        is_filter_small_bboxes: bool = True,
        min_side: int = 80,
        min_area: int = 6000,
    ) -> None:
        self.threshold = threshold
        self.is_filter_small_bboxes = is_filter_small_bboxes
        self.min_side = min_side
        self.min_area = min_area

    def _is_small_bbox(self, bbox):
        return any(
            [
                all([bbox[2] <= self.min_side, bbox[3] <= self.min_side]),
                bbox[2] * bbox[3] < self.min_area,
            ]
        )

    def __call__(self, saliency_map):
        saliency_map_gray = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY)
        _, thresholded_map = cv2.threshold(
            saliency_map_gray, self.threshold, 255, cv2.THRESH_BINARY
        )
        contours, _ = cv2.findContours(
            thresholded_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if self.is_filter_small_bboxes and self._is_small_bbox([x, y, w, h]):
                continue
            bboxes.append([x, y, w, h])

        bboxes = sorted(bboxes, key=lambda x: (x[1], x[0]))
        bboxes = torch.tensor(bboxes)
        return bboxes


class CLIPTextEncoder:
    def __init__(self, model_name: str = "ViT-B/32"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.proprocess = clip.load(self.model_name, self.device)

    @torch.no_grad()
    def __call__(self, text: str):
        token = clip.tokenize(text, truncate=True).to(self.device)
        text_feature = self.model.encode_text(token)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        return text_feature
