import torch
import copy
import math
import random
import numpy as np
from typing import Tuple, Union, List
# processing function


class ShuffleElements():

    def __call__(self, data):
        if 'gold_bboxes' not in data.keys():
            data['gold_bboxes'] = copy.deepcopy(data['bboxes'])

        ele_num = len(data['labels'])
        shuffle_idx = np.arange(ele_num)
        random.shuffle(shuffle_idx)
        data['bboxes'] = data['bboxes'][shuffle_idx]
        data['gold_bboxes'] = data['gold_bboxes'][shuffle_idx]
        data['labels'] = data['labels'][shuffle_idx]
        return data


class LabelDictSort():
    '''
    sort elements in one layout by their label
    '''
    def __init__(self, index2label=None):
        self.index2label = index2label

    def __call__(self, data):
        # NOTE: for refinement
        if 'gold_bboxes' not in data.keys():
            data['gold_bboxes'] = copy.deepcopy(data['bboxes'])

        labels = data['labels'].tolist()
        idx2label = [[idx, self.index2label[labels[idx]]] for idx in range(len(labels))]
        idx2label_sorted = sorted(idx2label, key=lambda x: x[1])
        idx_sorted = [d[0] for d in idx2label_sorted]
        data['bboxes'], data['labels'] = data['bboxes'][idx_sorted], data['labels'][idx_sorted]
        data['gold_bboxes'] = data['gold_bboxes'][idx_sorted]
        return data


class LexicographicSort():
    '''
    sort elements in one layout by their top and left postion
    '''

    def __call__(self, data):
        if 'gold_bboxes' not in data.keys():
            data['gold_bboxes'] = copy.deepcopy(data['bboxes'])
        l, t, _, _ = data['bboxes'].t()
        _zip = zip(*sorted(enumerate(zip(t, l)), key=lambda c: c[1:]))
        idx = list(list(_zip)[0])
        data['ori_bboxes'], data['ori_labels'] = data['gold_bboxes'], data[
            'labels']
        data['bboxes'], data['labels'] = data['bboxes'][idx], data['labels'][
            idx]
        data['gold_bboxes'] = data['gold_bboxes'][idx]
        return data


class AddGaussianNoise():
    '''
    Add Gaussian Noise to bounding box
    '''

    def __init__(self,
                 mean=0.,
                 std=1.,
                 normalized: bool = True,
                 bernoulli_beta: float = 1.0):
        self.std = std
        self.mean = mean
        self.normalized = normalized
        # adding noise to every element by default
        self.bernoulli_beta = bernoulli_beta
        print('Noise: mean={0}, std={1}, beta={2}'.format(
            self.mean, self.std, self.bernoulli_beta))

    def __call__(self, data):
        # Gold Label
        if 'gold_bboxes' not in data.keys():
            data['gold_bboxes'] = copy.deepcopy(data['bboxes'])

        num_elemnts = data['bboxes'].size(0)
        beta = data['bboxes'].new_ones(num_elemnts) * self.bernoulli_beta
        element_with_noise = torch.bernoulli(beta).unsqueeze(dim=-1)

        if self.normalized:
            data['bboxes'] = data['bboxes'] + torch.randn(
                data['bboxes'].size()) * self.std + self.mean
        else:
            canvas_width, canvas_height = data['canvas_size'][0], data[
                'canvas_size'][1]
            ele_x, ele_y = data['bboxes'][:, 0] * canvas_width, data[
                'bboxes'][:, 1] * canvas_height
            ele_w, ele_h = data['bboxes'][:, 2] * canvas_width, data[
                'bboxes'][:, 3] * canvas_height
            data['bboxes'] = torch.stack([ele_x, ele_y, ele_w, ele_h], dim=1)
            data['bboxes'] = data['bboxes'] + torch.randn(
                data['bboxes'].size()) * self.std + self.mean
            data['bboxes'][:, 0] /= canvas_width
            data['bboxes'][:, 1] /= canvas_height
            data['bboxes'][:, 2] /= canvas_width
            data['bboxes'][:, 3] /= canvas_height
        data['bboxes'][data['bboxes'] < 0] = 0.0
        data['bboxes'][data['bboxes'] > 1] = 1.0
        data['bboxes'] = data['bboxes'] * element_with_noise + data[
            'gold_bboxes'] * (1 - element_with_noise)
        return data

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, beta={2})'.format(
            self.mean, self.std, self.bernoulli_beta)


class CoordinateTransform():

    def __init__(self, bbox_format):
        self.bbox_format = bbox_format

    def _transform(self, x):
        convert_func = {
            "ltrb": convert_ltwh_to_ltrb,
            "xywh": convert_ltwh_to_xywh
        }
        return convert_func[self.bbox_format](x)

    def __call__(self, data):
        if 'gold_bboxes' not in data.keys():
            data['gold_bboxes'] = copy.deepcopy(data['bboxes'])
        data['bboxes_ltwh'] = data['bboxes']
        data['gold_bboxes_ltwh'] = data['gold_bboxes']
        if self.bbox_format != 'ltwh':
            data['bboxes'] = self._transform(data['bboxes'])
            data['gold_bboxes'] = self._transform(data['gold_bboxes'])
        return data


class DiscretizeBoundingBox():

    def __init__(self, num_x_grid: int, num_y_grid: int) -> None:
        self.num_x_grid = num_x_grid
        self.num_y_grid = num_y_grid
        self.max_x = self.num_x_grid - 1
        self.max_y = self.num_y_grid - 1

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
            [discrete_x1, discrete_y1, discrete_x2, discrete_y2],
            dim=-1).long()

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
        if 'gold_bboxes' not in data.keys():
            data['gold_bboxes'] = copy.deepcopy(data['bboxes'])
        discrete_bboxes = self.discretize(data['bboxes'])
        data['discrete_bboxes'] = discrete_bboxes
        discrete_gold_bboxes = self.discretize(data['gold_bboxes'])
        data['discrete_gold_bboxes'] = discrete_gold_bboxes
        return data


def convert_ltwh_to_xywh(bbox):
    l, t, w, h = decapulate(bbox)
    xc = l + w / 2
    yc = t + h / 2
    return torch.stack([xc, yc, w, h], axis=-1)


def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = decapulate(bbox)
    left = xc - w / 2
    top = yc - h / 2
    right = xc + w / 2
    bottom = yc + h / 2
    return torch.stack([left, top, right, bottom], axis=-1)


def convert_ltwh_to_ltrb(bbox):
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


class LayoutSequence:

    index2label = None
    label2index = None

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def build_seq(self, labels, bboxes) -> str:
        pass

    @classmethod
    def parse_seq(self, seq: str) -> Tuple[Union[List, None], Union[List, None]]:
        pass
