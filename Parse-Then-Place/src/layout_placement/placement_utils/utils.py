import json
import logging
import random

import numpy as np
import torch

RICO_LABEL2ID = {
    'text button': 0,
    'background image': 1,
    'icon': 2,
    'list item': 3,
    'text': 4,
    'toolbar': 5,
    'web view': 6,
    'input': 7,
    'card': 8,
    'advertisement': 9,
    'image': 10,
    'drawer': 11,
    'radio button': 12,
    'checkbox': 13,
    'multi-tab': 14,
    'pager indicator': 15,
    'modal': 16,
    'on/off switch': 17,
    'slider': 18,
    'map view': 19,
    'button bar': 20,
    'video': 21,
    'bottom navigation': 22,
    'number stepper': 23,
    'date picker': 24,
}


WEB_LABEL2ID = {
    'text': 0,
    'link': 1,
    'button': 2,
    'title': 3,
    'description': 4,
    'image': 5,
    'background': 6,
    'logo': 7,
    'icon': 8,
    'input': 9,
    'select': 10,
    'textarea': 11
}


LABEL2ID = {
    'web': WEB_LABEL2ID,
    'rico': RICO_LABEL2ID
}


LABEL = {
    'web': list(WEB_LABEL2ID.keys()),
    'rico': list(RICO_LABEL2ID.keys())
}


CANVAS_SIZE = {
    'web': (120, 120),
    'rico': (144, 256)
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_txt(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        content = content.strip().split('\n')
    return content


def write_txt(data_list, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(item + '\n')


def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def write_json(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f, indent=2)


class Logger:

    def __init__(self) -> None:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


class AverageMeter:
    """Keep track of average values over time.
    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.
        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count