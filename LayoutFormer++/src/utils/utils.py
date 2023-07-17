import re
import json
import random
from pathlib import Path
from typing import Dict, Tuple, List, Union

import torch
import numpy as np

from data.transforms import convert_ltwh_to_ltrb, convert_ltwh_to_xywh, convert_xywh_to_ltrb, decapulate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print("Random Seed:", seed)


def load_arguments(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def save_arguments(args: Dict, path: str) -> None:
    with open(path, 'w') as f:
        f.write(json.dumps(args, indent=2))


def init_experiment(args, out: str):
    if args.seed is None:
        args.seed = random.randint(0, 10000)

    set_seed(args.seed)

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / 'args.json'
    save_arguments(vars(args), json_path)

    return out_dir


def log_hyperparameters(args, world_size: int):
    config = {
        'seed': args.seed,
        'epoch': args.epoch,
        'gradient_accumulation': args.gradient_accumulation,
        'batch_size': args.batch_size * world_size * args.gradient_accumulation,
        'learning_rate': args.lr,
        'max_num_elements': args.max_num_elements,
        'dataset': args.dataset,
        'gradient_clip': args.clip_gradient
    }
    return config


def collate_fn(data_list):
    batch = {}
    keys = [set(data.keys()) for data in data_list]
    keys = list(set.union(*keys))

    for key in keys:
        batch[key] = [data[key] for data in data_list]

    return batch


def to_dense_batch(batch):
    '''
    padding a batch of data with value 0
    '''
    unsqueeze_flag = False
    if batch[-1].dim() == 1:
        unsqueeze_flag = True
        batch = [data.unsqueeze(-1) for data in batch]
    lens = [len(data) for data in batch]
    max_lens = max(lens)

    fill_size = batch[-1][-1].size()
    fill_unit = torch.zeros(fill_size, dtype=batch[-1].dtype)

    out = torch.cat((batch[0], fill_unit.repeat((max_lens - lens[0]), 1)),
                    dim=0).unsqueeze(0)
    for i in range(1, len(lens)):
        out = torch.cat((out, (torch.cat(
            (batch[i], fill_unit.repeat(
                (max_lens - lens[i]), 1)), dim=0).unsqueeze(0))),
                        dim=0)
    if unsqueeze_flag:
        out = out.squeeze(-1)

    mask = [[True] * i + [False] * (max_lens - i) for i in lens]
    mask = torch.from_numpy(np.array(mask))

    return out, mask


def parse_predicted_layout(
        output_str: str) -> Tuple[Union[List, None], Union[List, None]]:
    labels, bbox = list(), list()
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
            labels.append(" ".join(curr_labels))
            bbox.append(curr_bbox)
        else:
            return None, None
    if len(labels) == 0:
        return None, None
    return labels, bbox
