import json
from pathlib import Path
from typing import List, Set, Union, Dict

import torch
from pycocotools.coco import COCO
"""
load raw data
"""


def load_publaynet_data(raw_dir: str, max_num_elements: int,
                        label_set: Union[List, Set], label2index: Dict):

    def is_valid(element):
        label = coco.cats[element['category_id']]['name']
        if label not in set(label_set):
            return False
        x1, y1, width, height = element['bbox']
        x2, y2 = x1 + width, y1 + height

        if x1 < 0 or y1 < 0 or W < x2 or H < y2:
            return False
        if x2 <= x1 or y2 <= y1:
            return False

        return True

    train_list, val_list = None, None
    raw_dir = Path(raw_dir) / 'publaynet'
    for split_publaynet in ['train', 'val']:
        dataset = []
        coco = COCO(raw_dir / f'{split_publaynet}.json')
        for img_id in sorted(coco.getImgIds()):
            ann_img = coco.loadImgs(img_id)
            W = float(ann_img[0]['width'])
            H = float(ann_img[0]['height'])
            name = ann_img[0]['file_name']
            if H < W:
                continue

            elements = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
            _elements = list(filter(is_valid, elements))
            filtered = len(elements) != len(_elements)
            elements = _elements

            N = len(elements)
            if N == 0 or max_num_elements < N:
                continue

            bboxes = []
            labels = []

            for element in elements:
                # bbox
                x1, y1, width, height = element['bbox']
                b = [x1 / W, y1 / H,  width / W, height / H]  # bbox format: ltwh
                bboxes.append(b)

                # label
                label = coco.cats[element['category_id']]['name']
                labels.append(label2index[label])

            bboxes = torch.tensor(bboxes, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.long)

            data = {
                'name': name,
                'bboxes': bboxes,
                'labels': labels,
                'canvas_size': [W, H],
                'filtered': filtered,
            }
            dataset.append(data)

        if split_publaynet == 'train':
            train_list = dataset
        else:
            val_list = dataset

    # shuffle train with seed
    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(train_list), generator=generator)
    train_list = [train_list[i] for i in indices]

    # train_list -> train 95% / val 5%
    # val_list -> test 100%

    s = int(len(train_list) * .95)
    train_set = train_list[:s]
    test_set = train_list[s:]
    val_set = val_list
    split_dataset = [train_set, test_set, val_set]
    return split_dataset


def load_rico_data(raw_dir: str, max_num_elements: int,
                   label_set: Union[List, Set], label2index: Dict):

    def is_valid(element):
        if element['componentLabel'] not in set(label_set):
            return False
        x1, y1, x2, y2 = element['bounds']
        if x1 < 0 or y1 < 0 or W < x2 or H < y2:
            return False
        if x2 <= x1 or y2 <= y1:
            return False
        return True

    def append_child(element, elements):
        if 'children' in element.keys():
            for child in element['children']:
                elements.append(child)
                elements = append_child(child, elements)
        return elements

    dataset = []
    raw_dir = Path(raw_dir) / 'semantic_annotations'
    for json_path in sorted(raw_dir.glob('*.json')):
        with json_path.open() as f:
            ann = json.load(f)

        B = ann['bounds']
        W, H = float(B[2]), float(B[3])
        if B[0] != 0 or B[1] != 0 or H < W:
            continue

        elements = append_child(ann, [])
        _elements = list(filter(is_valid, elements))
        filtered = len(elements) != len(_elements)
        elements = _elements

        N = len(elements)
        if N == 0 or N > max_num_elements:
            continue

        bboxes = []
        labels = []

        for element in elements:
            x1, y1, x2, y2 = element['bounds']
            b = [x1 / W, y1 / H, (x2-x1) / W, (y2-y1) / H]  # bbox format: ltwh
            bboxes.append(b)

            label = label2index[element['componentLabel']]
            labels.append(label)

        bboxes = torch.tensor(bboxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)

        data = {
            'name': json_path.name,
            'bboxes': bboxes,
            'labels': labels,
            'canvas_size': [W, H],
            'filtered': filtered,
        }
        dataset.append(data)

    # shuffle with seed
    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(dataset), generator=generator)
    dataset = [dataset[i] for i in indices]

    # train 85% / val 5% / test 10%
    N = len(dataset)
    s = [int(N * .85), int(N * .90)]
    train_set = dataset[:s[0]]
    test_set = dataset[s[0]:s[1]]
    val_set = dataset[s[1]:]
    split_dataset = [train_set, test_set, val_set]

    return split_dataset
