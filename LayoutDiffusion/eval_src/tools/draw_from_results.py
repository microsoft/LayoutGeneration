import os
import pickle
import argparse
import sys
import torch
import numpy as np
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from utils import visualization
from utils.os_utils import makedirs
from data import RicoDataset, PubLayNetDataset
from tqdm import tqdm

def draw_images_from_results(dataset: str, dataset_dir: str, max_num_elements: int, path: str, 
                             save_path: str, num_images: int = 10) -> None:
    dataset_class = RicoDataset if dataset == 'rico' else PubLayNetDataset
    dataset_obj = dataset_class(root=dataset_dir, split='train', max_num_elements=max_num_elements)
    draw_colors = dataset_obj.colors

    with open(path, 'rb') as f:
        results = pickle.load(f)

    makedirs(save_path)

    for idx, item in enumerate(tqdm(results[:num_images])):
        for key in ['pred', 'gold', 'input']:
            if key not in item:
                continue
            img_bboxes, img_labels = item[key]
            img_bboxes=torch.from_numpy(np.array(img_bboxes))
            img_labels=torch.from_numpy(np.array(img_labels))
            img_masks = img_labels > 0
            visualization.save_image(img_bboxes.unsqueeze(dim=0), img_labels.unsqueeze(dim=0), img_masks.unsqueeze(dim=0), 
                                    draw_colors, os.path.join(save_path, f'{idx:04d}_{key}.png'), canvas_size=(360, 240))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--dataset', '-d', type=str, choices=['rico', 'publaynet'], help='dataset name')
    arg_parser.add_argument('--data_dir', type=str, default='../processed_datasets/', help='data dir')
    arg_parser.add_argument('--max_num_elements', type=int, default=20, help='max number of design elements')
    arg_parser.add_argument('--path', '-p', type=str, required=True, help='generation results')
    arg_parser.add_argument('--save_path', '-s', type=str, required=True, help='path to save the results')
    arg_parser.add_argument('--num_images', '-n', type=int, default=10, required=False, help='number of results to draw')
    args = arg_parser.parse_args()
    draw_images_from_results(args.dataset, args.data_dir, args.max_num_elements, args.path, args.save_path, args.num_images)
