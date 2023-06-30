import os.path as osp
from typing import List, Set, Union

import torch
from torch.utils.data import Dataset
import seaborn as sns

from utils import os_utils
from .load import load_publaynet_data, load_rico_data


class LayoutDataset(Dataset):

    _label2index = None
    _index2label = None
    _colors = None

    split_file_names = ['train.pt', 'val.pt', 'test.pt']

    def __init__(self,
                 root: str,
                 data_name: str,
                 split: str,
                 max_num_elements: int,
                 label_set: Union[List, Set],
                 online_process: bool = True):

        self.root = f'{root}/{data_name}/'
        self.raw_dir = osp.join(self.root, 'raw')
        self.max_num_elements = max_num_elements
        self.label_set = label_set
        self.pre_processed_dir = osp.join(
            self.root, 'pre_processed_{}_{}'.format(self.max_num_elements, len(self.label_set)))
        assert split in ['train', 'val', 'test']

        # cite out for draw pics!!
        # 
        # if os_utils.files_exist(self.pre_processed_paths):
        #     idx = self.split_file_names.index('{}.pt'.format(split))
        #     print(f'Loading {split}...')
        #     self.data = torch.load(self.pre_processed_paths[idx])
        # else:
        #     print(f'Pre-processing and loading {split}...')
        #     os_utils.makedirs(self.pre_processed_dir)
        #     split_dataset = self.load_raw_data()
        #     self.save_split_dataset(split_dataset)
        #     idx = self.split_file_names.index('{}.pt'.format(split))
        #     self.data = torch.load(self.pre_processed_paths[idx])
            
        # self.online_process = online_process
        # if not self.online_process:
        #     self.data = [self.process(item) for item in self.data]
        # 
        # cite out for draw pics!!

    @property
    def pre_processed_paths(self):
        return [
            osp.join(self.pre_processed_dir, f) for f in self.split_file_names
        ]

    @classmethod
    def label2index(self, label_set):
        if self._label2index is None:
            self._label2index = dict()
            for idx, label in enumerate(label_set):
                self._label2index[label] = idx + 1
        return self._label2index

    @classmethod
    def index2label(self, label_set):
        if self._index2label is None:
            self._index2label = dict()
            for idx, label in enumerate(label_set):
                self._index2label[idx + 1] = label
        return self._index2label

    @property
    def colors(self):
        if self._colors is None:
            n_colors = len(self.label_set) + 1
            colors = sns.color_palette('husl', n_colors=n_colors)
            self._colors = [
                tuple(map(lambda x: int(x * 255), c)) for c in colors
            ]
        return self._colors

    def save_split_dataset(self, split_dataset):
        torch.save(split_dataset[0], self.pre_processed_paths[0])
        torch.save(split_dataset[1], self.pre_processed_paths[1])
        torch.save(split_dataset[2], self.pre_processed_paths[2])

    def load_raw_data(self) -> list:
        raise NotImplementedError

    def process(self, data) -> dict:
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.online_process:
            sample = self.process(self.data[idx])
        else:
            sample = self.data[idx]
        return sample


class PubLayNetDataset(LayoutDataset):
    labels = [
        'text',
        'title',
        'list',
        'table',
        'figure',
    ]

    def __init__(self,
                 root: str,
                 split: str,
                 max_num_elements: int,
                 online_process: bool = True):
        data_name = 'publaynet'
        super().__init__(root,
                         data_name,
                         split,
                         max_num_elements,
                         label_set=self.labels,
                         online_process=online_process)

    def load_raw_data(self) -> list:
        return load_publaynet_data(self.raw_dir, self.max_num_elements,
                                   self.label_set, self.label2index(self.label_set))


class RicoDataset(LayoutDataset):

    labels = [
        'Text', 'Image', 'Icon', 'List Item', 'Text Button', 'Toolbar',
        'Web View', 'Input', 'Card', 'Advertisement', 'Background Image',
        'Drawer', 'Radio Button', 'Checkbox', 'Multi-Tab', 'Pager Indicator',
        'Modal', 'On/Off Switch', 'Slider', 'Map View', 'Button Bar', 'Video',
        'Bottom Navigation', 'Number Stepper', 'Date Picker'
    ]

    def __init__(self,
                 root: str,
                 split: str,
                 max_num_elements: int,
                 online_process: bool = True):
        data_name = 'rico'
        super().__init__(root,
                         data_name,
                         split,
                         max_num_elements,
                         label_set=self.labels,
                         online_process=online_process)

    def load_raw_data(self) -> list:
        return load_rico_data(self.raw_dir, self.max_num_elements,
                              self.label_set, self.label2index(self.label_set))
