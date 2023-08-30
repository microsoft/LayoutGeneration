# coding=utf8

import os.path as osp
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from utils import file_utils

from .ir_processor import IRProcessor
from .text_processor import TextProcessor


class SPDataset(Dataset):

    def __init__(self, root: str, split: str, tokenizer: PreTrainedTokenizer,
                 ir_processor: IRProcessor, text_processor: TextProcessor) -> None:
        super().__init__()
        self.root_dir = root
        if osp.exists(osp.join(self.root_dir, f"{split}.jsonl")):
            self.split_path = osp.join(self.root_dir, f"{split}.jsonl")
        elif osp.exists(osp.join(self.root_dir, f"{split}.json")):
            self.split_path = osp.join(self.root_dir, f"{split}.json")
        else:
            self._create_dataset()
            self.split_path = osp.join(self.root_dir, f"{split}.jsonl")

        self.ir_processor = ir_processor
        self.text_processor = text_processor
        self.tokenizer = tokenizer
        if self.split_path.endswith('jsonl'):
            examples = file_utils.read_jsonl(self.split_path)
        else:
            examples = file_utils.read_json(self.split_path)
        self.data = list()
        for ex in examples:
            _ex = self._preprocess_ex(ex)
            if _ex is None:
                continue
            self.data.append(_ex)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

    def _create_dataset(self):
        # train-val-test: 80-10-10
        base_name = 'all.jsonl'
        split_file_names = ['train.jsonl', 'val.jsonl', 'test.jsonl']
        examples_by_type = defaultdict(list)
        examples = file_utils.read_jsonl(osp.join(self.root_dir, base_name))
        for ex in examples:
            type = ex['region_type']
            examples_by_type[type].append(ex)

        train, val, test = list(), list(), list()
        for _, rexamples in examples_by_type.items():

            rexamples_by_region_id = defaultdict(list)
            for ex in rexamples:
                rid = ex['region_id'].split(".")[0]
                rexamples_by_region_id[rid].append(ex)
            region_ids = list(rexamples_by_region_id.keys())

            generator = torch.Generator().manual_seed(0)
            indices = torch.randperm(len(region_ids), generator=generator)
            shuffled_region_ids = [region_ids[i] for i in indices]

            N = len(shuffled_region_ids)
            ratio = [int(N * .8), int(N * .9)]
            for rid in shuffled_region_ids[:ratio[0]]:
                train.extend(rexamples_by_region_id[rid])
            for rid in shuffled_region_ids[ratio[0]:ratio[1]]:
                val.extend(rexamples_by_region_id[rid])
            for rid in shuffled_region_ids[ratio[1]:]:
                test.extend(rexamples_by_region_id[rid])

        for split_examples, name in zip([train, val, test], split_file_names):
            file_utils.write_jsonl(osp.join(self.root_dir, name), split_examples)

    def _preprocess_ex(self, ex: Dict) -> Dict:
        if 'ir' not in ex:
            return None
        ex_id, region_type = ex['region_id'], ex['region_type']
        text, value_map = self.text_processor.preprocess(ex['text'])
        _lf = None
        if isinstance(ex['ir'], str):
            _lf = ex['ir']
        else:
            # get the shortest ir if there are multiple candidates
            _lf = ex['ir'][0]
            min_length = len(ex['ir'][0])
            for lf in ex['ir']:
                if len(lf) < min_length:
                    _lf = lf
                    min_length = len(lf)
        lf = self.ir_processor.preprocess(_lf, value_map)
        result = {
            'ex_id': ex_id, 'type': region_type,
            'text': text, 'logical_form': lf,
            'value_map': value_map
        }

        # Tokenization
        text_tokenization = self.tokenizer(text, return_tensors='pt')
        text_ids, text_attention_mask = text_tokenization.input_ids[0], text_tokenization.attention_mask[0]

        lf_tokenization = self.tokenizer(lf, return_tensors='pt')
        lf_ids, lf_attention_mask = lf_tokenization.input_ids[0], lf_tokenization.attention_mask[0]
        lf_ids[lf_ids == self.tokenizer.pad_token_id] = -100
        result.update({
            "text_ids": text_ids,
            "text_attention_mask": text_attention_mask,
            "lf_ids": lf_ids,
            "lf_attention_mask": lf_attention_mask
        })
        return result


class CollateFn:

    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, examples: List[Dict]) -> Dict:
        batch = dict()
        max_input_length, max_label_length = 0, 0
        for ex in examples:
            for key, value in ex.items():
                if key not in batch:
                    batch[key] = list()
                batch[key].append(value)
                if key == 'text_ids':
                    max_input_length = max(max_input_length, len(value))
                elif key == 'lf_ids':
                    max_label_length = max(max_label_length, len(value))

        # pad text ids & input ids
        for idx, (input_ids, attention_mask,) in enumerate(zip(batch['text_ids'], batch['text_attention_mask'])):
            diff = max_input_length - len(input_ids)
            if diff > 0:
                batch['text_ids'][idx] = F.pad(input_ids, (0, diff,), 'constant', self.pad_id)
                batch['text_attention_mask'][idx] = F.pad(attention_mask, (0, diff,), 'constant', 0)
        batch['text_ids'] = torch.stack(batch['text_ids'], dim=0)
        batch['text_attention_mask'] = torch.stack(batch['text_attention_mask'], dim=0)

        if 'lf_ids' in batch:
            for idx, (input_ids, attention_mask,) in enumerate(zip(batch['lf_ids'], batch['lf_attention_mask'])):
                diff = max_label_length - len(input_ids)
                if diff > 0:
                    batch['lf_ids'][idx] = F.pad(input_ids, (0, diff,), 'constant', self.pad_id)
                    batch['lf_attention_mask'][idx] = F.pad(attention_mask, (0, diff,), 'constant', 0)
            batch['lf_ids'] = torch.stack(batch['lf_ids'], dim=0)
            batch['lf_attention_mask'] = torch.stack(batch['lf_attention_mask'], dim=0)

        return batch
