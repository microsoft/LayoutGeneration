# coding=utf8

import json
import torch
import torch.nn.functional as F
from typing import List, Union, Dict


class LayoutTransformerTokenizer:

    def __init__(self, tokens: List, bos_token: str = '<bos>',
                 eos_token: str = '<eos>', pad_token: str = '<pad>',
                 sep_token: str = '<sep>', unk_token: str = '<unk>'):
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.sep_token = sep_token
        self.unk_token = unk_token
        self.special_tokens = [self.bos_token, self.eos_token, self.pad_token,
                               self.sep_token, self.unk_token]
        self._build_vocabulary(tokens)

    def _build_vocabulary(self, tokens: List):
        self._token2id, self._id2token = dict(), dict()
        tid = 0
        for token in self.special_tokens:
            self._token2id[token] = tid
            self._id2token[tid] = token
            tid += 1
        for token in tokens:
            self._token2id[token] = tid
            self._id2token[tid] = token
            tid += 1

    @property
    def pad_token_id(self):
        return self._token2id[self.pad_token]

    @property
    def eos_token_id(self):
        return self._token2id[self.eos_token]

    @property
    def bos_token_id(self):
        return self._token2id[self.bos_token]

    @property
    def sep_token_id(self):
        return self._token2id[self.sep_token]

    @property
    def unk_token_id(self):
        return self._token2id[self.unk_token]

    def _tokenize(self, text: str) -> List[str]:
        return text.strip().split()

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self._token2id.get(t, self.unk_token_id) for t in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self._id2token[i] for i in ids]

    def __call__(self, text: Union[List[str], str], add_eos: bool = True,
                 add_bos: bool = False) -> Dict[str, torch.Tensor]:
        if isinstance(text, str):
            _text = [text]
        else:
            _text = text

        max_len = 0
        token_ids, mask = list(), list()
        for t in _text:
            tokens = self._tokenize(t)
            if add_eos:
                tokens.append(self.eos_token)
            if add_bos:
                tokens.insert(0, self.bos_token)
            num_tokens = len(tokens)
            max_len = max(max_len, num_tokens)
            token_ids.append(torch.tensor(self.convert_tokens_to_ids(tokens), dtype=torch.long))
            mask.append(torch.ones(num_tokens, dtype=bool))

        # Padding
        for i in range(len(token_ids)):
            ids = token_ids[i]
            diff = max_len - len(ids)
            if diff == 0:
                continue
            token_ids[i] = F.pad(token_ids[i], (0, diff,), 'constant', self.pad_token_id)
            mask[i] = F.pad(mask[i], (0, diff,), 'constant', False)
        token_ids = torch.stack(token_ids, dim=0)
        mask = torch.stack(mask, dim=0)
        return {
            'input_ids': token_ids, 'mask': mask
        }

    def __len__(self):
        return len(self._token2id)

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        tokens = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            _tokens = [t for t in tokens if t not in self.special_tokens]
        else:
            _tokens = tokens
        return " ".join(_tokens)

    def batch_decode(self, ids: Union[List[List], torch.Tensor],
                     skip_special_tokens: bool = False) -> List[str]:
        text = list()
        for _ids in ids:
            if isinstance(_ids, torch.Tensor):
                id_list = _ids.tolist()
            else:
                id_list = _ids
            text.append(self.decode(id_list, skip_special_tokens))
        return text

    def save_vocab(self, vocab_path: str) -> None:
        with open(vocab_path, 'w') as f:
            f.write(json.dumps(self._token2id))

    def from_vocab(self, vocab_path: str):
        with open(vocab_path, 'r') as f:
            token2id = json.load(f)
        self._token2id = token2id
        self._id2token = dict()
        for token, tid in self._token2id.items():
            self._id2token[tid] = token
