# from PIL import Image
# import blobfile as bf
from typing import List, Set, Union, Dict
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, default_data_collator, PreTrainedTokenizerFast, \
    PreTrainedTokenizer
import sys, os
import torch
from collections import Counter, defaultdict
from functools import partial
from itertools import chain
import os.path as osp
# import seaborn as sns
from einops import rearrange, reduce
import json


def load_data_text(
    *, data_dir, batch_size, seq_length, class_cond=False, deterministic=False, data_args=None, 
        task_mode='roc', model=None, padding_mode='pad', split='train', load_vocab=None,ungen=False,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param seq_length: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    print('loading text data.')
    if split=='test':
        data_args.checkpoint_path=os.path.join(os.path.split(data_args.model_path)[0])
    if data_args.experiment.startswith('random') and model is None:
        model = None


    if task_mode == 'e2e-tgt':
        training_data, model = get_corpus_rocstory(data_args, model, seq_length,
                                            padding_mode=padding_mode, split=split,
                                            load_vocab=load_vocab)

    dataset = TextDataset(
        training_data,
        seq_length,
        data_args,
        model_arch=data_args.model_arch,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )


    if deterministic:

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            drop_last=True,
            shuffle=False,
            num_workers=1,
        )

    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            drop_last=True,
            shuffle=True,
            num_workers=1,
        )
    
    while True:
        yield from data_loader

def helper_tokenize_encode_cond(sentence_lst, vocab_dict, model, seqlen, data_args):
    result_train_lst = []
    group_lst = defaultdict(list)
    with torch.no_grad():
        for (src_ids, input_ids) in sentence_lst:
            tokenized_ = [vocab_dict.get(x, vocab_dict['UNK']) for x in input_ids]
            tokenized_src = [vocab_dict.get(x, vocab_dict['UNK']) for x in src_ids]
            input_ids = [0] + tokenized_ + [1]
            group_lst['word_ids'].append(input_ids)
            group_lst['src_ids'].append(tokenized_src)

        max_length = seqlen
        group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)
        max_src_length = max([len(xx) for xx in group_lst['src_ids']])
        max_src_length = min(seqlen, max_src_length)
        group_lst['src_ids'], group_lst['src_mask'] = _collate_batch_helper(group_lst['src_ids'],
                                                                            vocab_dict['PAD'],
                                                                            max_src_length,
                                                                            return_mask=True)


        for input_ids, src_ids, src_mask in zip(group_lst['word_ids'], group_lst['src_ids'],
                                      group_lst['src_mask']):
            if data_args.experiment.startswith('random'):
                hidden_state = model(torch.tensor(input_ids))
            elif data_args.experiment == 'gpt2_pre_compress':
                input_ids2 = torch.tensor(input_ids).to(model.device)
                input_embs = model.transformer.wte(input_ids2)  # input_embs
                hidden_state = model.down_proj(input_embs)
                hidden_state = hidden_state * data_args.emb_scale_factor
            result_train_lst.append({'input_ids': input_ids,
                                     'hidden_states': hidden_state.cpu().tolist(),
                                     'src_ids':src_ids,
                                     'src_mask':src_mask
                                     })

    return result_train_lst


def helper_tokenize_encode(sentence_lst, vocab_dict, model, seqlen, data_args, padding_mode, ):
    result_train_lst = []
    group_lst = defaultdict(list)
    with torch.no_grad():
        if data_args.e2e_train.split('/')[-1]=='RICO_nosep':
            for input_ids in sentence_lst:
                tokenized_ = [vocab_dict.get(x) for x in input_ids]
                group_lst['word_ids'].append(tokenized_)
        else:
            for input_ids in sentence_lst:
                tokenized_ = [vocab_dict.get(x, vocab_dict['UNK']) for x in input_ids]
                input_ids = [0] + tokenized_ + [1]
                group_lst['word_ids'].append(input_ids)

        if padding_mode == 'block':
            concatenated_examples = {k: sum(group_lst[k], []) for k in group_lst.keys()}
            total_length = len(concatenated_examples[list(group_lst.keys())[0]])
            block_size = seqlen
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            group_lst = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
        elif padding_mode == 'pad':
            max_length = seqlen
            group_lst['word_ids'] = _collate_batch_helper(group_lst['word_ids'], vocab_dict['PAD'], max_length)

        for input_ids in group_lst['word_ids']:
            if data_args.training_mode=='bit':
                def int2bit(x, n=7):
                    # Convert integers into the corresponding binary bits.
                    device=x.device
                    x = torch.bitwise_right_shift(x.unsqueeze(-1).int(), th.range(0,n,dtype=th.int64,device=device))
                    x = torch.fmod(x, 2).float()
                    return x*2-1
                hidden_state = int2bit(torch.tensor(input_ids))
            elif data_args.experiment.startswith('random'):
                hidden_state = model(torch.tensor(input_ids))
            result_train_lst.append({'input_ids': input_ids, 'hidden_states': hidden_state.cpu().tolist()})
    return result_train_lst


def get_corpus_rocstory(data_args, model, seq_length, padding_mode='block',
                        split='train', load_vocab=None):
    import csv, torch, json
    from spacy.lang.en import English

    if data_args.experiment_mode == 'lm':
    
        if data_args.modality == 'e2e-tgt':
            sentence_lst = []
            nlp = English()
            tokenizer = nlp.tokenizer
            if split == 'train':
                print('loading form the TRAIN set')
                path = f'{data_args.e2e_train}/src1_train.txt'
            elif split == 'valid':
                print('loading form the VALID set')
                path = f'{data_args.e2e_train}/src1_valid.txt'
            elif split == 'test':
                print('loading form the TEST set')
                if split=='test': ##inference process use offline data
                    data_args.e2e_train='../data/processed_datasets/'+data_args.e2e_train.split('/')[-1]
                    # data_args.e2e_train='../data/processed_datasets/RICO_nosep'
                path = f'{data_args.e2e_train}/src1_test.txt'
            elif split == 'debug':
                print('loading form the DEBUG set')
                path = data_args.debug_path
                # import json
                with open(path, 'r') as ff:
                    for line in ff:
                        sentence_lst.append(json.loads(line)[0].split(' '))
                sentence_lst = sentence_lst + sentence_lst
            if split in ['train', 'valid', 'test']:
                with open(path, 'r') as ff:
                    for row in ff:
                        # word_lst = row.split('||')[1] 
                        word_lst=row.rstrip('\n') #junyi RICO
                        word_lst = [x.text for x in tokenizer(word_lst)]
                        sentence_lst.append(word_lst)

        # get tokenizer.
        if load_vocab is None:
            counter = Counter()
            for input_ids in sentence_lst:
                counter.update(input_ids)

    if load_vocab is None:
        if data_args.e2e_train.split('/')[-1]=='RICO_nosep':
            vocab_dict = {'PAD':0}
        else:
            vocab_dict = {'START': 0, 'END': 1, 'UNK':2, 'PAD':3, '|':4 }
        for k, v in counter.items():
            if v > 10:
                if k not in [str(num) for num in range(128)] and k not in vocab_dict:
                    vocab_dict[k] = len(vocab_dict)
        for num in range(128):
            vocab_dict[str(num)] = len(vocab_dict)

        path_save_vocab = f'{data_args.checkpoint_path}/vocab.json'
        print(f'save the vocab to {path_save_vocab}')
        with open(path_save_vocab, 'w') as f:
            json.dump(vocab_dict, f)
    else:
        vocab_dict = load_vocab
        path_save_vocab = f'{data_args.checkpoint_path}/vocab.json'
        if not os.path.exists(path_save_vocab):
            print(f'save the vocab to {path_save_vocab}')
            if isinstance(vocab_dict, dict):
                with open(path_save_vocab, 'w') as f:
                    json.dump(vocab_dict, f)
                # assert vocab_dict['START'] == 0
            elif isinstance(vocab_dict, PreTrainedTokenizerFast):
                vocab_dict.save_pretrained(data_args.checkpoint_path)
            else:
                assert False, "invalid type of vocab_dict"



    if model is None and data_args.experiment == 'random':
        model = torch.nn.Embedding(len(vocab_dict), data_args.in_channel)
        print('initializing the random embeddings', model)
        torch.nn.init.normal_(model.weight)
        path_save = f'{data_args.checkpoint_path}/random_emb.torch'
        print(f'save the random encoder to {data_args.checkpoint_path}/random_emb.torch')
        torch.save(model.state_dict(), path_save)


    path_save = f'{data_args.checkpoint_path}/random_emb.torch'
    if not os.path.exists(path_save) and data_args.experiment == 'random':
        torch.save(model.state_dict(), path_save)


    if data_args.experiment_mode == 'lm':
        result_train_lst = helper_tokenize_encode(sentence_lst, vocab_dict, model, seq_length, data_args, padding_mode)
    elif data_args.experiment_mode == 'conditional_gen':
        result_train_lst = helper_tokenize_encode_cond(sentence_lst, vocab_dict, model, seq_length, data_args)
    # print(result_train_lst[0]['hidden_states'],'hidden state in corpus')
    return {'train': result_train_lst}, model



class TextDataset(Dataset):
    def __init__(self, text_datasets, resolution, data_args, model_arch='conv-unet',
                 classes=None, shard=0, num_shards=1, eigen_transform=None,
                 mapping_func=None, model_emb=None):
        super().__init__()
        self.resolution = resolution
        self.text_datasets=text_datasets
        self.text_datasets['train'] = self.text_datasets['train'][shard:][::num_shards]
        self.length = len(self.text_datasets['train'])
        self.model_arch = model_arch
        self.data_args = data_args
        self.eigen_transform = eigen_transform
        self.mapping_func = mapping_func
        self.model_emb = model_emb
        # self.local_images = image_paths[shard:][::num_shards]
        # self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        if self.model_arch != 'conv-unet':
            arr = np.array(self.text_datasets['train'][idx]['hidden_states'],
                           dtype=np.float32)
            if self.eigen_transform  is not None:
                old_shape = arr.shape
                # arr = arr.reshape(1, -1) @ self.eigen_transform
                arr = arr.reshape(1, -1) - self.eigen_transform['mean']
                arr = arr @ self.eigen_transform['map']
                arr = arr.reshape(old_shape)
                
            if hasattr(self.data_args, 'noise_level') and self.data_args.noise_level > 0:
                # print(arr.dtype)
                # print(self.data_args.noise_level, 'using the noise level.')
                arr = arr + self.data_args.noise_level * np.random.randn(*arr.shape).astype(arr.dtype)
                # print(arr.dtype)

            out_dict = {}
            out_dict['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            # out_dict['mapping_func'] = self.mapping_func
            if self.data_args.experiment_mode == 'conditional_gen':
                out_dict['src_ids'] = np.array(self.text_datasets['train'][idx]['src_ids'])
                out_dict['src_mask'] = np.array(self.text_datasets['train'][idx]['src_mask'])
            # if self.local_classes is not None:
            #     out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
            # print(arr[0],'arr0')
            return arr, out_dict


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result
