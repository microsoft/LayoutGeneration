from typing import List, Dict, Union, Tuple
import inspect
import os

import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from data import RicoDataset, PubLayNetDataset
from model import LayoutTransformerTokenizer
from tasks.refinement import T5LayoutSequence
from tasks.gen_r import RelationTypes, T5LayoutSequenceForGenR
import tasks.multitask as multitask
from tasks.task_config import TASK_CONFIG
from model import LayoutTransformer
from utils import utils
from evaluation import metrics


def create_tokenizer(tasks: List[str], dataset: str, discrete_grid: int,
                     add_sep_token: bool = False, sep_token: str = T5LayoutSequence.SEP_TOKEN,
                     add_task_prompt: bool = False) -> LayoutTransformerTokenizer:
    tokens = list()
    if dataset == 'rico':
        _dataset = RicoDataset
    elif dataset == 'publaynet':
        _dataset = PubLayNetDataset
    else:
        raise NotImplementedError(f"No dataset: {dataset}")
    label2index_fn = _dataset.label2index(_dataset.labels)
    tokens.extend(map(lambda x: "label_{}".format(
        label2index_fn[x]), _dataset.labels))
    tokens.extend(map(str, range(discrete_grid)))
    if add_sep_token:
        tokens.append(sep_token)

    if 'gen_r' in tasks:
        # add canvas label
        tokens.append("label_0")
        # add relations
        type2index_fn = RelationTypes.type2index()
        tokens.extend(map(lambda x: "relation_{}".format(
            type2index_fn[x]), RelationTypes.types))
        # add index for element
        index_range = range(1, 21)
        tokens.extend(map(lambda x: "index_{}".format(x), index_range))
        # add twp-level sep token for relation
        tokens.append(T5LayoutSequenceForGenR.REL_BEG_TOKEN)
        tokens.append(T5LayoutSequenceForGenR.REL_SEP_TOKEN)
        tokens.append(T5LayoutSequenceForGenR.REL_ELE_SEP_TOKEN)

    if add_task_prompt:
        prompt_tokens = list()
        for tn in ['refinement', 'completion', 'ugen', 'gen_t', 'gen_ts']:
            task_prompt = TASK_CONFIG[tn]['prompt'].lower()
            for token in task_prompt.split():
                if token not in prompt_tokens:
                    prompt_tokens.append(token)
        if 'gen_r' in tasks:
            tn = 'gen_r'
            task_prompt = TASK_CONFIG[tn]['prompt'].lower()
            for token in task_prompt.split():
                if token not in prompt_tokens:
                    prompt_tokens.append(token)
        tokens.extend(prompt_tokens)

    return LayoutTransformerTokenizer(tokens)


def build_model(args, tokenizer: LayoutTransformerTokenizer) -> LayoutTransformer:
    if args.add_task_prompt and args.add_task_prompt_token_in_model:
        raise TypeError(
            "add_task_prompt and add_task_prompt_token_in_model is mutually exclusive")
    return LayoutTransformer(
        vocab_size=len(tokenizer), max_len=args.num_pos_embed, bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        d_model=args.d_model, num_layers=args.num_layers, nhead=args.nhead,
        dropout=args.dropout, d_feedforward=args.d_model * 4, share_embedding=args.share_embedding,
        add_task_embedding=args.add_task_embedding, num_task_embedding=len(
            TASK_CONFIG),
        add_task_prompt_token=args.add_task_prompt_token_in_model,
        num_task_prompt_token=args.num_task_prompt_token
    )


def create_dataset(args, tokenizer: PreTrainedTokenizer, task_config: Dict,
                   split: str, sort_by_pos: bool = True) -> Union[RicoDataset, PubLayNetDataset]:
    remove_too_long_layout = split == 'train' and args.remove_too_long_layout
    if args.dataset == 'rico':
        if split == 'train':
            if args.partition_training_data:
                return multitask.T5RicoMultiTaskPartitionDataset(args, tokenizer, task_config, split, online_process=True,
                                                                 remove_too_long_layout=remove_too_long_layout, sort_by_pos=sort_by_pos,
                                                                 task_buckets=args.partition_training_data_task_buckets)
            elif args.fine_grained_partition_training_data:
                return multitask.T5RicoMultiTaskFineGrainedPartitionDataset(args, tokenizer, task_config, split, online_process=True,
                                                                            remove_too_long_layout=remove_too_long_layout, sort_by_pos=sort_by_pos,
                                                                            task_data_size=args.fine_grained_partition_training_data_task_size,
                                                                            task_weights=args.task_weights)
            else:
                if args.single_task_per_batch:
                    return multitask.T5RicoMultiTaskConcatDataset(
                        args, tokenizer, task_config, split, online_process=True,
                        remove_too_long_layout=remove_too_long_layout,
                        sort_by_pos=sort_by_pos
                    )
                else:
                    return multitask.T5RicoMultiTaskSamplingDataset(args, tokenizer, task_config, split, online_process=True,
                                                                    remove_too_long_layout=remove_too_long_layout,
                                                                    sort_by_pos=sort_by_pos)
        else:
            return multitask.T5RicoMultiTaskRotationDataset(args, tokenizer, task_config, split, online_process=True,
                                                            remove_too_long_layout=remove_too_long_layout,
                                                            sort_by_pos=sort_by_pos)
    elif args.dataset == 'publaynet':
        if split == 'train':
            if args.partition_training_data:
                return multitask.T5PubLayNetMultiTaskPartitionDataset(args, tokenizer, task_config, split, online_process=True,
                                                                      remove_too_long_layout=remove_too_long_layout, sort_by_pos=sort_by_pos,
                                                                      task_buckets=args.partition_training_data_task_buckets)
            elif args.fine_grained_partition_training_data:
                return multitask.T5PubLayNetMultiTaskFineGrainedPartitionDataset(args, tokenizer, task_config, split, online_process=True,
                                                                                 remove_too_long_layout=remove_too_long_layout, sort_by_pos=sort_by_pos,
                                                                                 task_data_size=args.fine_grained_partition_training_data_task_size,
                                                                                 task_weights=args.task_weights)
            else:
                if args.single_task_per_batch:
                    return multitask.T5PubLayNetMultiTaskConcatDataset(
                        args, tokenizer, task_config, split, online_process=True,
                        remove_too_long_layout=remove_too_long_layout,
                        sort_by_pos=sort_by_pos
                    )
                else:
                    return multitask.T5PubLayNetMultiTaskSamplingDataset(args, tokenizer, task_config, split, online_process=True,
                                                                         remove_too_long_layout=remove_too_long_layout,
                                                                         sort_by_pos=sort_by_pos)
        else:
            return multitask.T5PubLayNetMultiTaskRotationDataset(args, tokenizer, task_config, split, online_process=True,
                                                                 remove_too_long_layout=remove_too_long_layout,
                                                                 sort_by_pos=sort_by_pos)
    raise NotImplementedError("No Valid Dataset")


class TrainFn:

    def __init__(self, task_loss_weights: Dict = None):
        self.task_loss_weights = task_loss_weights

    def __call__(self, model, data, tokenizer, device) -> torch.Tensor:
        in_tokenization = tokenizer(data['in_str'], add_eos=True, add_bos=False)
        in_ids = in_tokenization['input_ids'].to(device)
        mask = in_tokenization['mask'].to(device)
        padding_mask = ~mask

        out_tokenization = tokenizer(data['out_str'], add_eos=True, add_bos=False)
        out_ids = out_tokenization['input_ids'].to(device)

        task_ids = None
        if 'task_id' in data:
            task_ids = torch.tensor(data['task_id']).long().to(device)

        loss_weights = None
        if self.task_loss_weights is not None:
            loss_weights = torch.tensor([self.task_loss_weights[tn] for tn in data['task_name']]).to(device)
        model_outputs = model(in_ids, padding_mask, out_ids, loss_weights=loss_weights, task_ids=task_ids)
        loss = model_outputs['loss']
        return loss.mean()


class EvaluateFn:

    def __init__(self, max_num_elements: int, enable_task_measure: bool = False,
                 decode_max_length: int = 350, topk: int = 10, temperature: float = 0.7):
        self.enable_task_measure = enable_task_measure
        self.max_num_elements = max_num_elements
        self.decode_max_length = decode_max_length
        self.topk = topk
        self.temperature = temperature

    def _parse_seq(self, seq_processor, out_str) -> Tuple[List, List]:
        num_arguments = len(list(inspect.signature(seq_processor.parse_seq).parameters.keys()))
        if num_arguments == 1:
            return seq_processor.parse_seq(out_str)
        else:
            return seq_processor.parse_seq(None, out_str)

    def _measure_prediction(self, model, in_tokenization, tokenizer, seq_processor,
                            device, data) -> Dict[str, Dict]:
        task = data['task_name'][0]
        in_ids = in_tokenization['input_ids']
        in_padding_mask = in_tokenization['padding_mask']

        task_ids = None
        if 'task_id' in data:
            task_ids = torch.tensor(data['task_id']).long().to(device)

        if TASK_CONFIG[task]['decode_mode'] == 'greedy':
            # greedy
            output_sequences = model(in_ids, in_padding_mask,
                                     max_length=self.decode_max_length,
                                     task_ids=task_ids)['output']
        else:
            # topk sampling
            output_sequences = model(in_ids, in_padding_mask, do_sample=True,
                                     max_length=self.decode_max_length, top_k=self.topk,
                                     temperature=self.temperature, task_ids=task_ids)['output']
        out_str = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        gold_labels, gold_mask = utils.to_dense_batch(data['gold_labels'])
        gold_bbox, _ = utils.to_dense_batch(data['gold_bboxes'])
        gold_labels = gold_labels.to(device)
        gold_mask = gold_mask.to(device)
        gold_bbox = gold_bbox.to(device)

        task_pred = {
            'pred_bboxes': list(), 'pred_labels': list(), 'pred_mask': list(),
            'gold_bboxes': list(), 'gold_labels': list(), 'gold_mask': list(),
        }
        for idx, ostr in enumerate(out_str):
            task = data['task_name'][idx]
            out_str[idx] = ostr.strip()
            _pred_labels, _pred_bbox = self._parse_seq(seq_processor, out_str[idx])

            if _pred_labels is None:
                # something wrong in the model's predictions
                _pred_bbox = in_ids.new_zeros(self.max_num_elements, 4).long()
                _pred_labels = in_ids.new_zeros(self.max_num_elements).long()
                _pred_mask = torch.zeros(self.max_num_elements, dtype=bool).to(device)
            else:
                diff = self.max_num_elements - len(_pred_labels)
                _pred_labels = torch.tensor(_pred_labels).long().to(device)
                _pred_bbox = torch.tensor(_pred_bbox).long().to(device)
                _pred_mask = torch.ones(self.max_num_elements, dtype=bool).to(device)
                if diff > 0:
                    _pred_mask[len(_pred_labels):] = False
                    _pred_labels = F.pad(_pred_labels, (0, diff,),
                                         'constant', seq_processor.error_label_id)
                    _pred_bbox = F.pad(_pred_bbox, (0, 0, 0, diff,), 'constant', 0)
                elif diff < 0:
                    _pred_labels = _pred_labels[:self.max_num_elements]
                    _pred_bbox = _pred_bbox[:self.max_num_elements, :]

            task_pred['pred_bboxes'].append(_pred_bbox)
            task_pred['pred_labels'].append(_pred_labels)
            task_pred['pred_mask'].append(_pred_mask)
            task_pred['gold_labels'].append(gold_labels[idx])
            task_pred['gold_bboxes'].append(gold_bbox[idx])
            task_pred['gold_mask'].append(gold_mask[idx])

        # stack
        for key in list(task_pred.keys()):
            task_pred[key] = torch.stack(task_pred[key], dim=0)

        return task_pred

    def __call__(self, model, data, seq_processor, tokenizer,
                 device) -> Tuple[Dict, torch.Tensor]:
        in_tokenization = tokenizer(
            data['in_str'], add_eos=True, add_bos=False)
        in_tokenization['input_ids'] = in_tokenization['input_ids'].to(device)
        in_tokenization['mask'] = in_tokenization['mask'].to(device)
        in_tokenization['padding_mask'] = ~in_tokenization['mask']

        out_tokenization = tokenizer(
            data['out_str'], add_eos=True, add_bos=False)
        out_ids = out_tokenization['input_ids'].to(device)

        task_ids = None
        if 'task_id' in data:
            task_ids = torch.tensor(data['task_id']).long().to(device)

        model_outputs = model(in_tokenization['input_ids'], in_tokenization['padding_mask'],
                              out_ids, task_ids=task_ids)
        eval_loss = model_outputs['loss']

        if self.enable_task_measure:
            prediction = self._measure_prediction(model, in_tokenization, tokenizer,
                                                  seq_processor, device, data)
        else:
            prediction = None

        return eval_loss.mean(), prediction


def create_fid_model(args, device='cpu'):
    if args.dataset == 'rico':
        fid_net_path = os.path.join('net', 'fid_rico.pth.tar')
        fid_model = metrics.LayoutFID(max_num_elements=args.max_num_elements, num_labels=len(RicoDataset.labels),
                                      net_path=fid_net_path, device=device)
        return fid_model
    elif args.dataset == 'publaynet':
        fid_net_path = os.path.join('net', 'fid_publaynet.pth.tar')
        fid_model = metrics.LayoutFID(max_num_elements=args.max_num_elements, num_labels=len(PubLayNetDataset.labels),
                                      net_path=fid_net_path, device=device)
        return fid_model
    raise NotImplementedError("No Valid Dataset")
