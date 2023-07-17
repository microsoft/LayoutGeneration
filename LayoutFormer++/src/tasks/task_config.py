from typing import Callable

from model import LayoutTransformerTokenizer
from data import RicoDataset, PubLayNetDataset
from tasks.refinement import T5LayoutSequence, T5RefinementDataset, refinement_inference
from tasks.completion import T5CompletionLayoutSequence, T5CompletionDataset, completion_inference
from tasks.ugen import T5UGenDataset, ugen_inference
from tasks.gen_t import T5LayoutSequenceForGenT, T5GenTDataset, gen_t_inference
from tasks.gen_r import T5LayoutSequenceForGenR, T5GenRDataset, gen_r_inference
import tasks.supplement as supplement


def create_seq_processor(task: str) -> Callable:

    def _create(dataset: str, tokenizer: LayoutTransformerTokenizer,
                add_sep_token: bool = False, *args, **kwargs):
        _dataset = RicoDataset if dataset == 'rico' else PubLayNetDataset
        label2index, index2label = dict(), dict()
        for idx, _ in _dataset.index2label(_dataset.labels).items():
            ln = "label_{}".format(idx)
            index2label[idx] = ln
            label2index[ln] = idx

        seq_processor_class = None
        if task == 'refinement':
            seq_processor_class = T5LayoutSequence
        elif task in {'completion', 'ugen'}:
            seq_processor_class = T5CompletionLayoutSequence
        elif task in {'gen_t', 'gen_ts'}:
            seq_processor_class = T5LayoutSequenceForGenT
            if task == 'gen_t':
                return T5LayoutSequenceForGenT(T5LayoutSequenceForGenT.GEN_T, tokenizer,
                                               index2label, label2index, add_sep_token=add_sep_token, *args, **kwargs)
            else:
                return T5LayoutSequenceForGenT(T5LayoutSequenceForGenT.GEN_TS, tokenizer,
                                               index2label, label2index, add_sep_token=add_sep_token, *args, **kwargs)
        elif task == 'gen_r':
            return T5LayoutSequenceForGenR(tokenizer, index2label, label2index,
                                           add_sep_token=add_sep_token, *args, **kwargs)
        elif task == 'gen_tc':
            seq_processor_class = supplement.T5LayoutSequenceForGenTC
        elif task == 'gen_tsc':
            seq_processor_class = supplement.T5LayoutSequenceForGenTSC
        elif task == 'gen_rs':
            seq_processor_class = supplement.T5LayoutSequenceForGenRS
        elif task == 'gen_rp':
            seq_processor_class = supplement.T5LayoutSequenceForGenRP

        return seq_processor_class(tokenizer, index2label, label2index,
                                   add_sep_token=add_sep_token)

    return _create


TASK_CONFIG = {
    'refinement': {
        'task_id': 0,
        'dataset': T5RefinementDataset,
        'seq_processor': create_seq_processor('refinement'),
        'prompt': 'task-refinement :',
        'inference': refinement_inference,
        'saved_layouts': ['input', 'gold', 'pred'],
        'decode_mode': 'greedy',
        'use_constrained_decoding': True
    },
    'completion': {
        'task_id': 1,
        'dataset': T5CompletionDataset,
        'seq_processor': create_seq_processor('completion'),
        'prompt': 'task-completion :',
        'inference': completion_inference,
        'saved_layouts': ['pred'],
        'decode_mode': 'topk',
        'use_constrained_decoding': False
    },
    'ugen': {
        'task_id': 2,
        'dataset': T5UGenDataset,
        'seq_processor': create_seq_processor('ugen'),
        'prompt': 'task-ugen :',
        'inference': ugen_inference,
        'saved_layouts': ['pred'],
        'decode_mode': 'topk',
        'use_constrained_decoding': False
    },
    'gen_t': {
        'task_id': 3,
        'dataset': T5GenTDataset,
        'seq_processor': create_seq_processor('gen_t'),
        'seq_processor_params': [
            ('gen_t_add_unk_token', False)
        ],
        'prompt': 'task-gen_t :',
        'inference': gen_t_inference,
        'saved_layouts': ['input', 'gold', 'pred'],
        'decode_mode': 'greedy',
        'use_constrained_decoding': True
    },
    'gen_ts': {
        'task_id': 4,
        'dataset': T5GenTDataset,
        'seq_processor': create_seq_processor('gen_ts'),
        'seq_processor_params': [
            ('gen_ts_add_unk_token', False)
        ],
        'prompt': 'task-gen_ts :',
        'inference': gen_t_inference,
        'saved_layouts': ['input', 'gold', 'pred'],
        'decode_mode': 'greedy',
        'use_constrained_decoding': True,
    },
    'gen_r': {
        'task_id': 5,
        'dataset': T5GenRDataset,
        'seq_processor': create_seq_processor('gen_r'),
        'seq_processor_params': [
            ('gen_r_add_unk_token', False),
            ('gen_r_compact', False),
        ],
        'prompt': 'task-gen_r :',
        'inference': gen_r_inference,
        'saved_layouts': ['gold', 'pred', 'violate_num'],
        'decode_mode': 'topk',
        'use_constrained_decoding': True
    },
    'gen_tc': {
        'task_id': 6,
        'dataset': supplement.T5GenTCDataset,
        'seq_processor': create_seq_processor('gen_tc'),
        'prompt': 'task-gen_tc :',
        'inference': supplement.gen_tc_inference,
        'saved_layouts': ['gold', 'pred'],
        'decode_mode': 'topk',
        'use_constrained_decoding': False
    },
    'gen_tsc': {
        'task_id': 7,
        'dataset': supplement.T5GenTSCDataset,
        'seq_processor': create_seq_processor('gen_tsc'),
        'prompt': 'task-gen_tsc :',
        'inference': supplement.gen_tsc_inference,
        'saved_layouts': ['gold', 'pred'],
        'decode_mode': 'topk',
        'use_constrained_decoding': False
    },
    'gen_rs': {
        'task_id': 8,
        'dataset': supplement.T5GenRSDataset,
        'seq_processor': create_seq_processor('gen_rs'),
        'prompt': 'task-gen_rs :',
        'inference': supplement.gen_rs_inference,
        'saved_layouts': ['gold', 'pred'],
        'decode_mode': 'topk',
        'use_constrained_decoding': False
    },
    'gen_rp': {
        'task_id': 9,
        'dataset': supplement.T5GenRPDataset,
        'seq_processor': create_seq_processor('gen_rp'),
        'prompt': 'task-gen_rp :',
        'inference': supplement.gen_rp_inference,
        'saved_layouts': ['gold', 'pred'],
        'decode_mode': 'topk',
        'use_constrained_decoding': False
    }
}
