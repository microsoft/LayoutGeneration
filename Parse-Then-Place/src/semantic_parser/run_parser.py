# coding=utf8

import logging
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import transformers
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_pt_utils import LabelSmoother

from ir.executor_rico import Executor as Rico_Executor
from ir.executor_web import Executor as Web_Executor
from layout_placement.placement_utils.utils import LABEL
from semantic_parser.dataset import (CollateFn, IRProcessor, SPDataset,
                                     TextProcessor)
from semantic_parser.generator import Generator
from semantic_parser.metrics import (AccMetric, ElementAccMetric, LossMetric,
                                     SetAccMetric)
from semantic_parser.models import (AdapterArguments, AdapterPretrainedLM,
                                    PretrainedLM, PromptTuningModel,
                                    create_adapter_optimizer)
from semantic_parser.trainer import DSTrainer
from utils import logging_utils

logger = logging.getLogger(__name__)


@dataclass
class DatasetArguments:
    """
    Arguments for dataset
    """

    data_dir: str = field(metadata={"help": "data root dir"})
    ir_remove_value: bool = field(default=False, metadata={"help": "Whether to run remove value attributes in ir."})
    replace_explicit_value: bool = field(
        default=False,
        metadata={"help": "Whether to replace explicitly value with placeholder."}
    )
    eval_split: Optional[str] = field(
        default='test',
        metadata={
            "help": "split to evaluate"
        }
    )
    dataset_name: str = field(default='web', metadata={"help": "web or rico"})


@dataclass
class ModelArguments(AdapterArguments):
    """
    Arguments for Model
    """
    model_name: Optional[str] = field(
        default='google/t5-v1_1-small',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    generation_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `max_length` value of the model configuration."
        },
    )
    tuning_method: str = field(
        default="finetune",
        metadata={
            "help": "Tune model using finetuning or prompt tuning",
            "choices": ['finetune', 'prompt_tuning', 'adapter']
        }
    )
    num_prompt_tokens: int = field(
        default=100,
        metadata={"help": "number of prompt tokens"}
    )
    prompt_init_method: str = field(
        default='vocab',
        metadata={"help": "ways to initialize embedding of prompt tokens"}
    )

    ADAPTER = 'adapter'
    PROMPT_TUNING = 'prompt_tuning'


@dataclass
class TrainArguments:
    """
    Arguments for training
    """
    output_dir: str = field(
        metadata={"help": "The output directory where the model checkpoints will be written."},
    )
    deepscale_config: str = field(
        metadata={
            "help": "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already loaded json file as a dict"
        },
    )
    prediction_dir: str = field(
        default="",
        metadata={"help": "The output directory where the predictions will be written."},
    )

    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

    num_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    eval_micro_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    eval_delay: Optional[float] = field(
        default=0,
        metadata={
            "help": "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the evaluation_strategy."
        },
    )
    save_interval: Optional[int] = field(
        default=1,
        metadata={
            "help": "interval to save checkpoint."
        }
    )
    train_log_step: Optional[int] = field(
        default=50,
        metadata={
            "help": "logging steps"
        }
    )

    backend: Optional[str] = field(
        default="nccl",
        metadata={"help": 'distributed backend'}
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    training_from_scratch: bool = field(default=False, metadata={"help": "Whether to train from scratch."})

    log_level: Optional[str] = field(
        default="info",
        metadata={
            "help": "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and lets the application set the level. Defaults to 'passive'.",
            "choices": logging_utils.log_levels.keys(),
        },
    )

    ds_ckpt_tag: Optional[str] = field(
        default=None,
        metadata={"help": "deepspeed checkpoint tag (load for evaluation)"}
    )

    use_adafactor: bool = field(
        default=False, metadata={"help": "Whether to use adafactor"}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={
            "help": "learning rate for adafactor"
        }
    )
    weight_decay: Optional[float] = field(
        default=1e-5,
        metadata={
            "help": "weight decay"
        }
    )
    adam_beta1: Optional[float] = field(
        default=0.9,
        metadata={
            "help": "adam_beta1"
        }
    )
    adam_beta2: Optional[float] = field(
        default=0.999,
        metadata={
            "help": "adam_beta2"
        }
    )
    adam_epsilon: Optional[float] = field(
        default=1e-08,
        metadata={
            "help": "adam_epsilon"
        }
    )
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )


def get_args():
    parser = HfArgumentParser((DatasetArguments, ModelArguments, TrainArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    return data_args, model_args, training_args


def config_logger(training_args):
    log_level = logging_utils.config_logger(training_args.log_level)

    # transformer logging
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


class TrainFn:

    def __init__(self, label_smoothing_factor: float = 0.0) -> None:
        self.label_smoother = None
        if label_smoothing_factor > 0:
            logger.info(f"Use Label Smoothing: {label_smoothing_factor}")
            self.label_smoother = LabelSmoother(epsilon=label_smoothing_factor)

    def __call__(self, model, batch, device) -> torch.Tensor:
        input_ids, attention_mask = batch['text_ids'].to(device), batch['text_attention_mask'].to(device)
        labels = batch['lf_ids'].to(device)
        model_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        if self.label_smoother is not None:
            loss = self.label_smoother(model_outputs, labels)
        else:
            loss = model_outputs.loss
        return loss.mean()


class EvaluateFn:

    def __init__(self, tokenizer: AutoTokenizer, ir_processor: IRProcessor, dataset_name,
                 do_predict: bool = False, generation_max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.ir_processor = ir_processor
        self.generation_max_length = generation_max_length
        self.do_predict = do_predict
        self.dataset_name = dataset_name
        if self.dataset_name == 'web':
            self.executor = Web_Executor(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../ir/grammar_web.lark'))
        elif self.dataset_name == 'rico':
            self.executor = Rico_Executor(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../ir/grammar_rico.lark'))

    def __call__(self, model, batch, device) -> Tuple[Dict, Dict]:
        input_ids, attention_mask = batch['text_ids'].to(device), batch['text_attention_mask'].to(device)
        labels = batch['lf_ids'].to(device)

        model_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        eval_loss = model_outputs.loss
        metrics = {
            "loss": eval_loss.mean()
        }

        predictions = None
        if self.do_predict:
            predictions, num_correct, num_element_correct, num_set_correct = self._predict(model, batch, device)
            metrics.update({
                "num_correct": num_correct,
                "num_element_correct": num_element_correct,
                "num_set_correct": num_set_correct,
                "num_examples": len(predictions),
            })
        return metrics, predictions

    def _is_set_accuracy(self, gold_lf, pred_lf):
        _gold_seq = self.executor(gold_lf)[0].input
        try:
            _pred_seq = self.executor(pred_lf)[0].input
        except:
            return False
        label_set = LABEL[self.dataset_name]
        _pattern = f"((?:{'|'.join(label_set)}) [^ ]+ [^ ]+)"
        _gold_constraint = Counter(re.findall(_pattern, _gold_seq))
        _pred_constraint = Counter(re.findall(_pattern, _pred_seq))

        return _gold_constraint.items() == _pred_constraint.items()

    def _predict(self, model, batch, device):
        input_ids, attention_mask = batch['text_ids'].to(device), batch['text_attention_mask'].to(device)
        output_sequences = model(input_ids, attention_mask, generation_max_length=self.generation_max_length,
                                 do_generation=True, pad_token_id=self.tokenizer.pad_token_id)
        out_str = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)

        predictions = list()
        num_correct = 0
        num_element_correct = 0
        num_set_correct = 0
        for idx, ostr in enumerate(out_str):
            gold_lf = batch['logical_form'][idx]
            is_correct = (self.ir_processor.postprocess(ostr) == self.ir_processor.postprocess(gold_lf))
            is_element_correct = self.ir_processor.postprocess(ostr, remove_attrs=True) == \
                self.ir_processor.postprocess(gold_lf, remove_attrs=True)
            is_set_correct = self._is_set_accuracy(
                self.ir_processor.postprocess(gold_lf, recover_labels=True),
                self.ir_processor.postprocess(ostr, recover_labels=True)
            )

            predictions.append({
                "pred_lf": self.ir_processor.postprocess(ostr, recover_labels=True),
                "gold_lf": self.ir_processor.postprocess(gold_lf, recover_labels=True),
                "is_set_correct": is_set_correct
            })
            meta_info = {key: batch[key][idx] for key in ['text', 'ex_id', 'type']}
            predictions[-1].update(meta_info)

            if is_correct: num_correct += 1
            if is_element_correct: num_element_correct += 1
            if is_set_correct: num_set_correct += 1
        return predictions, num_correct, num_element_correct, num_set_correct


def main():
    data_args, model_args, training_args = get_args()
    config_logger(training_args)
    set_seed(training_args.seed)

    logger.info("Load model")
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_fast=False)
    if model_args.tuning_method == ModelArguments.PROMPT_TUNING:
        logger.info("Use prompt tuning")
        model = PromptTuningModel(model_args.model_name, num_prompt_tokens=model_args.num_prompt_tokens,
                                  initialization_option=model_args.prompt_init_method)
    elif model_args.tuning_method == ModelArguments.ADAPTER:
        logger.info("Use adapter tuning")
        model = AdapterPretrainedLM(model_args.model_name, adapter_args=model_args)
    else:
        model = PretrainedLM(model_args.model_name, training_args.training_from_scratch)

    logger.info("Load dataset")
    logger.info(f"Replace explicit values: {data_args.replace_explicit_value}")
    text_processor = TextProcessor(replace_value=data_args.replace_explicit_value)
    ir_processor = IRProcessor(remove_value=data_args.ir_remove_value,
                               replace_value=data_args.replace_explicit_value)
    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        train_dataset = SPDataset(data_args.data_dir, 'train', tokenizer, ir_processor, text_processor)
        eval_dataset = SPDataset(data_args.data_dir, 'val', tokenizer, ir_processor, text_processor)
    else:
        eval_dataset = SPDataset(data_args.data_dir, data_args.eval_split, tokenizer, ir_processor, text_processor)
    collate_fn = CollateFn(pad_id=tokenizer.pad_token_id)

    eval_fn = EvaluateFn(tokenizer, ir_processor, data_args.dataset_name, do_predict=True,
                         generation_max_length=model_args.generation_max_length)

    if training_args.do_train:
        logger.info("Training...")
        experiment_config = {
            'model_name': model_args.model_name,
            'generation_max_length': model_args.generation_max_length
        }
        if model_args.tuning_method == ModelArguments.ADAPTER:
            optimizer = create_adapter_optimizer(model, training_args)
        else:
            optimizer = None
        trainer = DSTrainer(
            project_name='nl2web-sp', args=training_args, model=model,
            train_dataset=train_dataset, val_dataset=eval_dataset,
            collate_fn=collate_fn, experiment_config=experiment_config,
            metrics=[AccMetric(), LossMetric(), ElementAccMetric(), SetAccMetric()], ckpt_metric=AccMetric.METRIC_NAME,
            optimizer=optimizer
        )
        train_fn = TrainFn(training_args.label_smoothing_factor)
        trainer(train_fn, eval_fn)
    elif training_args.do_eval:
        logger.info("Inference...")
        generator = Generator(args=training_args, model=model, dataset=eval_dataset,
                              collate_fn=collate_fn, metrics=[AccMetric(), LossMetric(), ElementAccMetric(), SetAccMetric()],
                              save_prefix=data_args.eval_split)
        generator(eval_fn)


if __name__ == "__main__":
    main()
