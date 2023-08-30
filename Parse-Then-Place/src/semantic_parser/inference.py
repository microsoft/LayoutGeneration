# coding=utf8

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import deepspeed
import torch
from transformers import AutoTokenizer

from semantic_parser.dataset import IRProcessor, TextProcessor
from semantic_parser.models import (AdapterArguments, AdapterPretrainedLM,
                                    PretrainedLM)

logger = logging.getLogger(__name__)


@dataclass
class PredictorArguments:
    parser_device: int = field(
        metadata={"help": "loading device for parser"}
    )
    deepscale_config: str = field(
        metadata={"help": "deepspeed config"}
    )
    parser_model_name: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    parser_ckpt_path: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    parser_ds_ckpt_tag: Optional[str] = field(
        default=None,
        metadata={"help": "deepspeed checkpoint tag (load for evaluation)"}
    )
    parser_generation_max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `max_length` value of the model configuration."
        },
    )
    parser_tuning_method: Optional[str] = field(
        default="finetune",
        metadata={
            "help": "Tune model using finetuning or prompt tuning",
            "choices": ['finetune', 'adapter']
        }
    )

    parser_replace_explicit_value: Optional[int] = field(
        default=False,
        metadata={
            "help": "replace explicit value during parsing"
        }
    )
    parser_remove_pagination: Optional[bool] = field(
        default=True,
        metadata={
            "help": "remove pagination"
        }
    )

    backend: str = field(
        default="nccl",
        metadata={"help": 'distributed backend'}
    )

    adapter_config: Optional[AdapterArguments] = field(
        default=None,
        metadata={"help": "adapter configuration"}
    )


class Predictor:

    MODEL_BIN_NAME = "pytorch_model.bin"

    def __init__(self, predictor_args) -> None:
        self.args = predictor_args
        self._setup_model()
        self.text_processor = TextProcessor(replace_value=self.args.parser_replace_explicit_value)
        self.ir_processor = IRProcessor(remove_value=False,
                                        replace_value=self.args.parser_replace_explicit_value)

    def _setup_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.parser_model_name, use_fast=False)
        if self.args.parser_tuning_method == 'adapter':
            logger.info("Using adapter model")
            logger.info(self.args.adapter_config)
            model = AdapterPretrainedLM(self.args.parser_model_name,
                                        adapter_args=self.args.adapter_config)
        else:
            model = PretrainedLM(self.args.parser_model_name)
        self.device = torch.device("cuda:{}".format(self.args.parser_device))

        if os.path.exists(os.path.join(self.args.parser_ckpt_path, self.MODEL_BIN_NAME)):
            state = torch.load(os.path.join(self.args.parser_ckpt_path, self.MODEL_BIN_NAME))
            model.load_state_dict(state, strict=False)
            self.model = model.to(self.device)
        else:
            # deepspeed
            deepspeed.init_distributed(dist_backend=self.args.backend)
            params = filter(lambda p: p.requires_grad, model.parameters())
            self.model, _, _, _ = deepspeed.initialize(args=self.args, model=model,
                                                    model_parameters=params)
            self.model.load_checkpoint(self.args.parser_ckpt_path, tag=self.args.parser_ds_ckpt_tag,
                                       load_module_only=True,
                                       load_optimizer_states=False,
                                       load_lr_scheduler_states=False) # model engine
            self.model.to(self.device)
        self.model.eval()

    def __call__(self, text: str) -> List[str]:
        processed_text, value_map = self.text_processor.preprocess(text)
        tokenization = self.tokenizer(processed_text, return_tensors='pt')
        text_ids = tokenization.input_ids.to(self.device)
        text_attention_mask = tokenization.attention_mask.to(self.device)

        results = list()
        with torch.no_grad():
            output_sequences = self.model(text_ids, text_attention_mask,
                                          generation_max_length=self.args.parser_generation_max_length,
                                          do_generation=True, pad_token_id=self.tokenizer.pad_token_id)
            out_str = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            for ostr in out_str:
                results.append(self.ir_processor.postprocess(ostr, recover_labels=True,
                                                             recover_values=self.args.parser_replace_explicit_value,
                                                             value_map=value_map))
        return results
