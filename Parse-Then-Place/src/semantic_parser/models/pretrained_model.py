# coding=utf8

from typing import Callable

import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM


class PretrainedLM(nn.Module):

    def __init__(self, model_name: str, training_from_scratch: bool = False):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if training_from_scratch:
            config = self.model.config
            config.num_layers = 2
            config.num_decoder_layers = 2
            self.model = AutoModelForSeq2SeqLM.from_config(config)

    def forward(self, input_ids, attention_mask, labels=None, pad_token_id=None,
                do_generation=False, generation_max_length: int = 512, prefix_allowed_tokens_fn: Callable = None):
        if do_generation:
            outputs = self.model.generate(input_ids, attention_mask=attention_mask,
                                          max_length=generation_max_length, pad_token_id=pad_token_id,
                                          prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
        else:
            outputs = self.model(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 labels=labels)
        return outputs
