# coding=utf8

import logging
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)


class PromptTuningModel(nn.Module):

    def __init__(self, model_name: str, num_prompt_tokens, initialization_option = "normal"):
        super().__init__()

        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._freeze_base_model()

        self.num_prompt_tokens = num_prompt_tokens
        self.new_tokens =  ["<V{}>".format(i) for i in range(num_prompt_tokens)]
        self.prompt_embeddings = nn.Embedding(num_prompt_tokens, self.base_model.config.d_model)
        self.indices = [i for i in range(num_prompt_tokens)]

        # initialize
        self._initialize_prompts(initialization_option)

    def _freeze_base_model(self):
        for p in self.base_model.parameters():
            p.requires_grad = False

    def _initialize_prompts(self, option = "normal"):
        logger.info("Initialize Prompt Embeddings from {}".format(option))
        if option == 'normal':
            factor = self.base_model.config.initializer_factor
            self.prompt_embeddings.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif option == "uniform":
            _range = 0.5
            self.prompt_embeddings.weight.data.uniform_(-1 * _range, _range)
        else:
            # intialize from vocabs (sample tokens from vocabulary)
            selected_indices = np.random.choice(range(6, self.base_model.config.vocab_size), self.num_prompt_tokens, replace=False)
            selected_indices = torch.from_numpy(selected_indices).long()
            weights = self.base_model.shared(selected_indices).clone().detach()
            self.prompt_embeddings.weight.data.copy_(weights)

    def load_prompt_embeddings(self, weight):
        self.prompt_embeddings.weight.data.copy_(torch.from_numpy(weight).float())

    def _prepend_prompt(self, input_ids, attention_mask = None):
        # Prepend Soft Prompt Tokens
        batch_size = input_ids.size(0)
        prompt_token_indices = input_ids.new_tensor(self.indices).unsqueeze(dim=0).repeat(batch_size, 1)
        prompt_embeds = self.prompt_embeddings(prompt_token_indices)
        input_token_embeds = self.base_model.shared(input_ids)
        input_embeds = torch.cat([prompt_embeds, input_token_embeds], dim=1)

        if attention_mask is not None:
            # Attention Mask
            prompt_attention_mask = attention_mask.new_ones(batch_size, self.num_prompt_tokens)
            input_attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)
        else:
            input_attention_mask = None

        return input_embeds, input_attention_mask

    def forward(self, input_ids, attention_mask, labels=None, pad_token_id=None,
                do_generation=False, generation_max_length: int = 512, prefix_allowed_tokens_fn: Callable = None):
        input_embeds, input_attention_mask = self._prepend_prompt(input_ids, attention_mask)
        if do_generation:
            decoder_start_token_id = self.base_model.config.decoder_start_token_id
            decoder_input_ids = input_ids.new_ones(input_ids.size(0), 1) * decoder_start_token_id
            outputs = self.base_model.generate(inputs_embeds=input_embeds, attention_mask=input_attention_mask,
                                          decoder_input_ids=decoder_input_ids, max_length=generation_max_length,
                                          pad_token_id=pad_token_id, prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
        else:
            outputs = self.base_model(inputs_embeds=input_embeds,
                                 attention_mask=input_attention_mask,
                                 labels=labels)
        return outputs
