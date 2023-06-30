from random import random

import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder

from .nn import (SiLU, avg_pool_nd, checkpoint, conv_nd, linear,
                 timestep_embedding, zero_module)


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class TransformerModel(nn.Module):
    """
    Diffusion-lm
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        dropout=0,
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        config=None,
        config_name='bert-base-uncased',
        training_mode='emb',
        vocab_size=None,
        experiment_mode='lm',
        init_pretrained=False,
        logits_mode=1,
        layout=0,
        constrained=None,
        self_cond=False,
    ):
        super().__init__()
        self.constrained=constrained
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout
            # config.hidden_size = 512
            
        ##for rico dataset
        self.layout=layout


        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.logits_mode = logits_mode
        self.training_mode=training_mode
        self.self_cond=self_cond
        self.vocab_size=vocab_size

        if training_mode == 'e2e':
            self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
            if self.logits_mode == 2:
                # self.lm_head = nn.Linear(self.in_channels, vocab_size, bias=False)
                self.lm_head = nn.Linear(self.in_channels, vocab_size, bias=True)

            else:
                self.lm_head = nn.Linear(self.in_channels, vocab_size)
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
        elif training_mode == 'e2e-simple':
            self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
            self.lm_head = nn.Linear(self.in_channels, vocab_size)
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight

        if experiment_mode == 'conditional_gen':
            self.conditional_gen = True
            self.encoder_emb = nn.Embedding(vocab_size, config.hidden_size)
            self.encoder = BertEncoder(config)
            config.is_decoder = True
            config.add_cross_attention = True
        elif experiment_mode == 'lm':
            self.conditional_gen = False


        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)


        # self.input_up_proj = trans_nd(config, in_channels, model_channels // attention_head_size, attention_head_size)
        if self.self_cond:
            self.input_up_proj = nn.Sequential(nn.Linear(2*in_channels, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        else:
            self.input_up_proj = nn.Sequential(nn.Linear(in_channels, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        if init_pretrained:
            from transformers.models.bert.modeling_bert import BertModel
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            del temp_bert.embeddings
            del temp_bert.pooler
            self.input_transformers = temp_bert.encoder
        else:
            self.input_transformers = BertEncoder(config)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, out_channels))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2:
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  # vocab
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  # d, bsz*seqlen
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)  # bsz*seqlen, 1
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  # (vocab, d) x (d, bsz*seqlen)
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) # vocab, bsz*seqlen
            scores = -scores.permute(1, 2, 0).contiguous()

            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps, self_cond=None, y=None, src_ids=None, src_mask=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.

        """
        
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.conditional_gen:
            assert src_ids is not None
            src_emb = self.encoder_emb(src_ids)
            encoder_hidden_states = self.encoder(src_emb).last_hidden_state
            encoder_attention_mask = src_mask.unsqueeze(1).unsqueeze(1)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        if self.constrained=='type':
            mask=y.le(self.vocab_size-129).unsqueeze(-1) #(bz,len,1)
            x=self.word_embedding(y)*mask+x*(~mask)

        if self.self_cond:
            self_cond = default(self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((self_cond, x), dim = -1)

        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))
        if self.conditional_gen:
            input_trans_hidden_states = self.input_transformers(emb_inputs,
                                                                encoder_hidden_states=encoder_hidden_states,
                                                                encoder_attention_mask=encoder_attention_mask,
                                                                ).last_hidden_state
        else:
            input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        h = self.output_down_proj(input_trans_hidden_states) 

        h = h.type(x.dtype)

        return h

class DiscreteTransformerModel(nn.Module):
    """
    Discrete transformer
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        num_res_blocks,
        dropout=0,
        num_classes=None,
        use_checkpoint=False,
        config=None,
        config_name='bert-base-uncased',
        training_mode='emb',
        vocab_size=None,
        init_pretrained=False,
        constrained=None,
        self_cond=False,
        matrix_policy=0,
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout
            # config.hidden_size = 512
            
        self.constrained=constrained
        self.in_channels = 768
        self.model_channels = model_channels
        self.out_channels = vocab_size-1
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.training_mode=training_mode
        self.self_cond=self_cond
        self.matrix_policy=matrix_policy

        if training_mode == 'discrete':
            self.word_embedding = nn.Embedding(vocab_size, self.in_channels)
            
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        # self.input_up_proj = trans_nd(config, in_channels, model_channels // attention_head_size, attention_head_size)
        
        self.input_up_proj = nn.Sequential(nn.Linear(self.in_channels, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        if init_pretrained:
            from transformers.models.bert.modeling_bert import BertModel
            temp_bert = BertModel.from_pretrained(config_name, config=config)
            del temp_bert.embeddings
            del temp_bert.pooler
            self.input_transformers = temp_bert.encoder
        else:
            self.input_transformers = BertEncoder(config)

        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, self.out_channels))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids)

    def index_to_log_onehot(self,x):
        num_classes=self.out_channels
        assert x.max().item() < num_classes, \
            f'Error: {x.max().item()} >= {num_classes}'
        x_onehot = F.one_hot(x, num_classes)
        permute_order = (0, -1) + tuple(range(1, len(x.size())))
        x_onehot = x_onehot.permute(permute_order)
        log_x = torch.log(x_onehot.float().clamp(min=1e-30))
        return log_x

    def forward(self, x, timesteps, y=None):

        if self.constrained=='type':
            mask=y.le(self.out_channels-129).unsqueeze(-1) #(bz,len,1)
            x=self.word_embedding(y)*mask+self.word_embedding(x)*(~mask)

        elif self.constrained=='format':
            mask=y.le(4).unsqueeze(-1)
            x=self.word_embedding(y)*mask+self.word_embedding(x)*(~mask)

        elif self.constrained=='completion' and y is not None:
            # mask1=y.le(4).unsqueeze(-1) #(bz,len,1)
            mask2=torch.tensor([1]*6+[0]*(y.shape[1]-6)).expand(y.shape[0],-1).unsqueeze(-1).bool().to(x.device)
            # mask=mask1|mask2
            mask=mask2
            x=self.word_embedding(y)*mask+self.word_embedding(x)*(~mask)
        else:    
            x=self.word_embedding(x)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        emb_x = self.input_up_proj(x)
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length ]
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.dropout(self.LayerNorm(emb_inputs))

        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        h = self.output_down_proj(input_trans_hidden_states)
        
        h = rearrange(h, 'b l c -> b c l')
        
        h = h.type(x.dtype)

        return h
