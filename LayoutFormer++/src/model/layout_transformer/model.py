# coding=utf8
import random
import copy
from typing import Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
         Unmasked positions are filled with float(0.0).
     """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_token = nn.Parameter(torch.rand(max_len, 1, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pos_token[:x.size(0)]
        return self.dropout(x)


class LayoutTransformer(nn.Module):

    def __init__(self, vocab_size: int, max_len: int, bos_token_id: int, pad_token_id: int,
                 eos_token_id: int, d_model: int, nhead: int, num_layers: int, dropout: int,
                 d_feedforward: int = None, share_embedding: bool = False, add_task_embedding: bool = False,
                 num_task_embedding: int = 1, add_task_prompt_token: bool = False, num_task_prompt_token: int = 1):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        self.enc_embedding = nn.Embedding(vocab_size, embedding_dim=d_model)
        self.enc_pos_embedding = PositionalEncoding(
            d_model, dropout, max_len=max_len)

        if d_feedforward is None:
            d_feedforward = d_model * 4
        te = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout,
                                        dim_feedforward=d_feedforward)
        self.encoder = nn.TransformerEncoder(te, num_layers=num_layers)

        # Decoder
        if share_embedding:
            self.dec_embedding = self.enc_embedding
        else:
            self.dec_embedding = nn.Embedding(
                vocab_size, embedding_dim=d_model)

        self.dec_pos_embedding = PositionalEncoding(
            d_model, dropout, max_len=max_len)
        de = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout,
                                        dim_feedforward=d_feedforward)
        self.decoder = nn.TransformerDecoder(de, num_layers=num_layers)

        # Output Layer
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.dec_embedding.weight

        self.task_embedding = None
        if add_task_embedding:
            print("Enable Task Embedding")
            self.task_embedding = nn.Embedding(num_task_embedding, d_model)

        self.task_prompt_embed = None
        if add_task_prompt_token:
            print(
                f"Add {num_task_prompt_token} task prompt tokens in Transformer")
            self.num_task_prompt_token = num_task_prompt_token
            self.task_prompt_embed = nn.Parameter(torch.Tensor(num_task_embedding, num_task_prompt_token,
                                                               d_model), requires_grad=True)
            nn.init.normal_(self.task_prompt_embed)

        if add_task_embedding and add_task_prompt_token:
            raise TypeError(
                "add_task_embedding and add_task_prompt_token is mutually exclusive")

    def encode(self, input_ids: torch.LongTensor, padding_mask: torch.BoolTensor,
               task_ids: torch.LongTensor = None) -> torch.Tensor:
        if self.task_prompt_embed is not None:
            assert task_ids is not None
            x = self.enc_embedding(input_ids)
            prompts = self.task_prompt_embed[task_ids]
            x = torch.cat([prompts, x], dim=1).permute(1, 0, 2)
            bsz = input_ids.size(0)
            enc_padding_mask = torch.cat([padding_mask.new_zeros(
                bsz, self.num_task_prompt_token).bool(), padding_mask], dim=1)
        else:
            x = self.enc_embedding(input_ids).permute(1, 0, 2)
            enc_padding_mask = padding_mask

        x = self.enc_pos_embedding(x)
        enc_hs = self.encoder(x, src_key_padding_mask=enc_padding_mask)

        if self.task_embedding is not None:
            task_embed = self.task_embedding(task_ids).unsqueeze(dim=0)
            enc_hs += task_embed

        return enc_hs, enc_padding_mask

    def forward(self, input_ids: torch.LongTensor, padding_mask: torch.BoolTensor,
                labels: torch.LongTensor = None, max_length: int = 512,
                do_sample: bool = False, top_k: int = 10, temperature: float = 0.7,
                constrained_decoding: bool = False,
                generation_constraint_fn: Callable = None,
                loss_weights: torch.Tensor = None,
                task_ids: torch.Tensor = None) -> Dict:
        if do_sample:
            return self.top_k_sample(input_ids, padding_mask, max_length=max_length,
                                     top_k=top_k, temperature=temperature, task_ids=task_ids)
        elif labels is not None:
            return self.compute_loss(input_ids, padding_mask, labels, task_ids=task_ids,
                                     loss_weights=loss_weights)
        else:
            if constrained_decoding:
                return self.decoding_space_restriction(input_ids, padding_mask, max_length=max_length,
                                                       generation_constraint_fn=generation_constraint_fn,
                                                       task_ids=task_ids)
            else:
                return self.generate(input_ids, padding_mask, max_length=max_length,
                                     generation_constraint_fn=generation_constraint_fn,
                                     task_ids=task_ids)

    def compute_loss(self, input_ids: torch.LongTensor, padding_mask: torch.BoolTensor,
                     labels: torch.LongTensor, task_ids: torch.LongTensor = None,
                     loss_weights: torch.Tensor = None,) -> Dict:
        enc_hs, enc_padding_mask = self.encode(
            input_ids, padding_mask, task_ids)

        # training
        bsz, _ = input_ids.size()
        dec_input_ids = torch.cat([labels.new_ones((bsz, 1)) * self.bos_token_id,
                                   labels[:, :-1]], dim=1)
        dec_input = self.dec_embedding(dec_input_ids).permute(1, 0, 2)
        dec_input = self.dec_pos_embedding(dec_input)
        tgt_mask = generate_square_subsequent_mask(
            dec_input.size(0)).to(dec_input.device)

        # decoder
        y = self.decoder(tgt=dec_input, memory=enc_hs, tgt_mask=tgt_mask,
                         memory_key_padding_mask=enc_padding_mask)
        logits = self.out(y.permute(1, 0, 2))

        _labels = torch.clone(labels)
        _labels[_labels == self.pad_token_id] == -100
        if loss_weights is None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), _labels.view(-1))
        else:
            bsz, length, _ = logits.size()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            loss = loss_fct(logits.view(-1, logits.size(-1)), _labels.view(-1))
            loss = loss.reshape(bsz, length).mean(dim=-1)  # (bsz,)
            loss *= loss_weights
            loss = loss.mean()
        return {
            'loss': loss, 'logits': logits
        }

    def generate(self, input_ids: torch.LongTensor, padding_mask: torch.BoolTensor, max_length: int,
                 generation_constraint_fn: Callable = None, task_ids: torch.LongTensor = None) -> Dict:
        enc_hs, enc_padding_mask = self.encode(input_ids, padding_mask, task_ids)

        bsz = input_ids.size(0)
        device = input_ids.device
        step_outs, step_logits, stop_indicators = list(), list(), input_ids.new_zeros(bsz, dtype=bool)
        pred_ids = input_ids.new_ones((bsz, 1)) * self.bos_token_id
        for idx in range(max_length):
            curr_len = idx + 1
            dec_input = self.dec_embedding(pred_ids).permute(1, 0, 2)
            dec_input = self.dec_pos_embedding(dec_input)
            tgt_mask = generate_square_subsequent_mask(curr_len).to(device)

            y = self.decoder(tgt=dec_input, memory=enc_hs, tgt_mask=tgt_mask,
                             memory_key_padding_mask=enc_padding_mask)
            curr_logits = self.out(y.permute(1, 0, 2)[:, -1, :])

            if generation_constraint_fn is not None:
                if len(step_outs) > 0:
                    _curr_out = torch.stack(step_outs, dim=1)
                else:
                    _curr_out = tgt_mask.new_empty(bsz, 0)
                num_vocabs = curr_logits.size(-1)
                for bidx in range(bsz):
                    plasubile_mask = tgt_mask.new_ones(num_vocabs).bool()
                    plasubile_token_ids, _ = generation_constraint_fn(bidx, idx, _curr_out[bidx, :])
                    plasubile_mask[plasubile_token_ids] = False
                    curr_logits[bidx].masked_fill_(plasubile_mask, -float('Inf'))

            step_logits.append(curr_logits)

            curr_pred = torch.argmax(curr_logits, dim=-1)
            is_eos = (curr_pred == self.eos_token_id)
            curr_pred[stop_indicators] = self.pad_token_id

            step_outs.append(curr_pred)
            pred_ids = torch.cat([pred_ids, curr_pred.unsqueeze(dim=1)], dim=1)
            stop_indicators = torch.logical_or(stop_indicators, is_eos)

            if torch.all(stop_indicators):
                break

        outs = torch.stack(step_outs, dim=1)
        return {
            "output": outs
        }

    def decoding_space_restriction(self, input_ids: torch.LongTensor, padding_mask: torch.BoolTensor, max_length: int,
                                   generation_constraint_fn: Callable = None, task_ids: torch.LongTensor = None, top_k: int = 10, temperature: float = 0.7) -> Dict:
        enc_hs, enc_padding_mask = self.encode(
            input_ids, padding_mask, task_ids)

        bsz = input_ids.size(0)
        device = input_ids.device
        step_outs, step_logits, step_prob, stop_indicators = list(
        ), list(), list(), input_ids.new_zeros(bsz, dtype=bool)
        pred_ids = input_ids.new_ones((bsz, 1)) * self.bos_token_id
        idx = 0
        flag_idx = list()
        back_flag = False
        while idx < max_length:
            curr_len = idx + 1
            dec_input = self.dec_embedding(pred_ids).permute(1, 0, 2)
            dec_input = self.dec_pos_embedding(dec_input)
            tgt_mask = generate_square_subsequent_mask(curr_len).to(device)

            y = self.decoder(tgt=dec_input, memory=enc_hs, tgt_mask=tgt_mask,
                             memory_key_padding_mask=enc_padding_mask)
            curr_logits = self.out(y.permute(1, 0, 2)[:, -1, :])
            curr_prob = torch.softmax(curr_logits, dim=-1)

            # prune by constrained decoding
            back_idx = None
            if generation_constraint_fn is not None:
                if len(step_outs) > 0:
                    _curr_out = torch.stack(step_outs, dim=1)
                else:
                    _curr_out = tgt_mask.new_empty(bsz, 0)
                num_vocabs = curr_logits.size(-1)
                for bidx in range(bsz):
                    plasubile_mask = tgt_mask.new_ones(num_vocabs).bool()
                    plasubile_token_ids, back_idx = generation_constraint_fn(
                        bidx, idx, _curr_out[bidx, :])
                    plasubile_mask[plasubile_token_ids] = False
                    curr_logits[bidx].masked_fill_(
                        plasubile_mask, -float('Inf'))

            # prune by currente token probability
            prob_gate = 0.3
            pruned_curr_logits = copy.deepcopy(curr_logits)
            num_vocabs = curr_logits.size(-1)
            plasubile_mask = tgt_mask.new_ones(num_vocabs).bool()
            pruned_mask = tgt_mask.new_zeros(num_vocabs).bool()
            pruned_mask = torch.where(
                curr_prob[0] < prob_gate, plasubile_mask, pruned_mask)
            pruned_curr_logits[0].masked_fill_(pruned_mask, -float('Inf'))

            # check and back
            if (not back_flag) and ((back_idx) or (pruned_curr_logits[0].max() == -float('Inf'))) and (flag_idx.count(idx) < 3):
                back_flag = True
                flag_idx.append(idx)
                if back_idx:
                    idx = back_idx
                else:
                    idx = random.randint(0, max(0, idx-1))
                step_outs = step_outs[:idx]
                step_logits = step_logits[:idx]
                step_prob = step_prob[:idx]
                pred_ids = pred_ids[:, :idx+1]
                continue

            if pruned_curr_logits[0].max() != -float('Inf'):
                curr_logits[0] = pruned_curr_logits[0]

            if back_flag:
                back_flag = False
                # # Dropout last max prob toke
                # temp_curr_logits = torch.softmax(curr_logits[0],dim=0)
                # if temp_curr_logits.max() != 1:
                #     curr_logits[0][curr_pred] = -float('Inf')

                # Scale by temperature
                scaled_curr_logits = curr_logits / 1.5
            else:
                scaled_curr_logits = curr_logits / temperature

            # crop probabilities to only the top k options
            scaled_curr_logits = top_k_logits(scaled_curr_logits, top_k)
            probs = F.softmax(scaled_curr_logits, dim=-1)
            curr_pred = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)

            step_logits.append(curr_logits)
            step_prob.append(curr_prob[0][curr_pred])
            is_eos = (curr_pred == self.eos_token_id)
            curr_pred[stop_indicators] = self.pad_token_id

            step_outs.append(curr_pred)
            pred_ids = torch.cat([pred_ids, curr_pred.unsqueeze(dim=1)], dim=1)
            stop_indicators = torch.logical_or(stop_indicators, is_eos)

            if torch.all(stop_indicators):
                break

            idx += 1

        outs = torch.stack(step_outs, dim=1)
        return {
            "output": outs
        }

    def top_k_sample(self, input_ids: torch.LongTensor, padding_mask: torch.BoolTensor, max_length: int,
                     top_k: int = 10, temperature: float = 0.7, task_ids: torch.LongTensor = None) -> Dict:
        enc_hs, enc_padding_mask = self.encode(
            input_ids, padding_mask, task_ids)

        bsz = input_ids.size(0)
        device = input_ids.device
        step_outs, step_logits, stop_indicators = list(
        ), list(), input_ids.new_zeros(bsz, dtype=bool)
        pred_ids = input_ids.new_ones((bsz, 1)) * self.bos_token_id
        for idx in range(max_length):
            curr_len = idx + 1
            dec_input = self.dec_embedding(pred_ids).permute(1, 0, 2)
            dec_input = self.dec_pos_embedding(dec_input)
            tgt_mask = generate_square_subsequent_mask(curr_len).to(device)

            y = self.decoder(tgt=dec_input, memory=enc_hs, tgt_mask=tgt_mask,
                             memory_key_padding_mask=enc_padding_mask)
            curr_logits = self.out(y.permute(1, 0, 2)[:, -1, :])

            # Scale by temperature
            curr_logits = curr_logits / temperature
            # crop probabilities to only the top k options
            curr_logits = top_k_logits(curr_logits, top_k)
            step_logits.append(curr_logits)

            # sample
            probs = F.softmax(curr_logits, dim=-1)
            curr_pred = torch.multinomial(probs, num_samples=1).squeeze(dim=-1)

            is_eos = (curr_pred == self.eos_token_id)
            curr_pred[stop_indicators] = self.pad_token_id

            step_outs.append(curr_pred)
            pred_ids = torch.cat([pred_ids, curr_pred.unsqueeze(dim=1)], dim=1)
            stop_indicators = torch.logical_or(stop_indicators, is_eos)

            if torch.all(stop_indicators):
                break

        outs = torch.stack(step_outs, dim=1)
        return {
            "output": outs
        }
