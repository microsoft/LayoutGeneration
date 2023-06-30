# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# modified By Junyi Zhang
# ------------------------------------------

from random import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast
from tqdm import tqdm

eps = 1e-8

def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.to(t.device).gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def gaussian_matrix2(t,bt):
    r"""Computes transition matrix for q(x_t|x_{t-1}).

    This method constructs a transition matrix Q with
    decaying entries as a function of how far off diagonal the entry is.
    Normalization option 1:
    Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= transition_bands
             1 - \sum_{l \neq i} Q_{il}  if i==j.
             0                          else.

    Normalization option 2:
    tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= transition_bands
                     0                        else.

    Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

    Args:
      t: timestep. integer scalar (or numpy array?)

    Returns:
      Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    num_pixel_vals=128
    transition_bands = num_pixel_vals - 1

    beta_t = bt.numpy()[t]

    mat = np.zeros((num_pixel_vals, num_pixel_vals),
                    dtype=np.float64)

    # Make the values correspond to a similar type of gaussian as in the
    # gaussian diffusion case for continuous state spaces.
    values = np.linspace(start=0., stop=127., num=num_pixel_vals,
                          endpoint=True, dtype=np.float64)
    values = values * 2./ (num_pixel_vals - 1.)
    values = values[:transition_bands+1]
    values = -values * values / beta_t

    values = np.concatenate([values[:0:-1], values], axis=0)
    values = np.exp(values)/np.sum(np.exp(values),axis=0)
    values = values[transition_bands:]
    for k in range(1, transition_bands + 1):
      off_diag = np.full(shape=(num_pixel_vals - k,),
                          fill_value=values[k],
                          dtype=np.float64)

      mat += np.diag(off_diag, k=k)
      mat += np.diag(off_diag, k=-k)

    # Add diagonal values such that rows and columns sum to one.
    # Technically only the ROWS need to sum to one
    # NOTE: this normalization leads to a doubly stochastic matrix,
    # which is necessary if we want to have a uniform stationary distribution.
    diag = 1. - mat.sum(1)
    mat += np.diag(diag, k=0)

    return mat

def alpha_schedule(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999, matrix_policy=0, type_classes=25):

    if matrix_policy==1: #for gaussian refine
        sep=5
        sep_1=sep-1
        att= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.0001 - 0.99999) + 0.99999,
        np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.000009- 0.00009) + 0.00009))
        att = np.concatenate(([1], att))
        at = att[1:]/att[:-1]

        att1= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.9999 - 0.99999) + 0.99999,
        np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.000009- 0.9999) + 0.9999))
        att1 = np.concatenate(([1], att1))
        at1 = att1[1:]/att1[:-1]

        ctt= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.00009 - 0.000009) + 0.000009,
        np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.9999- 0.0001) + 0.0001))
        ctt = np.concatenate(([0], ctt))
        one_minus_ctt = 1 - ctt
        one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
        ct = 1-one_minus_ct

        ctt1= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.00009 - 0.000009) + 0.000009,
        np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.9998- 0.00009) + 0.00009))
        ctt1 = np.concatenate(([0], ctt1))
        one_minus_ctt1 = 1 - ctt1
        one_minus_ct1 = one_minus_ctt1[1:] / one_minus_ctt1[:-1]
        ct1 = 1-one_minus_ct1

        att = np.concatenate((att[1:], [1]))
        ctt = np.concatenate((ctt[1:], [0]))
        att1 = np.concatenate((att1[1:], [1]))
        ctt1 = np.concatenate((ctt1[1:], [0]))
        btt1 = (1-att1-ctt1) / type_classes
        btt2 = (1-att-ctt)

        bt1 = (1-at1-ct1) / type_classes #bt for type_classestypes
        btt2 = np.concatenate(([0], btt2))
        one_minus_btt2 = 1 - btt2
        one_minus_bt2 = one_minus_btt2[1:] / one_minus_btt2[:-1]
        bt2 = 1-one_minus_bt2
        btt2 = (1-att-ctt)/128

        bt2=np.concatenate((bt2[:time_step*sep_1//sep],at1[time_step*sep_1//sep:]/128))
        at=np.concatenate((at[:time_step*sep_1//sep],(1-ct-bt2*128)[time_step*sep_1//sep:])).clip(min=1e-30)
        ct=np.concatenate(((1-at-bt2)[:time_step*sep_1//sep],ct[time_step*sep_1//sep:])).clip(min=1e-30)

        return at,at1, bt1,bt2, ct,ct1, att,att1, btt1,btt2, ctt,ctt1 
        
    if matrix_policy==2: #gaussian refine (wo bbox absorb) produce similar results as policy=1
        sep=5
        sep_1=sep-1

        att= np.concatenate((np.arange(0, time_step*4//5)/(time_step*4//5-1)*(0.0001 - 0.99999) + 0.99999,
        np.arange(0, time_step-time_step*4//5)/(time_step-time_step*4//5-1)*(0.99999- 0.0001) + 0.0001))
        att = np.concatenate(([1], att))
        at = att[1:]/att[:-1]

        att1= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.9999 - 0.99999) + 0.99999,
        np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.000009- 0.9998) + 0.9998))
        att1 = np.concatenate(([1], att1))
        at1 = att1[1:]/att1[:-1]

        ctt= np.arange(0, time_step)/(time_step-1)*(0.00009 - 0.000009) + 0.000009
        ctt = np.concatenate(([0], ctt))
        one_minus_ctt = 1 - ctt
        one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
        ct = 1-one_minus_ct

        ctt1= np.concatenate((np.arange(0, time_step*sep_1//sep)/(time_step*sep_1//sep-1)*(0.00009 - 0.000009) + 0.000009,
        np.arange(0, time_step-time_step*sep_1//sep)/(time_step-time_step*sep_1//sep-1)*(0.9999- 0.0001) + 0.0001))
        ctt1 = np.concatenate(([0], ctt1))
        one_minus_ctt1 = 1 - ctt1
        one_minus_ct1 = one_minus_ctt1[1:] / one_minus_ctt1[:-1]
        ct1 = 1-one_minus_ct1

        att = np.concatenate((att[1:], [1]))
        ctt = np.concatenate((ctt[1:], [0]))
        att1 = np.concatenate((att1[1:], [1]))
        ctt1 = np.concatenate((ctt1[1:], [0]))
        btt1 = (1-att1-ctt1) / type_classes
        btt2 = (1-att-ctt)

        bt1 = (1-at1-ct1) / type_classes
        btt2 = np.concatenate(([0], btt2))
        btt2[-1]=1.0
        one_minus_btt2 = 1 - btt2
        one_minus_bt2 = one_minus_btt2[1:] / one_minus_btt2[:-1]
        bt2 = 1-one_minus_bt2
        return at,at1, bt1,bt2, ct,ct1, att,att1, btt1,btt2, ctt,ctt1


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        *,
        diffusion_step=100,
        alpha_init_type='alpha1',
        auxiliary_loss_weight=0.001,
        adaptive_auxiliary_loss=False,
        mask_weight=[1,1],
        learnable_cf=False,
        matrix_policy=1,
        att_1=0.99999,
        num_classes=159,
        rescale_weight=False,
        content_seq_len=121,
        alignment_loss=False,
        alignment_weight=1e5,
        pow_num=2.5,
        mul_num=12.4,
    ):
        super().__init__()
        self.gaussian_matrix = False
        self.wo_bbox_absorb = False    
        self.alignment_loss=alignment_loss
        self.alignment_weight=alignment_weight
        self.ori_schedule_type=alpha_init_type

        if alpha_init_type=='gaussian_refine_pow2.5_wo_bbox_absorb':
            self.wo_bbox_absorb = True #for no mask gaussian refine

        if alpha_init_type=='gaussian_refine_pow2.5': # in case refine case overwrite replace schedule
            self.gaussian_matrix = True #for gaussian
            
        if type(matrix_policy)==tuple:
            matrix_policy=matrix_policy[0]
        self.schedule_type=alpha_init_type
        self.content_seq_len = content_seq_len
        self.amp = False
        self.num_classes = num_classes #self.num_classes-1+1
        # if matrix_policy==1:
        #     self.num_classes+=1
        self.type_classes = num_classes-1-128-5 #-1 is the mask token; -128 is the coord token; -5 is the special token
        self.loss_type = 'vb_stochastic'
        self.shape = content_seq_len
        self.num_timesteps = diffusion_step
        self.parametrization = 'x0'
        self.adaptive_auxiliary_loss = adaptive_auxiliary_loss
        self.rescale_weight=rescale_weight
        if adaptive_auxiliary_loss:
            self.auxiliary_loss_weight = auxiliary_loss_weight
        else:
            self.auxiliary_loss_weight=0

        self.mask_weight = mask_weight

        if self.gaussian_matrix: #gaussian refine
            at,at1, bt1,bt2, ct,ct1, att,att1, btt1,btt2, ctt,ctt1 = alpha_schedule(self.num_timesteps, N=self.num_classes-1, matrix_policy=1)

        elif self.wo_bbox_absorb: #gaussian refine no absorb
            at,at1, bt1,bt2, ct,ct1, att,att1, btt1,btt2, ctt,ctt1 = alpha_schedule(self.num_timesteps, N=self.num_classes-1, matrix_policy=2)

        at1 = torch.tensor(at1.astype('float64'))
        ct1 = torch.tensor(ct1.astype('float64'))
        log_at1 = torch.log(at1).clamp(-70,0)
        log_ct1 = torch.log(ct1).clamp(-70,0)
        att1 = torch.tensor(att1.astype('float64'))
        ctt1 = torch.tensor(ctt1.astype('float64'))
        log_cumprod_at1 = torch.log(att1).clamp(-70,0)
        log_cumprod_ct1 = torch.log(ctt1).clamp(-70,0)

        log_1_min_ct1 = log_1_min_a(log_ct1)
        log_1_min_cumprod_ct1 = log_1_min_a(log_cumprod_ct1)
        assert log_add_exp(log_ct1, log_1_min_ct1).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct1, log_1_min_cumprod_ct1).abs().sum().item() < 1.e-5
        self.register_buffer('log_ct1', log_ct1.float())
        self.register_buffer('log_at1', log_at1.float())
        self.register_buffer('log_cumprod_at1', log_cumprod_at1.float())
        self.register_buffer('log_cumprod_ct1', log_cumprod_ct1.float())
        self.register_buffer('log_1_min_ct1', log_1_min_ct1.float())
        self.register_buffer('log_1_min_cumprod_ct1', log_1_min_cumprod_ct1.float())

        at = torch.tensor(at.astype('float64'))
        bt1 = torch.tensor(bt1.astype('float64'))
        bt2 = torch.tensor(bt2.astype('float64'))
        ct = torch.tensor(ct.astype('float64'))
        log_at = torch.log(at)
        log_bt1 = torch.log(bt1)
        log_bt2 = torch.log(bt2)
        log_ct = torch.log(ct).clamp(-70,0)
        att = torch.tensor(att.astype('float64'))
        btt1 = torch.tensor(btt1.astype('float64'))
        btt2 = torch.tensor(btt2.astype('float64'))
        ctt = torch.tensor(ctt.astype('float64'))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt1 = torch.log(btt1)
        log_cumprod_bt2 = torch.log(btt2)
        log_cumprod_ct = torch.log(ctt).clamp(-70,0)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1.e-5
        # Convert to float32 and register buffers.
        self.register_buffer('log_at', log_at.float())
        self.register_buffer('log_bt1', log_bt1.float())
        self.register_buffer('log_bt2', log_bt2.float())
        self.register_buffer('log_ct', log_ct.float())
        self.register_buffer('log_cumprod_at', log_cumprod_at.float())
        self.register_buffer('log_cumprod_bt1', log_cumprod_bt1.float())
        self.register_buffer('log_cumprod_bt2', log_cumprod_bt2.float())
        self.register_buffer('log_cumprod_ct', log_cumprod_ct.float())
        self.register_buffer('log_1_min_ct', log_1_min_ct.float())
        self.register_buffer('log_1_min_cumprod_ct', log_1_min_cumprod_ct.float())
        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        bt2=torch.where(bt2==0.,bt2.max(),bt2)
        q_one_step_mats = [gaussian_matrix2(t,bt=bt2.pow(2).pow(pow_num/2)*mul_num)
                        for t in range(0, self.num_timesteps)]
        q_one_step_mats.append(np.ones((128,128))/(128**2))
        q_onestep_mats = np.stack(q_one_step_mats, axis=0)
        q_onestep_mats=torch.from_numpy(q_onestep_mats).float()
        self.register_buffer('q_onestep_mats', q_onestep_mats)
        # Construct transition matrices for q(x_t|x_start)
        q_mat_t = self.q_onestep_mats[0]
        q_mats = [q_mat_t]
        for t in range(1, self.num_timesteps):
            # Q_{1...t} = Q_{1 ... t-1} Q_t = Q_1 Q_2 ... Q_t
            q_mat_t = np.tensordot(q_mat_t, self.q_onestep_mats[t],
                                    axes=[[1], [0]])
            q_mats.append(q_mat_t)
        q_mats.append(np.ones((128,128))/(128**2))
        q_mats = np.stack(q_mats, axis=0)
        q_mats=torch.from_numpy(q_mats).float()
        self.register_buffer('q_mats', q_mats)
        assert self.q_mats.shape == (self.num_timesteps+1, 128,
                                    128), self.q_mats.shape

        self.diffusion_acc_list = [0] * self.num_timesteps
        self.diffusion_keep_list = [0] * self.num_timesteps
        
        self.zero_vector = None

        self.prior_rule = 0    # inference rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
        self.prior_ps = 1024   # max number to sample per step
        self.prior_weight = 0  # probability adjust parameter, 'r' in Equation.11 of Improved VQ-Diffusion

        
        self.learnable_cf = learnable_cf

    def multinomial_kl(self, log_prob1, log_prob2):   # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t, t):         # q(xt|xt_1)
        device=log_x_t.device
        bz=log_x_t.shape[0]

        log_at = extract(self.log_at, t, log_x_t.shape)             # at
        log_bt1 = extract(self.log_bt1, t, log_x_t.shape)             # bt
        log_bt2 = extract(self.log_bt2, t, log_x_t.shape)             # bt
        log_ct = extract(self.log_ct, t, log_x_t.shape)             # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)          # 1-ct

        if self.gaussian_matrix: #for gaussian_relu
            log_at1 = extract(self.log_at1, t, log_x_t.shape)         # at~
            log_ct1 = extract(self.log_ct1, t, log_x_t.shape)         # ct~
            log_1_min_ct1 = extract(self.log_1_min_ct1, t, log_x_t.shape)       # 1-ct~
            
            mask=(t<(self.num_timesteps*4//5)).unsqueeze(-1).unsqueeze(-1)

            matrix1=torch.cat([
            torch.cat([torch.eye(5,device=device).expand(bz,-1,-1),torch.zeros(bz,5,self.num_classes-5,device=device)],dim=-1),
            torch.cat([
            torch.zeros(bz,self.type_classes,5,device=device),
            log_add_exp(torch.eye(self.type_classes,device=device).clamp(min=1e-30).log().expand(bz,-1,-1) +log_at1, log_bt1).exp(),
            torch.zeros(bz,self.type_classes,self.num_classes-5-self.type_classes,device=device)],dim=-1),
            torch.cat([
            torch.zeros(bz,128,self.type_classes+5,device=device),
            log_add_exp(torch.eye(128,device=device).clamp(min=1e-30).log().expand(bz,-1,-1) +log_at, log_bt2).exp(),
            torch.zeros(bz,128,self.num_classes-5-self.type_classes-128,device=device)],dim=-1),
            torch.cat([torch.zeros(bz,1,5,device=device),
            log_add_exp(torch.zeros(bz,1,self.type_classes,device=device).clamp(min=1e-30).log() +log_1_min_ct1, log_ct1).exp(),
            log_add_exp(torch.zeros(bz,1,128,device=device).clamp(min=1e-30).log() +log_1_min_ct, log_ct).exp(),
            torch.ones(bz,1,1,device=device)],dim=-1),
            ],dim=-2)

            matrix2=torch.cat([
            torch.cat([torch.eye(5,device=device).expand(bz,-1,-1),torch.zeros(bz,5,self.num_classes-5,device=device)],dim=-1),
            torch.cat([
            torch.zeros(bz,self.type_classes,5,device=device),
            log_add_exp(torch.eye(self.type_classes,device=device).clamp(min=1e-30).log().expand(bz,-1,-1) +log_at1, log_bt1).exp(),
            torch.zeros(bz,self.type_classes,self.num_classes-5-self.type_classes,device=device)],dim=-1),
            torch.cat([
            torch.zeros(bz,128,self.type_classes+5,device=device),
            self.q_onestep_mats[t].to(device),
            torch.zeros(bz,128,self.num_classes-5-self.type_classes-128,device=device)],dim=-1),
            torch.cat([torch.zeros(bz,1,5,device=device),
            log_add_exp(torch.zeros(bz,1,self.type_classes,device=device).clamp(min=1e-30).log() +log_1_min_ct1, log_ct1).exp(),
            log_add_exp(torch.zeros(bz,1,128,device=device).clamp(min=1e-30).log() +log_1_min_ct, log_ct).exp(),
            torch.ones(bz,1,1,device=device)],dim=-1),
            ],dim=-2)

            matrix=torch.where(mask, matrix2, matrix1)

        elif self.wo_bbox_absorb:
            log_at1 = extract(self.log_at1, t, log_x_t.shape)         # at~
            log_ct1 = extract(self.log_ct1, t, log_x_t.shape)         # ct~
            log_1_min_ct1 = extract(self.log_1_min_ct1, t, log_x_t.shape)       # 1-ct~

            matrix=torch.cat([
            torch.cat([torch.eye(5,device=device).expand(bz,-1,-1),torch.zeros(bz,5,self.num_classes-5,device=device)],dim=-1),
            
            torch.cat([torch.zeros(bz,self.type_classes,5,device=device),
            log_add_exp(torch.eye(self.type_classes,device=device).clamp(min=1e-30).log().expand(bz,-1,-1) +log_at1, log_bt1).exp(),
            torch.zeros(bz,self.type_classes,self.num_classes-5-self.type_classes,device=device)],dim=-1),

            torch.cat([
            torch.zeros(bz,128,self.type_classes+5,device=device),
            self.q_onestep_mats[t].to(device),
            torch.zeros(bz,128,self.num_classes-5-self.type_classes-128,device=device)],dim=-1),

            torch.cat([torch.zeros(bz,1,5,device=device),
            log_add_exp(torch.zeros(bz,1,self.type_classes,device=device).clamp(min=1e-30).log() +log_1_min_ct1, log_ct1).exp(),
            log_add_exp(torch.zeros(bz,1,128,device=device).clamp(min=1e-30).log() +log_1_min_ct, log_ct).exp(),
            torch.ones(bz,1,1,device=device)],dim=-1),
            ],dim=-2)

        log_probs=matrix.matmul(log_x_t.exp()).clamp(min=1e-30).log()

        return log_probs

    def q_pred(self, log_x_start, t):           # q(xt|x0)
        device=log_x_start.device
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        bz=log_x_start.shape[0]

        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)         # at~
        log_cumprod_bt1 = extract(self.log_cumprod_bt1, t, log_x_start.shape)         # bt~
        log_cumprod_bt2 = extract(self.log_cumprod_bt2, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~

        if self.gaussian_matrix:
            log_cumprod_at1 = extract(self.log_cumprod_at1, t, (1,1,1))         # at~
            log_cumprod_ct1 = extract(self.log_cumprod_ct1, t, (1,1,1))         # ct~
            log_1_min_cumprod_ct1 = extract(self.log_1_min_cumprod_ct1, t, (1,1,1))       # 1-ct~
            
            mask=(t<(self.num_timesteps*4//5)).unsqueeze(-1).unsqueeze(-1)
            
            matrix1=torch.cat([
            torch.cat([torch.eye(5,device=device).expand(bz,-1,-1),torch.zeros(bz,5,self.num_classes-5,device=device)],dim=-1),
            torch.cat([
            torch.zeros(bz,self.type_classes,5,device=device),
            log_add_exp(torch.eye(self.type_classes,device=device).clamp(min=1e-30).log().expand(bz,-1,-1) +log_cumprod_at1, log_cumprod_bt1).exp(),
            torch.zeros(bz,self.type_classes,self.num_classes-5-self.type_classes,device=device)],dim=-1),
            torch.cat([
            torch.zeros(bz,128,self.type_classes+5,device=device),
            log_add_exp(torch.eye(128,device=device).clamp(min=1e-30).log().expand(bz,-1,-1) +log_cumprod_at, log_cumprod_bt2).exp(),
            torch.zeros(bz,128,self.num_classes-5-self.type_classes-128,device=device)],dim=-1),
            torch.cat([torch.zeros(bz,1,5,device=device),
            log_add_exp(torch.zeros(bz,1,self.type_classes,device=device).clamp(min=1e-30).log() +log_1_min_cumprod_ct1, log_cumprod_ct1).exp(),
            log_add_exp(torch.zeros(bz,1,128,device=device).clamp(min=1e-30).log() +log_1_min_cumprod_ct, log_cumprod_ct).exp(),
            torch.ones(bz,1,1,device=device)],dim=-1),
            ],dim=-2)

            matrix2=torch.cat([
            torch.cat([torch.eye(5,device=device).expand(bz,-1,-1),torch.zeros(bz,5,self.num_classes-5,device=device)],dim=-1),
            torch.cat([
            torch.zeros(bz,self.type_classes,5,device=device),
            log_add_exp(torch.eye(self.type_classes,device=device).clamp(min=1e-30).log().expand(bz,-1,-1) +log_cumprod_at1, log_cumprod_bt1).exp(),
            torch.zeros(bz,self.type_classes,self.num_classes-5-self.type_classes,device=device)],dim=-1),
            torch.cat([
            torch.zeros(bz,128,self.type_classes+5,device=device),
            self.q_mats[t].to(device),
            torch.zeros(bz,128,self.num_classes-5-self.type_classes-128,device=device)],dim=-1),
            torch.cat([torch.zeros(bz,1,5,device=device),
            log_add_exp(torch.zeros(bz,1,self.type_classes,device=device).clamp(min=1e-30).log() +log_1_min_cumprod_ct1, log_cumprod_ct1).exp(),
            log_add_exp(torch.zeros(bz,1,128,device=device).clamp(min=1e-30).log() +log_1_min_cumprod_ct, log_cumprod_ct).exp(),
            torch.ones(bz,1,1,device=device)],dim=-1),
            ],dim=-2)
            matrix=torch.where(mask,matrix2,matrix1)

        elif self.wo_bbox_absorb:
            log_cumprod_at1 = extract(self.log_cumprod_at1, t, (1,1,1))         # at~
            log_cumprod_ct1 = extract(self.log_cumprod_ct1, t, (1,1,1))         # ct~
            log_1_min_cumprod_ct1 = extract(self.log_1_min_cumprod_ct1, t, (1,1,1))       # 1-ct~

            matrix=torch.cat([
            torch.cat([torch.eye(5,device=device).expand(bz,-1,-1),torch.zeros(bz,5,self.num_classes-5,device=device)],dim=-1),
            torch.cat([
            torch.zeros(bz,self.type_classes,5,device=device),
            log_add_exp(torch.eye(self.type_classes,device=device).clamp(min=1e-30).log().expand(bz,-1,-1) +log_cumprod_at1, log_cumprod_bt1).exp(),
            torch.zeros(bz,self.type_classes,self.num_classes-5-self.type_classes,device=device)],dim=-1),
            torch.cat([
            torch.zeros(bz,128,self.type_classes+5,device=device),
            self.q_mats[t].to(device),
            torch.zeros(bz,128,self.num_classes-5-self.type_classes-128,device=device)],dim=-1),

            torch.cat([torch.zeros(bz,1,5,device=device),
            log_add_exp(torch.zeros(bz,1,self.type_classes,device=device).clamp(min=1e-30).log() +log_1_min_cumprod_ct1, log_cumprod_ct1).exp(),
            log_add_exp(torch.zeros(bz,1,128,device=device).clamp(min=1e-30).log() +log_1_min_cumprod_ct, log_cumprod_ct).exp(),
            torch.ones(bz,1,1,device=device)],dim=-1),
            ],dim=-2)

        log_probs=matrix.matmul(log_x_start.exp()).clamp(min=1e-30).log()
            
        return log_probs

    def predict_start(self, log_x_t, model,y, t):          # p(x0|xt)
        x_t = log_onehot_to_index(log_x_t)
        if self.amp == True:
            with autocast():
                out = model(x_t, t,y=y)
        else:
            out = model(x_t, t, y=y)
        assert out.size(0) == x_t.size(0)
        assert out.size(1) == self.num_classes-1
        assert out.size()[2:] == x_t.size()[1:]
        
        log_pred = F.log_softmax(out.double(), dim=1).float()
        batch_size = log_x_t.size()[0]
        if self.zero_vector is None or self.zero_vector.shape[0] != batch_size:
            self.zero_vector = torch.zeros(batch_size, 1, self.content_seq_len).type_as(log_x_t)- 70
        log_pred = torch.cat((log_pred, self.zero_vector), dim=1)
        log_pred = torch.clamp(log_pred, -70, 0)
        return log_pred
    
    def cf_predict_start(self, log_x_t,model,y, t):
        return self.predict_start(log_x_t,model, y, t)

    def q_posterior(self, log_x_start, log_x_t, t):            # p_theta(xt_1|xt) = sum(q(xt-1|xt,x0')*p(x0'))
        # notice that log_x_t is onehot
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps
        batch_size = log_x_start.size()[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == self.num_classes-1).unsqueeze(1)
        log_one_vector = torch.zeros(batch_size, 1, 1).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1.0e-30).expand(-1, -1, self.content_seq_len)
        log_zero_vector_aux=torch.log(log_one_vector+1.0e-30).expand(-1, -1, -1)
        log_qt = self.q_pred(log_x_t, t)                                  # q(xt|x0)
        log_qt = log_qt[:,:-1,:]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)         # ct~

        if self.schedule_type=='relu':
            log_cumprod_ct1 = extract(self.log_cumprod_ct1, t, log_x_start.shape)         # ct~
            ct_cumprod_vector = torch.cat([log_zero_vector_aux.expand(-1, 5, -1),log_cumprod_ct1.expand(-1, self.type_classes, -1),log_cumprod_ct.expand(-1, self.num_classes-1-self.type_classes-5, -1)],dim=1)
        else:
            ct_cumprod_vector = torch.cat([log_zero_vector_aux.expand(-1, 5, -1),log_cumprod_ct.expand(-1, self.num_classes-1-5, -1)],dim=1)

        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector

        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t)        # q(xt|xt_1)
        log_qt_one_timestep = torch.cat((log_qt_one_timestep[:,:-1,:], log_zero_vector), dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)         # ct

        if self.schedule_type=='relu':
            log_ct1 = extract(self.log_ct1, t, log_x_start.shape)         # ct~
            ct_vector = torch.cat([log_zero_vector_aux.expand(-1, 5, -1),log_ct1.expand(-1, self.type_classes, -1),log_ct.expand(-1, self.num_classes-1-self.type_classes-5, -1)],dim=1)
        else:
            ct_vector = torch.cat([log_zero_vector_aux.expand(-1, 5, -1),log_ct.expand(-1, self.num_classes-1-5, -1)],dim=1)

        ct_vector = torch.cat((ct_vector, log_one_vector), dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector
        q = log_x_start[:,:-1,:] - log_qt
        q = torch.cat((q, log_zero_vector), dim=1)

        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1) + log_qt_one_timestep

        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70, 0)

    def p_pred(self, log_x,model,  t):             # if x0, first p(x0|xt), than sum(q(xt-1|xt,x0)*p(x0|xt))
        if self.parametrization == 'x0':
            log_x_recon = self.cf_predict_start(log_x, model, t)
            log_model_pred = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t)
        elif self.parametrization == 'direct':
            log_model_pred = self.predict_start(log_x,model,  t)
        else:
            raise ValueError
        return log_model_pred, log_x_recon

    @torch.no_grad()
    def p_sample(self, log_x,model,  t, sampled=None, to_sample=None):               # sample q(xt-1) for next step from  xt, actually is p(xt-1|xt)
        model_log_prob, log_x_recon = self.p_pred(log_x, model, t)

        max_sample_per_step = self.prior_ps  # max number to sample per step
        if t[0] > 0 and self.prior_rule > 0 and to_sample is not None: # prior_rule: 0 for VQ-Diffusion v1, 1 for only high-quality inference, 2 for purity prior
            log_x_idx = log_onehot_to_index(log_x)

            if self.prior_rule == 1:
                score = torch.ones((log_x.shape[0], log_x.shape[2])).to(log_x.device)
            elif self.prior_rule == 2:
                score = torch.exp(log_x_recon).max(dim=1).values.clamp(0, 1)
                score /= (score.max(dim=1, keepdim=True).values + 1e-10)

            if self.prior_rule != 1 and self.prior_weight > 0:
                # probability adjust parameter, prior_weight: 'r' in Equation.11 of Improved VQ-Diffusion
                prob = ((1 + score * self.prior_weight).unsqueeze(1) * log_x_recon).softmax(dim=1)
                prob = prob.log().clamp(-70, 0)
            else:
                prob = log_x_recon

            out = self.log_sample_categorical(prob)
            out_idx = log_onehot_to_index(out)

            out2_idx = log_x_idx.clone()
            _score = score.clone()
            if _score.sum() < 1e-6:
                _score += 1
            _score[log_x_idx != self.num_classes - 1] = 0

            for i in range(log_x.shape[0]):
                n_sample = min(to_sample - sampled[i], max_sample_per_step)
                if to_sample - sampled[i] - n_sample == 1:
                    n_sample = to_sample - sampled[i]
                if n_sample <= 0:
                    continue
                sel = torch.multinomial(_score[i], n_sample)
                out2_idx[i][sel] = out_idx[i][sel]
                sampled[i] += ((out2_idx[i] != self.num_classes - 1).sum() - (log_x_idx[i] != self.num_classes - 1).sum()).item()

            out = index_to_log_onehot(out2_idx, self.num_classes)
        else:
            # Gumbel sample
            out = self.log_sample_categorical(model_log_prob)
            sampled = [1024] * log_x.shape[0]

        if to_sample is not None:
            return out, sampled
        else:
            return out

    def log_sample_categorical(self, logits):           # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)
        log_sample = index_to_log_onehot(sample, self.num_classes)
        return log_sample

    def q_sample(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def q_sample_onestep(self, log_x_start, t):                 # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred_one_timestep(log_x_start, t)
        log_sample = self.log_sample_categorical(log_EV_qxt_x0)

        return log_sample

    def sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()
            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)

            return t, pt

        elif method == 'uniform':
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError

    def noise_process_new(self,x_start,sep):
        xt_all=[]
        bz=x_start.shape[0]
        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        for t in range(self.num_timesteps):
            log_xt = self.q_sample_onestep(log_x_start=log_x_start,t=torch.tensor([t]).cuda().expand(bz))
            xt=log_onehot_to_index(log_xt)
            
            if t in np.linspace(0, self.num_timesteps-1, num=sep).astype(int):
                xt_all.append(xt.cpu().detach().numpy())
            log_x_start = index_to_log_onehot(xt, self.num_classes)
        return xt_all
    @property
    def device(self):
        return self.transformer.to_logits[-1].weight.device ##todo

    def parameters(self, recurse=True, name=None):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # return super().parameters(recurse=True)
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            # separate out all parameters to those that will and won't experience regularizing weight decay
            print("GPTLikeTransformer: get parameters by the overwrite method!")
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (torch.nn.Linear, )
            blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                    if pn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
            # special case the position embedding parameter as not decayed
            module_name = ['content_emb']
            pos_emb_name = ['pos_emb', 'width_emb', 'height_emb', 'pad_emb', 'token_type_emb']
            for mn in module_name:
                if hasattr(self, mn) and getattr(self, mn) is not None:
                    for pn in pos_emb_name:
                        if hasattr(getattr(self, mn), pn):
                            if isinstance(getattr(getattr(self, mn), pn), torch.nn.Parameter):
                                no_decay.add('{}.{}'.format(mn, pn))

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.transformer.named_parameters()}# if p.requires_grad} ##todo
            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]
            return optim_groups

    def training_losses(
        self,
        model,
        x,
        model_kwargs=None,
        is_train=True
        ):
        b, device = x.size(0), x.device

        assert self.loss_type == 'vb_stochastic'
        x_start = x
        t, pt = self.sample_time(b, device, 'importance')
        log_x_start = index_to_log_onehot(x_start, self.num_classes)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t) #gt x_t #use matrix
        xt = log_onehot_to_index(log_xt)

        ############### go to p_theta function ###############
        log_x0_recon = self.predict_start(log_xt, model,y=model_kwargs['input_ids'], t=t) 

        log_model_prob = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t)      # go through q(xt_1|xt,x0) #pred x_t

        ################## compute acc list ################
        x0_recon = log_onehot_to_index(log_x0_recon)
        x0_real = x_start

        xt_1_recon = log_onehot_to_index(log_model_prob)
        xt_recon = log_onehot_to_index(log_xt)

        for index in range(t.size()[0]):
            this_t = t[index].item()
            same_rate = (x0_recon[index] == x0_real[index]).sum().cpu()/x0_real.size()[1] #the ratio of same codebook
            self.diffusion_acc_list[this_t] = same_rate.item()*0.1 + self.diffusion_acc_list[this_t]*0.9
            same_rate = (xt_1_recon[index] == xt_recon[index]).sum().cpu()/xt_recon.size()[1]
            self.diffusion_keep_list[this_t] = same_rate.item()*0.1 + self.diffusion_keep_list[this_t]*0.9

        # compute log_true_prob now 
        log_true_prob = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t)

        # compute loss
        kl = self.multinomial_kl(log_true_prob, log_model_prob)
        
        mask_region = (xt == self.num_classes-1).float()
        mask_weight = mask_region * self.mask_weight[0] + (1. - mask_region) * self.mask_weight[1]

        kl = kl * mask_weight
        kl = sum_except_batch(kl)
        
        decoder_nll = -log_categorical(log_x_start, log_model_prob)
        decoder_nll = sum_except_batch(decoder_nll)

        mask = (t == torch.zeros_like(t)).float()

        kl_loss = mask * decoder_nll + (1. - mask) * kl

        Lt2 = kl_loss.pow(2)
        Lt2_prev = self.Lt_history.to(model.device).gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.to(model.device).scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.to(model.device).scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))
        
        # Upweigh loss term of the kl
        # vb_loss = kl_loss / pt + kl_prior
        loss1 = kl_loss / pt 
        vb_loss={}
        vb_loss['loss1'] = loss1

        if self.auxiliary_loss_weight != 0 and is_train==True:
            kl_aux = self.multinomial_kl(log_x_start[:,:-1,:], log_x0_recon[:,:-1,:])
            kl_aux = kl_aux * mask_weight
            kl_aux = sum_except_batch(kl_aux)
            kl_aux_loss = mask * decoder_nll + (1. - mask) * kl_aux
            if self.adaptive_auxiliary_loss == True:
                addition_loss_weight = (1-t/self.num_timesteps) + 1.0
            else:
                addition_loss_weight = 1.0

            loss2 = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss / pt
            vb_loss['loss2'] = loss2
            vb_loss['loss']=vb_loss['loss1']+vb_loss['loss2']
        else:
            vb_loss['loss'] = vb_loss['loss1']
        if self.rescale_weight:
            vb_loss['loss']=vb_loss['loss']*0.2+0.8*2*vb_loss['loss']*(self.num_timesteps-t)/self.num_timesteps
        
        return vb_loss

    def sample_fast(
            self,
            model,
            sample_shape,
            sample_start_step=200,
            content_token = None,
            filter_ratio = 0,
            multistep=False,
            skip_step = 0,
            constrained = None,
            **kwargs):
        y=content_token

        if sample_start_step!=self.num_timesteps:
            gen_refine=True
            assert y is not None
        else:
            gen_refine=False
        batch_size = sample_shape[0]
        device = model.word_embedding.weight.device

        zero_logits = torch.zeros((batch_size, self.num_classes-1, self.shape),device=device)
        one_logits = torch.ones((batch_size, 1, self.shape),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)

        if gen_refine:
            mask_logits=index_to_log_onehot(y,self.num_classes).exp()
            print("****we are in refine generation***")

        if not gen_refine:
            from torch.distributions.categorical import Categorical
            if self.num_classes==159:
                print("sampling rico...")
                probs=torch.tensor([0.04849498, 0.03704171, 0.0534486,  0.06045308, 0.06354515, 0.07585032,
                0.08045687, 0.0644917,  0.05676153, 0.05742412, 0.05471067, 0.04944153,
                0.04552912, 0.04190068, 0.04426705, 0.0387455,  0.03533792, 0.03167792,
                0.02997413, 0.0304474 ])
            elif self.num_classes==139:
                print("sampling publaynet...")
                probs=torch.tensor([0.00321776,0.03342678, 0.04233181, 0.04218409, 0.05404355, 0.07231605,
                0.08247029, 0.0905211,  0.0949399,  0.0959322,  0.08953522, 0.07810608,
                0.0619627,  0.04775897, 0.03585776, 0.0261788,  0.018812,   0.01404317,
                0.00972071, 0.00664104])

            m = Categorical(probs)
            mask=self.num_classes-1
            if y is None or constrained=='completion':
                if constrained=='completion':
                    x_start=y
                samples=m.sample(torch.tensor([batch_size]))
            else: #constrained
                type_mask=torch.tensor([0,1,0,0,0,0]*20+[0]).to(device).bool()
                padding_mask=(torch.masked_select(y,type_mask)!=3).reshape(batch_size,-1)
                samples=padding_mask.sum(-1)-1

            head=torch.tensor([0,mask,mask,mask,mask,mask],device=device)
            body=torch.tensor([4,mask,mask,mask,mask,mask],device=device)
            bottom=torch.tensor([1],device=device)
            input=torch.ones(batch_size,self.content_seq_len,dtype=torch.int64,device=device)*3 ##padding

            for i in range(batch_size):
                tmp=torch.cat([head,body.repeat(samples[i]),bottom],dim=-1)
                input[i][:len(tmp)]=tmp

            if constrained=='type':
                print("type constrained")
                uniform=torch.ones(128)/128
                m = Categorical(uniform)
                noise=m.sample(torch.tensor([batch_size,self.content_seq_len])).to(device)+self.type_classes+5
                bbox_mask=torch.tensor([0,0,1,1,1,1]*20+[0]).to(device)
                input=torch.where((bbox_mask==1)&(input==mask),noise,input)
                sample_start_step=160

            if self.ori_schedule_type.startswith('gaussian_refine_pow2.5_wo_bbox_absorb'): #wo bbox absorb
                print("wo type absorb")
                uniform=torch.ones(128)/128
                m = Categorical(uniform)
                noise=m.sample(torch.tensor([batch_size,self.content_seq_len])).to(device)+self.type_classes+5
                bbox_mask=torch.tensor([0,0,1,1,1,1]*20+[0]).to(device)
                input=torch.where((bbox_mask==1)&(input==mask),noise,input)

            mask_logits=index_to_log_onehot(input,self.num_classes).exp()

        log_z = torch.log(mask_logits)
        start_step = int(sample_start_step)
        print("sample from timestep", start_step)
        with torch.no_grad():
            if multistep:
                tmp=[]
            diffusion_list = [index for index in range(start_step-1, -1, -1-skip_step)]
            if diffusion_list[-1] != 0:
                diffusion_list.append(0)
            for diffusion_index in tqdm(diffusion_list):
                t = torch.full((batch_size,), diffusion_index, device=device, dtype=torch.long)
                log_x_recon = self.cf_predict_start(log_z, model,y,t)
                if diffusion_index > skip_step:
                    model_log_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t-skip_step)
                else:
                    model_log_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t)
                
                if constrained=='completion': #align the y gap between training and sampling
                    if diffusion_index==start_step-1:
                        log_x_recon = self.cf_predict_start(log_z, model,y=None,t=t)
                        model_log_prob = self.q_posterior(log_x_start=log_x_recon, log_x_t=log_z, t=t)
                    y_log_prob = self.q_posterior(log_x_start=index_to_log_onehot(x_start,self.num_classes), log_x_t=log_z, t=t)
                    y=log_onehot_to_index(y_log_prob)

                log_z = self.log_sample_categorical(model_log_prob)
                if multistep and (diffusion_index in np.linspace(0, start_step-1, num=20).astype(int)):
                    tmp.append(log_onehot_to_index(log_z))

        content_token = log_onehot_to_index(log_z)

        if multistep:
            output=torch.stack(tmp,0)
        else:
            output =content_token

        return output