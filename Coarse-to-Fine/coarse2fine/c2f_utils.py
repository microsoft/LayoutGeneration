import numpy as np
import torch
import torch.nn as nn


def repeat_unit(fill_unit, max_lens, cur_len):
    if fill_unit.dim() < 2:
        return fill_unit.repeat((max_lens - cur_len), 1)
    elif fill_unit.dim() == 2:
        return fill_unit.repeat((max_lens - cur_len), 1, 1)
    raise NotImplementedError


def to_dense_batch(batch, max_lens=None):
    '''
    padding a batch of data with value 0
    '''
    unsqueeze_flag = False
    if batch[-1].dim() == 1:
        unsqueeze_flag = True
        batch = [data.unsqueeze(-1) for data in batch]

    lens = [len(data) for data in batch]
    if max_lens is None:
        max_lens = max(lens)

    fill_size = batch[-1][-1].size()
    fill_unit = torch.zeros(fill_size, dtype=batch[-1].dtype)

    out = torch.cat((batch[0], repeat_unit(fill_unit, max_lens, lens[0])),
                    dim=0).unsqueeze(0)
    for i in range(1, len(lens)):
        out = torch.cat((out, (torch.cat(
            (batch[i], repeat_unit(fill_unit, max_lens, lens[i])), dim=0).unsqueeze(0))), dim=0)
    if unsqueeze_flag:
        out = out.squeeze(-1)

    mask = [[True] * i + [False] * (max_lens - i) for i in lens]
    mask = torch.from_numpy(np.array(mask))

    return out, mask


def padding(data, device):
    # padding sequential box and label
    labels, masks = to_dense_batch(data['labels'])
    bboxes, _ = to_dense_batch(data['bboxes'])
    labels, bboxes, masks = labels.to(device), bboxes.to(device), masks.to(device)

    # padding groupbox and grouplabel
    label_in_one_group, group_masks = to_dense_batch(data['label_in_one_group'])
    label_in_one_group = label_in_one_group.type(torch.float)
    group_bounding_box, _ = to_dense_batch(data['group_bounding_box'])

    # padding hierarchical box and label
    grouped_labels = data['grouped_label']
    grouped_bboxes = data['grouped_box']
    grouped_ele_masks = list()
    max_num_group_ele = 0
    for i in range(len(grouped_labels)):  # layout
        for j in range(len(grouped_labels[i])):  # group
            if len(grouped_labels[i][j]) > max_num_group_ele:
                max_num_group_ele = len(grouped_labels[i][j])

    for i in range(len(grouped_labels)):  # layout
        grouped_labels[i], mask_i = to_dense_batch(grouped_labels[i], max_num_group_ele)
        grouped_bboxes[i], _ = to_dense_batch(grouped_bboxes[i], max_num_group_ele)
        grouped_ele_masks.append(mask_i)

    grouped_labels, _ = to_dense_batch(grouped_labels)
    grouped_bboxes, _ = to_dense_batch(grouped_bboxes)
    grouped_ele_masks, _ = to_dense_batch(grouped_ele_masks)

    return {
        'bboxes': bboxes,
        'labels': labels,
        'masks': masks,
        'group_bounding_box': group_bounding_box,
        'label_in_one_group': label_in_one_group,
        'group_masks': group_masks,
        'grouped_bboxes': grouped_bboxes,
        'grouped_labels': grouped_labels,
        'grouped_ele_masks': grouped_ele_masks
    }


def get_mask(ori, device):

    mask = ori['masks']
    # ori_label_mask = torch.from_numpy(np.zeros([mask.size(0), mask.size(1)], dtype='int64')).to(device)
    ori_box_mask = torch.from_numpy(np.zeros([mask.size(0), mask.size(1)], dtype='int64')).to(device)
    mask_number = (mask == 1).cumsum(dim=1)
    for i in range(mask.size(0)):
        # ori_label_mask[i][1:mask_number[i][-1]] = mask[i][1:mask_number[i][-1]]
        ori_box_mask[i][1:mask_number[i][-1]-1] = mask[i][1:mask_number[i][-1]-1]

    gourp_mask = ori['group_masks']
    # group bounding box mask and group in one label mask
    rec_group_bounding_box_mask = torch.from_numpy(np.zeros([gourp_mask.size(0), gourp_mask.size(1)], dtype='int64')).to(device)
    ori_group_bounding_box_mask = torch.from_numpy(np.zeros([gourp_mask.size(0), gourp_mask.size(1)], dtype='int64')).to(device)
    rec_label_in_one_group_mask = torch.from_numpy(np.zeros([gourp_mask.size(0), gourp_mask.size(1)], dtype='int64')).to(device)
    ori_label_in_one_group_mask = torch.from_numpy(np.zeros([gourp_mask.size(0), gourp_mask.size(1)], dtype='int64')).to(device)
    group_mask_number = (gourp_mask == 1).cumsum(dim=-1)
    for i in range(gourp_mask.size(0)):
        rec_label_in_one_group_mask[i][:group_mask_number[i][-1]-1] = gourp_mask[i][:group_mask_number[i][-1]-1]
        ori_label_in_one_group_mask[i][1:group_mask_number[i][-1]] = gourp_mask[i][1:group_mask_number[i][-1]]
        rec_group_bounding_box_mask[i][:group_mask_number[i][-1]-2] = gourp_mask[i][:group_mask_number[i][-1]-2]
        ori_group_bounding_box_mask[i][1:group_mask_number[i][-1]-1] = gourp_mask[i][1:group_mask_number[i][-1]-1]

    grouped_ele_mask = ori['grouped_ele_masks']
    # grouped_box mask and grouped_label mask
    rec_grouped_box_mask = torch.from_numpy(np.zeros([grouped_ele_mask.size(0), grouped_ele_mask.size(1), grouped_ele_mask.size(2)], dtype='int64')).to(device)
    ori_grouped_box_mask = torch.from_numpy(np.zeros([grouped_ele_mask.size(0), grouped_ele_mask.size(1), grouped_ele_mask.size(2)], dtype='int64')).to(device)
    rec_grouped_label_mask = torch.from_numpy(np.zeros([grouped_ele_mask.size(0), grouped_ele_mask.size(1), grouped_ele_mask.size(2)], dtype='int64')).to(device)
    ori_grouped_label_mask = torch.from_numpy(np.zeros([grouped_ele_mask.size(0), grouped_ele_mask.size(1), grouped_ele_mask.size(2)], dtype='int64')).to(device)
    mask_number = (grouped_ele_mask == 1).cumsum(dim=-1)
    for i in range(grouped_ele_mask.size(0)):
        for j in range(grouped_ele_mask.size(1)):
            rec_grouped_label_mask[i][j][:mask_number[i][j][-1]-1] = grouped_ele_mask[i][j][:mask_number[i][j][-1]-1]
            ori_grouped_label_mask[i][j][1:mask_number[i][j][-1]] = grouped_ele_mask[i][j][1:mask_number[i][j][-1]]
            rec_grouped_box_mask[i][j][:mask_number[i][j][-1]-2] = grouped_ele_mask[i][j][:mask_number[i][j][-1]-2]
            ori_grouped_box_mask[i][j][1:mask_number[i][j][-1]-1] = grouped_ele_mask[i][j][1:mask_number[i][j][-1]-1]

    mask_info = {
        # "ori_label_mask": ori_label_mask,
        "ori_box_mask": ori_box_mask,
        "rec_label_in_one_group_mask": rec_label_in_one_group_mask,
        "ori_label_in_one_group_mask": ori_label_in_one_group_mask,
        "rec_group_bounding_box_mask": rec_group_bounding_box_mask,
        "ori_group_bounding_box_mask": ori_group_bounding_box_mask,
        "rec_grouped_label_mask": rec_grouped_label_mask,
        "ori_grouped_label_mask": ori_grouped_label_mask,
        "rec_grouped_box_mask": rec_grouped_box_mask,
        "ori_grouped_box_mask": ori_grouped_box_mask
    }

    return mask_info


def cal_loss(args, ori, rec, kl_info, device):

    ce_loss_func = nn.CrossEntropyLoss()
    mse_loss_func = nn.MSELoss()

    d_box = max(args.discrete_x_grid, args.discrete_y_grid)

    masks = get_mask(ori, device)
    ori_grouped_bboxes = ori['grouped_bboxes'].reshape(-1, 4)
    ori_grouped_labels = ori['grouped_labels'].reshape(-1).squeeze(-1).long()
    ori_grouped_box_masks = masks['ori_grouped_box_mask'].reshape(-1).squeeze(-1).long()
    ori_grouped_label_masks = masks['ori_grouped_label_mask'].reshape(-1).squeeze(-1).long()

    ori_group_bounding_box = ori['group_bounding_box'].reshape(-1, 4)
    ori_label_in_one_group = ori['label_in_one_group'].reshape(-1, args.num_labels+2)
    ori_group_bounding_box_masks = masks['ori_group_bounding_box_mask'].reshape(-1).squeeze(-1).long()
    ori_label_in_one_group_masks = masks['ori_label_in_one_group_mask'].reshape(-1).squeeze(-1).long()

    rec_grouped_bboxes = rec['grouped_bboxes'].reshape(-1, 4, d_box)
    rec_grouped_labels = rec['grouped_labels'].reshape(-1, args.num_labels+3)
    rec_grouped_box_masks = masks['rec_grouped_box_mask'].reshape(-1).squeeze(-1).long()
    rec_grouped_label_masks = masks['rec_grouped_label_mask'].reshape(-1).squeeze(-1).long()

    rec_group_bounding_box = rec['group_bounding_box'].reshape(-1, 4, d_box)
    rec_label_in_one_group = rec['label_in_one_group'].reshape(-1, args.num_labels+2)
    rec_group_bounding_box_masks = masks['rec_group_bounding_box_mask'].reshape(-1).squeeze(-1).long()
    rec_label_in_one_group_masks = masks['rec_label_in_one_group_mask'].reshape(-1).squeeze(-1).long()

    group_bounding_box_loss = args.group_box_weight * ce_loss_func(rec_group_bounding_box[rec_group_bounding_box_masks.bool()].reshape(-1, d_box), ori_group_bounding_box[ori_group_bounding_box_masks.bool()].reshape(-1).squeeze(-1).long())
    label_in_one_group_loss = args.group_label_weight * mse_loss_func(rec_label_in_one_group[rec_label_in_one_group_masks.bool()], ori_label_in_one_group[ori_label_in_one_group_masks.bool()])

    grouped_box_loss = args.box_weight * ce_loss_func(rec_grouped_bboxes[rec_grouped_box_masks.bool()].reshape(-1, d_box), ori_grouped_bboxes[ori_grouped_box_masks.bool()].reshape(-1).squeeze(-1).long())
    grouped_label_loss = args.label_weight * ce_loss_func(rec_grouped_labels[rec_grouped_label_masks.bool()], ori_grouped_labels[ori_grouped_label_masks.bool()])

    # KL divergence
    logvar = kl_info['logvar']
    mu = kl_info['mu']
    kl_loss = args.kl_weight * -0.5 * torch.mean(1 + logvar - mu.pow(2) - torch.exp(logvar))

    return {
        "group_bounding_box": group_bounding_box_loss,
        "label_in_one_group": label_in_one_group_loss,
        "grouped_box": grouped_box_loss,
        "grouped_label": grouped_label_loss,
        "KL": kl_loss
    }
