import argparse
import copy
import os.path as osp
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from coarse2fine.c2f_model.model import C2FLayoutTransformer
from data import PubLayNetDataset, RicoDataset, LayoutDataset, transforms
from utils import config, os_utils, utils
from coarse2fine.c2f_trainer import Trainer
from coarse2fine.c2f_generator import Generator
from evaluation import metrics
from coarse2fine.cut_hierarchy import CutHierarchy
from coarse2fine.c2f_utils import padding, cal_loss, get_mask


def add_sos_and_eos(data, group_info, sos, eos):
    # add sos and eos for sequential boxes and labels
    discrete_gold_bboxes = data['discrete_gold_bboxes']
    labels = data['labels']
    discrete_gold_bboxes = F.pad(discrete_gold_bboxes, (0, 0, 1, 1,), 'constant', 0)
    labels = F.pad(labels, (1, 0), 'constant', sos)  # sos
    labels = F.pad(labels, (0, 1), 'constant', eos)  # eos

    # add sos and eos for groupbox and grouplabel
    group_bounding_box = F.pad(group_info['group_bounding_box'], (0, 0, 1, 1,), 'constant', 0)
    label_in_one_group = F.pad(group_info['label_in_one_group'], (1, 1, 1, 1,), 'constant', 0)
    label_in_one_group[0][0] = 1  # SOS
    label_in_one_group[-1][-1] = 1  # EOS

    # add sos and eos for hierarchical boxes and labels
    grouped_label = group_info['grouped_label']
    grouped_box = group_info['grouped_box']
    # pad_grouped_label = torch.zeros(grouped_label.size(0),grouped_label.size(1))
    for i in range(len(grouped_label)):
        grouped_box[i] = F.pad(grouped_box[i], (0, 0, 1, 1,), 'constant', 0)
        grouped_label[i] = F.pad(grouped_label[i], (1, 0), 'constant', sos)  # sos
        grouped_label[i] = F.pad(grouped_label[i], (0, 1), 'constant', eos)  # eos

    return {
        'bboxes': discrete_gold_bboxes,
        'labels': labels,
        'group_bounding_box': group_bounding_box,
        'label_in_one_group': label_in_one_group,
        'grouped_box': grouped_box,
        'grouped_label': grouped_label
    }


class C2FRicoDataset(RicoDataset):
    def __init__(self, args, split, online_process=False):
        self.transform_functions = [
            transforms.LexicographicSort(),
            transforms.DiscretizeBoundingBox(
                num_x_grid=args.discrete_x_grid,
                num_y_grid=args.discrete_y_grid)
        ]
        self.transform = T.Compose(self.transform_functions)
        self.get_hierarchy = CutHierarchy()
        self.args = args

        super().__init__(root=args.data_dir,
                         split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)

    def process(self, data):
        nd = self.transform(copy.deepcopy(data))
        group_info = self.get_hierarchy(nd, self.args.num_labels, self.transform_functions[1].discretize)
        processed_data = add_sos_and_eos(nd, group_info, self.args.num_labels+1, self.args.num_labels+2)
        processed_data['name'] = data['name']
        return processed_data


class C2FPubLayNetDataset(PubLayNetDataset):
    def __init__(self, args, split, online_process=False):
        self.transform_functions = [
            transforms.LexicographicSort(),
            transforms.DiscretizeBoundingBox(
                num_x_grid=args.discrete_x_grid,
                num_y_grid=args.discrete_y_grid)
        ]
        self.transform = T.Compose(self.transform_functions)
        self.get_hierarchy = CutHierarchy()
        self.args = args

        super().__init__(root=args.data_dir,
                         split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)

    def process(self, data):
        nd = self.transform(copy.deepcopy(data))
        group_info = self.get_hierarchy(nd, self.args.num_labels, self.transform_functions[1].discretize)
        processed_data = add_sos_and_eos(nd, group_info, self.args.num_labels+1, self.args.num_labels+2)
        processed_data['name'] = data['name']
        return processed_data


def create_dataset(args, split: str) -> LayoutDataset:
    if args.dataset == 'rico':
        return C2FRicoDataset(args, split, online_process=True)
    elif args.dataset == 'publaynet':
        return C2FPubLayNetDataset(args, split, online_process=True)
    raise NotImplementedError("No Valid Dataset")


def train_step(args, model, data, device):
    padded_data = padding(data, device)
    ori, rec, kl_info = model(padded_data, device)
    loss = cal_loss(args, ori, rec, kl_info, device)
    return loss


def eval_step(model, data, device):
    padded_data = padding(data, device)
    ori, rec, _ = model(padded_data, device, feed_gt=False)
    masks = get_mask(ori, device)

    rec_grouped_box_masks = masks['rec_grouped_box_mask']

    N, G, S, _ = rec['grouped_labels'].shape
    batch_labels = list()
    batch_bboxes = list()
    for i in range(N):
        labels, bboxes = None, None
        for j in range(G):
            group_i_j_label = rec['grouped_labels'][i][j].argmax(-1)[rec_grouped_box_masks[i][j].bool()].cpu()
            labels = group_i_j_label if labels is None else torch.cat((labels, group_i_j_label), dim=-1)
            group_i_j_bboxes = rec['grouped_bboxes'][i][j].argmax(-1)[rec_grouped_box_masks[i][j].bool()].cpu()
            bboxes = group_i_j_bboxes if bboxes is None else torch.cat((bboxes, group_i_j_bboxes), dim=-2)
        batch_labels.append(labels)
        batch_bboxes.append(bboxes)

    batch_labels, rec_masks = utils.to_dense_batch(batch_labels)
    batch_bboxes, _ = utils.to_dense_batch(batch_bboxes)
    rec['bboxes'] = batch_bboxes
    rec['labels'] = batch_labels
    # masks['rec_label_mask'] = rec_masks
    masks['rec_box_mask'] = rec_masks
    masks['ori_box_mask'] = masks['ori_box_mask'].bool()

    return ori, rec, masks


def rel_to_abs(elements_boxes, group_box, args):
    group_l = group_box[0]
    group_t = group_box[1]
    group_w = group_box[2]
    group_h = group_box[3]
    for i in range(len(elements_boxes)):
        elements_boxes[i][0] = int((elements_boxes[i][0]/args.discrete_x_grid) * group_w + group_l)
        elements_boxes[i][1] = int((elements_boxes[i][1]/args.discrete_y_grid) * group_h + group_t)
        elements_boxes[i][2] = int((elements_boxes[i][2]/args.discrete_x_grid) * group_w)
        elements_boxes[i][3] = int((elements_boxes[i][3]/args.discrete_y_grid) * group_h)
    return elements_boxes


def inference_step(args, model, data, device):
    padded_data = padding(data, device)

    ori = {
            'bboxes': padded_data['bboxes'],
            'labels': padded_data['labels'],
            'group_bounding_box': padded_data['group_bounding_box'],
            'label_in_one_group': padded_data['label_in_one_group'],
            }

    gen = model.inference(device)

    batch_label_in_one_group = list()
    batch_group_bounding_box = list()
    batch_labels = list()
    batch_bboxes = list()
    for i in range(len(gen['label_in_one_group'])):
        labels, bboxes = None, None
        for j in range(len(gen['label_in_one_group'][i])-2):
            # print(gen['label_in_one_group'][i][j])
            # input()
            if int(gen['label_in_one_group'][i][j].argmax(-1)) != (args.num_labels+1):
                for k in range(len(gen['grouped_labels'][i][j])):
                    # print(gen['grouped_labels'][i][j][k])
                    # input()
                    if int(gen['grouped_labels'][i][j][k].argmax(-1)) == (args.num_labels+2):
                        # print('break')
                        break
                group_i_j_label = gen['grouped_labels'][i][j][:k].argmax(-1).cpu()
                labels = group_i_j_label if labels is None else torch.cat((labels, group_i_j_label), dim=-1)
                group_i_j_bboxes = gen['grouped_bboxes'][i][j][:k].argmax(-1).cpu()
                group_i_j_bboxes = rel_to_abs(group_i_j_bboxes, gen['group_bounding_box'][i][j].argmax(-1).cpu(), args)
                bboxes = group_i_j_bboxes if bboxes is None else torch.cat((bboxes, group_i_j_bboxes), dim=-2)
            else:
                break
        label_in_one_group = gen['label_in_one_group'][i][:j].cpu()
        group_bounding_box = gen['group_bounding_box'][i][:j].argmax(-1).cpu()
        batch_label_in_one_group.append(label_in_one_group)
        batch_group_bounding_box.append(group_bounding_box)
        batch_labels.append(labels)
        batch_bboxes.append(bboxes)

    batch_labels, gen_masks = utils.to_dense_batch(batch_labels)
    batch_bboxes, _ = utils.to_dense_batch(batch_bboxes)
    gen['bboxes'] = batch_bboxes
    gen['labels'] = batch_labels
    _masks = get_mask(padded_data, device)
    masks = dict()
    masks['ori_box_mask'] = _masks['ori_box_mask'].bool()
    masks['gen_box_mask'] = gen_masks

    batch_label_in_one_group, gen_masks = utils.to_dense_batch(batch_label_in_one_group)
    batch_group_bounding_box, _ = utils.to_dense_batch(batch_group_bounding_box)
    gen['label_in_one_group'] = batch_label_in_one_group
    gen['group_bounding_box'] = batch_group_bounding_box
    masks['gen_group_bounding_box_mask'] = gen_masks

    return ori, gen, masks


def train(args):
    os_utils.makedirs(args.out_dir)

    task_config = {
        'kl_start_ste': args.kl_start_step,
        'kl_end_step': args.kl_end_step,
    }
    model = C2FLayoutTransformer(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=200, verbose=True, threshold=1e-6, threshold_mode='rel', cooldown=300, min_lr=0, eps=1e-06)

    train_dataset = create_dataset(args, split='train')
    val_dataset = create_dataset(args, split='val')

    print(f'train data: {len(train_dataset)}')
    print(f'valid data: {len(val_dataset)}')

    trainer = Trainer('coarse2fine', args, model, train_dataset, val_dataset, optimizer, scheduler, is_debug=True, is_label_condition=False, task_config=task_config)
    trainer(train_step, eval_step)
    trainer.clean_up()


def inference(args):
    os_utils.makedirs(args.out_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = C2FLayoutTransformer(args)
    model_path = os.path.join(args.out_dir, 'checkpoint.pth.tar')
    # state_dict = torch.load(model_path, map_location=device)

    # if torch.cuda.device_count() > 1:
    #     print("Use", torch.cuda.device_count(), "GPUs!")
    #     args.batch_size *= torch.cuda.device_count()
    #     model = nn.DataParallel(model)
    # else:
    #     state_dict = OD([(key.split("module.")[-1], state_dict[key]) for key in state_dict])
    # model.to(device)
    # model.load_state_dict(state_dict)

    test_dataset = create_dataset(args, split='test')
    print(f'test data: {len(test_dataset)}')

    fid_net_path = osp.join('net', f'fid_{args.dataset}.pth.tar')
    fid_model = metrics.LayoutFID(max_num_elements=args.max_num_elements, num_labels=args.num_labels, net_path=fid_net_path, device=device)

    generator = Generator(args, model, test_dataset, fid_model, model_path, is_label_condition=False)
    generator(inference_step, draw_colors=test_dataset.colors)


def add_task_arguments(parser):
    parser.add_argument('--num_labels', type=int, default=25,
                        help='number of labels')

    parser.add_argument('--d_model', type=int, default=512,
                        help='Transformer model dimensionality')
    parser.add_argument('--d_z', type=int, default=512,
                        help='latent z dimention')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of Encoder blocks')
    parser.add_argument('--n_layers_decoder', type=int, default=4,
                        help='Number of Decoder blocks')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Transformer config: number of heads')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help='Transformer config: FF dimensionality')

    parser.add_argument('--box_weight', type=float, default=1.0,
                        help='set box loss weight')
    parser.add_argument('--label_weight', type=float, default=1.0,
                        help='set label loss weight')
    parser.add_argument('--group_box_weight', type=float, default=1.0,
                        help='set box loss weight')
    parser.add_argument('--group_label_weight', type=float, default=10.0,
                        help='set label loss weight')
    parser.add_argument('--kl_weight', type=float, default=1.0,
                        help='set kl divergence loss weight')

    parser.add_argument('--kl_start_step', type=int, default=500,
                        help='at what epoch to start adding kld?')
    parser.add_argument('--kl_end_step', type=int, default=35000,
                        help='at what epoch the kl beta is 1.0?')


def add_trainer_arguments(parser) -> None:
    parser.add_argument('--local_rank', type=int, help='distributed trainer', default=0)
    parser.add_argument('--trainer', type=str, choices=['basic', 'ddp', 'deepspeed'],
                        default='basic', help='trainer name')
    parser.add_argument('--backend', type=str, default='nccl', help='distributed backend')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config.add_arguments(parser)
    add_task_arguments(parser)
    add_trainer_arguments(parser)
    args = parser.parse_args()

    if not args.test:
        train(args)
    else:
        inference(args)
