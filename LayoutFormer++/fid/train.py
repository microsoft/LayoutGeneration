import argparse
import os
import os.path as osp
import shutil
import copy

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from data import RicoDataset, PubLayNetDataset, transforms
from utils import utils, os_utils
from config import add_arguments
from fid_model import LayoutNet


class RicoFidDataset(RicoDataset):

    def __init__(self, args, split: str, online_process: bool = True):
        self.args = args

        transform_functions = [
            transforms.AddGaussianNoise(
                mean=self.args.gaussian_noise_mean,
                std=self.args.gaussian_noise_std,
                bernoulli_beta=self.args.train_bernoulli_beta),
            transforms.LexicographicSort(),
            transforms.CoordinateTransform('ltrb'),
        ]
        self.transform = T.Compose(transform_functions)

        super().__init__(root=args.data_dir,
                         split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)

    def process(self, _data):
        data = copy.deepcopy(_data)
        data = self.transform(data)
        return data


class PubLayNetFidDataset(PubLayNetDataset):

    def __init__(self, args, split: str, online_process: bool = True):
        self.args = args

        transform_functions = [
            transforms.AddGaussianNoise(
                mean=self.args.gaussian_noise_mean,
                std=self.args.gaussian_noise_std,
                bernoulli_beta=self.args.train_bernoulli_beta),
            transforms.LexicographicSort(),
            transforms.CoordinateTransform('ltrb'),
        ]
        self.transform = T.Compose(transform_functions)

        super().__init__(root=args.data_dir,
                         split=split,
                         max_num_elements=args.max_num_elements,
                         online_process=online_process)

    def process(self, _data):
        data = copy.deepcopy(_data)
        data = self.transform(data)
        return data


def cal_loss(args, gold_bboxes, gold_labels, mask, disc, logit_cls, pred_bboxes,
             r_or_f):

    if r_or_f:
        loss_D = F.softplus(-disc).mean()
    else:
        loss_D = F.softplus(disc).mean()

    loss_recl = F.cross_entropy(logit_cls, gold_labels[mask].reshape(-1))
    loss_recb = args.box_loss_weight * F.mse_loss(pred_bboxes, gold_bboxes[mask])

    return loss_recb, loss_recl, loss_D


def disc_acc(D_real, D_fake):
    total = len(D_real) + len(D_fake)
    correct = 0

    for d in D_real:
        if d > 0:
            correct += 1
    for d in D_fake:
        if d < 0:
            correct += 1
    return correct, total


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    args = parser.parse_args()

    wandb.init(project=f'{args.dataset}-fid')
    os_utils.makedirs(osp.join(args.out_dir, args.dataset))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = LayoutNet(num_label=args.num_label,
                      max_bbox=args.max_num_elements)
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs!")
        args.batch_size *= torch.cuda.device_count()
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=100, verbose=True, threshold=1e-5, threshold_mode='rel', cooldown=100, min_lr=0, eps=1e-08)

    if args.dataset == 'rico':
        train_dataset = RicoFidDataset(args, 'train')
        val_dataset = RicoFidDataset(args, 'val')
    elif args.dataset == 'publaynet':
        train_dataset = PubLayNetFidDataset(args, 'train')
        val_dataset = PubLayNetFidDataset(args, 'val')

    else:
        raise NotImplementedError

    print(f'train data: {len(train_dataset)}')
    print(f'val data: {len(val_dataset)}')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              collate_fn=utils.collate_fn,
                              drop_last=True,
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.batch_size,
                            collate_fn=utils.collate_fn,
                            drop_last=True,
                            shuffle=True)

    global_step = 0
    best_measure = 1e-8
    for epoch in range(args.epoch):
        epoch_correct, epoch_total = 0, 0
        model.train()
        for batch_idx, data in enumerate(train_loader):
            labels, mask = utils.to_dense_batch(data['labels'])
            fake_bboxes, _ = utils.to_dense_batch(data['bboxes'])
            real_bboxes, _ = utils.to_dense_batch(data['gold_bboxes'])

            labels = labels.to(device)
            mask = mask.to(device)
            fake_bboxes = fake_bboxes.to(device)
            real_bboxes = real_bboxes.to(device)
            padding_mask = ~mask

            real_logit_disc, real_logit_cls, real_bbox_pred = model(
                real_bboxes, labels, padding_mask)
            fake_logit_disc, fake_logit_cls, fake_bbox_pred = model(
                fake_bboxes, labels, padding_mask)

            real_box_loss, real_label_loss, real_disc_loss = cal_loss(
                args, real_bboxes, labels, mask, real_logit_disc, real_logit_cls,
                real_bbox_pred, True)
            fake_box_loss, fake_label_loss, fake_disc_loss = cal_loss(
                args, fake_bboxes, labels, mask, fake_logit_disc, fake_logit_cls,
                fake_bbox_pred, False)

            box_loss = real_box_loss + fake_box_loss
            label_loss = real_label_loss + fake_label_loss
            disc_loss = real_disc_loss + fake_disc_loss

            batch_correct, batch_total = disc_acc(real_logit_disc, fake_logit_disc)
            acc = batch_correct / batch_total
            epoch_correct += batch_correct
            epoch_total += batch_total

            loss = box_loss + label_loss + disc_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step(loss)

            if batch_idx % args.train_log_step == 0:
                wandb.log({
                    'batch_loss': loss,
                    'box': box_loss,
                    'label': label_loss,
                    'disc': disc_loss,
                    'acc': acc,
                    },
                    step=global_step)
                print(f'[{epoch}/{args.epoch}][{batch_idx}/{len(train_loader)}]')
                print('batch_loss:', loss.item(), 'box:', box_loss.item(), 'label:', label_loss.item(),
                      'disc:', disc_loss.item(), 'acc:', acc, 'lr:',
                      str(optimizer.state_dict()['param_groups'][0]['lr']))
            global_step += 1

        train_acc = epoch_correct / epoch_total
        wandb.log({
            'train_acc': train_acc
            },
            step=global_step)
        print('train_acc:', train_acc)

        normal_ckpt_path = osp.join(args.out_dir, args.dataset, 'checkpoint.pth.tar')
        torch.save(model.state_dict(), normal_ckpt_path)

        model.eval()
        with torch.no_grad():
            val_correct, val_total = 0, 0
            for batch_idx, data in enumerate(val_loader):
                labels, mask = utils.to_dense_batch(data['labels'])
                fake_bboxes, _ = utils.to_dense_batch(data['bboxes'])
                real_bboxes, _ = utils.to_dense_batch(data['gold_bboxes'])

                labels = labels.to(device)
                mask = mask.to(device)
                fake_bboxes = fake_bboxes.to(device)
                real_bboxes = real_bboxes.to(device)
                padding_mask = ~mask

                real_logit_disc, real_logit_cls, real_bbox_pred = model(
                    real_bboxes, labels, padding_mask)
                fake_logit_disc, fake_logit_cls, fake_bbox_pred = model(
                    fake_bboxes, labels, padding_mask)

                batch_correct, batch_total = disc_acc(real_logit_disc, fake_logit_disc)
                val_correct += batch_correct
                val_total += batch_total

            val_acc = val_correct / val_total
            wandb.log({
                'val_acc': val_acc
                },
                step=global_step)
            print('val_acc:', val_acc)

        is_best = False
        measure = val_acc
        if best_measure < measure:
            is_best = True
            best_measure = measure

        if is_best:
            best_ckpt_path = osp.join(args.out_dir, args.dataset, 'model_best.pth.tar')
            if os.path.exists(best_ckpt_path):
                # Only keep on best checkpoint (Save memory)
                os.remove(best_ckpt_path)
            shutil.copy(normal_ckpt_path, best_ckpt_path)

        if epoch != args.epoch - 1:
            os.remove(normal_ckpt_path)


if __name__ == '__main__':
    main()
