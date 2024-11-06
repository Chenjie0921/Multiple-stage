import torch
import time
import os
import sys

import torch
import torch.distributed as dist
from torch.optim import SGD, lr_scheduler

from utils import AverageMeter, calculate_accuracy
from loss import GeneratorLoss, DiscriminatorLoss


def train_epoch(epoch,
                data_loader,
                current_model,
                future_model,
                criterion,
                # optimizer,
                # optimizer_1,
                device,
                current_lr,
                epoch_logger,
                batch_logger,
                criterion_L2,
                MLP,
                queryen,
                dis,
                classifer,
                optimizer_2,
                distributed=False,
                tb_writer=None,
                ):
    print('train at epoch {}'.format(epoch)
          )

    current_model.train()
    future_model.train()
    MLP.train()
    dis.train()
    classifer.train()

    # #TODO change the loss
    # criterion_L2 = torch.nn.L1Loss()
    # # criterion_L2 = torch.nn.MSELoss()
    # #TODO change the cengshu of MLP
    # MLP = torch.nn.Sequential(
    #     torch.nn.Linear(in_features=2048, out_features=2048, bias=True),
    #     # torch.nn.ReLU(inplace=True),
    #     # torch.nn.Linear(in_features=2048, out_features=2048, bias=True)
    # )
    # MLP.to(device)
    # classifer = torch.nn.Sequential(
    #     torch.nn.Linear(2048,101)
    # )
    # classifer.to(device)
    # optimizer_2 = SGD([{'params':MLP.parameters()},
    #                    {'params':classifer.parameters()}],lr=0.001,momentum=0.9)
    # scheduler_3 = lr_scheduler.MultiStepLR(optimizer_2,[50, 100, 150])
    # optimizer_3 = SGD(classifer.parameters(),lr=0.01,momentum=0.9)
    # fc1 = torch.nn.Linear(in_features=2048,out_features=2048,bias=True)
    # fc2 = torch.nn.Linear(in_features=2048,out_features=2048,bias=True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_total = AverageMeter()
    loss_1 = AverageMeter()
    loss_2 = AverageMeter()
    loss_3 = AverageMeter()
    loss_4 = AverageMeter()
    loss_D = AverageMeter()
    loss_G = AverageMeter()
    acc_part_1s = AverageMeter()
    acc_part_2s= AverageMeter()
    acc_ronghes = AverageMeter()

    end_time = time.time()
    for i, (inputs, inputs_1, inputs_2, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)

        targets = targets.to(device, non_blocking=True)
        outputs, feature = current_model(inputs)   #[1,ratio]
        outputs_1, feature_1 = current_model(inputs_1)   #[ratio,middle]
        outputs_2, feature_2 = current_model(inputs_2)   #[middle,end]
        #ronghe 20%and80%
        #TODO change the MLP
        # for n in range(2, 5):
        #     feature = torch.squeeze(feature, 2)
        #     feature_1 = torch.squeeze(feature_1,2)
        # output_future = MLP(feature)
        #TODO FOR Conv
        # for n in range(2, 4):
        #     feature = torch.squeeze(feature, 2)
        #     feature_1 = torch.squeeze(feature_1,2)
        # output_future = MLP(feature)
        # output_future = torch.squeeze(output_future,2)
        # feature = torch.squeeze(feature,2)
        # feature_1 = torch.squeeze(feature_1,2)
        #TODO FOR Transformer
        for n in range(2, 5):
            feature = torch.squeeze(feature, 2)  # B * N
            feature_1 = torch.squeeze(feature_1,2)
            feature_2 = torch.squeeze(feature_2, 2)
        feature = torch.unsqueeze(feature, 1)  # B * 1 * N
        feature_2 = torch.unsqueeze(feature_2, 1)
        #stage_1
        output_future = MLP(feature)  # B * 1 * N  [4,2,2048]
        # output_future = torch.squeeze(output_future, 1) # B * N
        part_1 = output_future[:, 0:1, :]# query 1 shengchengdetezhen  [4,1,2048]
        part_2 = output_future[:, 1:2, :]# query 2  [4,1,2048]
        feature = torch.squeeze(feature, 1)# B * N
        feature_1 = torch.unsqueeze(feature_1, 1)
        # part_1 = torch.unsqueeze(part_1, 1)
        real_output = dis(feature_1)  # B * 1 * N
        fake_output = dis(part_1)
        real_output = torch.squeeze(real_output, 1)  # B * N
        fake_output = torch.squeeze(fake_output, 1)
        loss_g_1 = GeneratorLoss()(fake_output)
        loss_d_1 = DiscriminatorLoss()(real_output, fake_output)

        #stage_2
        # feature_stage = feature + output_future
        # output_future_2 = MLP(feature_stage)
        # part_2 = torch.unsqueeze(part_2, 1)
        real_output_1 = dis(feature_2)  # B * 1 * N
        fake_output_1 = dis(part_2)
        real_output_1 = torch.squeeze(real_output_1, 1)  # B * N
        fake_output_1 = torch.squeeze(fake_output_1, 1)
        feature = torch.squeeze(feature, 1)  # B * N
        loss_g_2 = GeneratorLoss()(fake_output_1)
        loss_d_2 = DiscriminatorLoss()(real_output_1, fake_output_1)
        loss_g = loss_g_1 + loss_g_2
        loss_d = loss_d_1 + loss_d_2
        #TODO change the way of fusion
        # outputs_ronghe = torch.cat((feature,output_future),1)
        # output_future = torch.squeeze(output_future, 1)
        feature_1 = torch.squeeze(feature_1, 1)
        feature_2 = torch.squeeze(feature_2, 1)
        part_1 = torch.squeeze(part_1, 1)
        part_2 = torch.squeeze(part_2, 1)
        # output_future_2 = torch.squeeze(output_future_2, 1)
        outputs_ronghe = feature + part_1 + part_2
        outputs_ronghe = queryen(outputs_ronghe)
        # outputs_ronghe = output_future

        #classifer
        # outputs_ronghe = torch.unsqueeze(outputs_ronghe, 1)
        outputs_ronghe_fc = classifer(outputs_ronghe)
        outputs_part_1 = classifer(feature_1)
        outputs_part_2 = classifer(feature_2)
        # outputs_ronghe_fc = torch.squeeze(outputs_ronghe_fc, 1)
        # loss_current = criterion(outputs, targets)
        # loss_future = criterion(outputs_1, targets)
        # loss_feature_1 = criterion_L2(feature_1, part_1)
        # loss_feture_2 = criterion_L2(feature_2, part_2)
        # loss_feature = loss_feature_1 + loss_feture_2
        loss_part_1 = 1 * criterion(outputs_part_1, targets)
        loss_part_2 = 1 * criterion(outputs_part_2, targets)
        loss_ronghe = 1*criterion(outputs_ronghe_fc, targets)
        loss_cls = loss_part_1 + loss_part_2 + loss_ronghe
        acc_ronghe = calculate_accuracy(outputs_ronghe_fc,targets)
        acc_part_1 = calculate_accuracy(outputs_part_1, targets)
        acc_part_2 = calculate_accuracy(outputs_part_2,targets)

        # loss_1.update(loss_current.item(), inputs.size(0))
        # loss_2.update(loss_future.item(),inputs.size(0))
        # loss_3.update(loss_feature.item(),inputs.size(0))
        loss_4.update(loss_cls.item(),inputs.size(0))
        loss_G.update(loss_g.item(), inputs.size(0))
        loss_D.update(loss_d.item(), inputs.size(0))
        loss_total.update(loss_cls.item() + loss_g.item() + loss_d.item(), inputs.size(0))
        acc_part_1s.update(acc_part_1, inputs.size(0))
        acc_part_2s.update(acc_part_2, inputs.size(0))
        acc_ronghes.update(acc_ronghe, inputs.size(0))

        #TODO ablation
        # loss_totals = loss_current+loss_future+loss_feature+loss_ronghe
        # loss_totals = loss_current+loss_feature+loss_ronghe
        loss_totals = loss_cls + loss_d + loss_g
        # optimizer.zero_grad()
        # optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        loss_totals.backward()
        # optimizer.step()
        # optimizer_1.step()
        optimizer_2.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if batch_logger is not None:
            batch_logger.log({
                'epoch': epoch,
                'batch': i + 1,
                'iter': (epoch - 1) * len(data_loader) + (i + 1),
                'loss_total': loss_total.val,
                # 'loss_current': loss_1.val,
                # 'loss_future': loss_2.val,
                # 'loss_feature': loss_3.val,
                'loss_cls': loss_4.val,
                'loss_d': loss_D.val,
                'loss_g': loss_G.val,
                'acc_part_1': acc_part_1s.val,
                'acc_part_2': acc_part_2s.val,
                'acc_ronghe': acc_ronghes.val,
                'lr': current_lr
            })

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss_total {loss.val:.4f} ({loss.avg:.4f}) = loss_cls {loss_cls.val:.4f}({loss_cls.avg:.4f})+loss_d {loss_d.val:.4f}({loss_d.avg:.4f})+loss_g {loss_g.val:.4f}({loss_g.avg:.4f})\t'
              'Acc_ronghe {acc_ronghe.val:.3f} ({acc_ronghe.avg:.3f})\t'
              'Acc_part_1 {acc_part_1.val:.3f} ({acc_part_1.avg:.3f})\t'
              'Acc_part_2 {acc_part_2.val:.3f} ({acc_part_2.avg:.3f})'.format(epoch,
                                                         i + 1,
                                                         len(data_loader),
                                                         batch_time=batch_time,
                                                         data_time=data_time,
                                                         loss=loss_total,
                                                         # loss_current=loss_1,
                                                         # loss_future=loss_2,
                                                         # loss_feature=loss_3,
                                                         loss_cls=loss_4,
                                                         loss_d=loss_D,
                                                         loss_g=loss_G,
                                                         acc_ronghe=acc_ronghes,
                                                         acc_part_1=acc_part_1s,
                                                         acc_part_2=acc_part_2s))


    if epoch_logger is not None:
        epoch_logger.log({
            'epoch': epoch,
            'loss_total': loss_total.avg,
            'loss_cls': loss_4.avg,
            'loss_g': loss_G.avg,
            'loss_d': loss_D.avg,
            'acc_part_1': acc_part_1s.avg,
            'acc_part_2': acc_part_2s.avg,
            'acc_ronghe': acc_ronghes.avg,
            'lr': current_lr
        })

    if tb_writer is not None:
        tb_writer.add_scalar('train/loss', loss_total.avg, epoch)
        tb_writer.add_scalar('train/acc', acc_ronghes.avg, epoch)
        tb_writer.add_scalar('train/lr', current_lr, epoch)

