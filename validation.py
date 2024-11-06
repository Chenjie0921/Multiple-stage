import torch
import time
import sys

import torch
import torch.distributed as dist

from utils import AverageMeter, calculate_accuracy


def val_epoch(epoch,
              data_loader,
              current_model,
              MLP,
              criterion,
              classifer,
              device,
              logger,
              tb_writer=None,
              distributed=False):
    print('validation at epoch {}'.format(epoch))

    current_model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loss_2s = AverageMeter()
    acc_2s = AverageMeter()
    accuracies = AverageMeter()

    end_time = time.time()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            data_time.update(time.time() - end_time)

            targets = targets.to(device, non_blocking=True)
            outputs,feature = current_model(inputs)
            #FOR MLP
            # for n in range(2, 5):
            #     feature = torch.squeeze(feature, 2)
            # features = MLP(feature)
            #FOR Conv
            for n in range(2, 4):
                feature = torch.squeeze(feature, 2)
            features = MLP(feature)
            features = torch.squeeze(features,2)
            feature = torch.squeeze(feature,2)
            # feature_result = torch.cat((features,feature),1)
            feature_result = features+feature
            outputs_result = classifer(feature_result)
            loss = 0.1*criterion(outputs_result, targets)
            loss_2 = criterion(outputs,targets)
            acc = calculate_accuracy(outputs_result, targets)
            acc_2 = calculate_accuracy(outputs,targets)

            losses.update(loss.item(), inputs.size(0))
            loss_2s.update(loss_2.item(),inputs.size(0))
            accuracies.update(acc, inputs.size(0))
            acc_2s.update(acc_2,inputs.size(0))

            batch_time.update(time.time() - end_time)
            end_time = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss_ronghe {loss.val:.4f} ({loss.avg:.4f})\t Loss {loss_2.val:.4f} ({loss_2.avg:.4f})\t '
                  'Acc_ronghe {acc.val:.3f} ({acc.avg:.3f})\t Acc {acc_2.val:.3f} ({acc_2.avg:.3f})'.format(
                      epoch,
                      i + 1,
                      len(data_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      loss_2 = loss_2s,
                      acc=accuracies,
                      acc_2 = acc_2s))

    if distributed:
        loss_sum = torch.tensor([losses.sum],
                                dtype=torch.float32,
                                device=device)
        loss_count = torch.tensor([losses.count],
                                  dtype=torch.float32,
                                  device=device)
        acc_sum = torch.tensor([accuracies.sum],
                               dtype=torch.float32,
                               device=device)
        acc_count = torch.tensor([accuracies.count],
                                 dtype=torch.float32,
                                 device=device)

        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(loss_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(acc_count, op=dist.ReduceOp.SUM)

        losses.avg = loss_sum.item() / loss_count.item()
        accuracies.avg = acc_sum.item() / acc_count.item()

    if logger is not None:
        logger.log({'epoch': epoch, 'loss_total': losses.avg, 'acc': accuracies.avg})

    if tb_writer is not None:
        tb_writer.add_scalar('val/loss', losses.avg, epoch)
        tb_writer.add_scalar('val/acc', accuracies.avg, epoch)

    return losses.avg
