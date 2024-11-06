from pathlib import Path

import json
import random
import os

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.backends import cudnn
import torchvision

from opts import parse_opts
from network import U_Net
from model import (generate_model, load_pretrained_model, make_data_parallel,
                   get_fine_tuning_parameters)
from mean import get_mean_std
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)
from temporal_transforms import (LoopPadding, TemporalRandomCrop,
                                 TemporalCenterCrop, TemporalEvenCrop,
                                 SlidingWindow, TemporalSubsampling)
from temporal_transforms import Compose as TemporalCompose
from dataset import get_training_data, get_validation_data, get_inference_data
from utils import Logger, worker_init_fn, get_lr
from training import train_epoch
from validation import val_epoch
import inference
from util_scripts.eval_accuracy import *
from models.Decorder import decoder_fuser
from models.Encoder import encoder_fuser
from models.transformer import TransformerEncoder, TransformerDecoder
from models.detr.models.transformer import TransformerEncoder, TransformerDecoder
from models.vit.timesformer.models.vit import TimeSformer


def json_serial(obj):
    if isinstance(obj, Path):
        return str(obj)


def get_opt():
    opt = parse_opts()

    if opt.root_path is not None:
        opt.video_path = opt.root_path / opt.video_path
        opt.annotation_path = opt.root_path / opt.annotation_path
        opt.result_path = opt.root_path / opt.result_path
        if opt.resume_path is not None:
            opt.resume_path = opt.root_path / opt.resume_path
        if opt.pretrain_path is not None:
            opt.pretrain_path = opt.root_path / opt.pretrain_path

    if opt.pretrain_path is not None:
        opt.n_finetune_classes = opt.n_classes
        opt.n_classes = opt.n_pretrain_classes

    if opt.output_topk <= 0:
        opt.output_topk = opt.n_classes

    if opt.inference_batch_size == 0:
        opt.inference_batch_size = opt.batch_size

    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)
    opt.begin_epoch = 1
    opt.mean, opt.std = get_mean_std(opt.value_scale, dataset=opt.mean_dataset)
    opt.n_input_channels = 3
    if opt.input_type == 'flow':
        opt.n_input_channels = 2
        opt.mean = opt.mean[:2]
        opt.std = opt.std[:2]

    if opt.distributed:
        opt.dist_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        # opt.dist_rank = int(os.environ["PMI_RANK"])
        if opt.dist_rank == 0:
            print(opt)
            with (opt.result_path / 'opts.json').open('w') as opt_file:
                json.dump(vars(opt), opt_file, default=json_serial)
    else:
        print(opt)
        with (opt.result_path / 'opts.json').open('w') as opt_file:
            json.dump(vars(opt), opt_file, default=json_serial)

    return opt


def resume_model(resume_path, arch, model):
    print('loading checkpoint {} model'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')
    assert arch == checkpoint['arch']

    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model


def resume_train_utils(resume_path, begin_epoch, optimizer, scheduler):
    print('loading checkpoint {} train utils'.format(resume_path))
    checkpoint = torch.load(resume_path, map_location='cpu')

    begin_epoch = checkpoint['epoch'] + 1
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return begin_epoch, optimizer, scheduler


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_utils(opt, ratio):
    assert opt.train_crop in ['random', 'corner', 'center']
    spatial_transform = []
    if opt.train_crop == 'random':
        spatial_transform.append(
            RandomResizedCrop(
                opt.sample_size, (opt.train_crop_min_scale, 1.0),
                (opt.train_crop_min_ratio, 1.0 / opt.train_crop_min_ratio)))
    elif opt.train_crop == 'corner':
        scales = [1.0]
        scale_step = 1 / (2 ** (1 / 4))
        for _ in range(1, 5):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(opt.sample_size, scales))
    elif opt.train_crop == 'center':
        spatial_transform.append(Resize(opt.sample_size))
        spatial_transform.append(CenterCrop(opt.sample_size))
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    if not opt.no_hflip:
        spatial_transform.append(RandomHorizontalFlip())
    if opt.colorjitter:
        spatial_transform.append(ColorJitter())
    spatial_transform.append(ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.append(ScaleValue(opt.value_scale))
    spatial_transform.append(normalize)
    spatial_transform = Compose(spatial_transform)

    assert opt.train_t_crop in ['random', 'center']
    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    if opt.train_t_crop == 'random':
        temporal_transform.append(TemporalRandomCrop(opt.sample_duration))
    elif opt.train_t_crop == 'center':
        temporal_transform.append(TemporalCenterCrop(opt.sample_duration))
    temporal_transform = TemporalCompose(temporal_transform)
    train_data = get_training_data(video_path=opt.video_path,
                                   annotation_path=opt.annotation_path,
                                   dataset_name=opt.dataset,
                                   input_type=opt.input_type,
                                   file_type=opt.file_type,
                                   spatial_transform=spatial_transform,
                                   temporal_transform=temporal_transform,
                                   target_transform=None,
                                   ratio=ratio)
    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=opt.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=opt.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler,
                                               worker_init_fn=worker_init_fn)

    if opt.is_master_node:
        train_logger = Logger(opt.result_path / 'train.log',
                              ['epoch', 'loss_total', 'loss_cls', 'loss_g', 'loss_d', 'acc_part_1', 'acc_part_2', 'acc_ronghe' ,'lr'])
        acc_logger = Logger(opt.result_path / 'acc.log', ['epoch', 'acc_epoch_test'])
        train_batch_logger = Logger(
            opt.result_path / 'train_batch.log',
            ['epoch', 'batch', 'iter', 'loss_total', 'loss_cls', 'loss_d', 'loss_g', 'acc_part_1', 'acc_part_2', 'acc_ronghe', 'lr'])
    else:
        train_logger = None
        train_batch_logger = None
        acc_logger = None

    if opt.nesterov:
        dampening = 0
    else:
        dampening = opt.dampening
    # optimizer = torch.optim.AdamW(model_parameters,
    #                 lr=opt.learning_rate,
    #                 # momentum=opt.momentum,
    #                 # dampening=dampening,
    #                 weight_decay=opt.weight_decay,
    #                 # nesterov=opt.nesterov
    #                               )
    # optimizer_1 = torch.optim.AdamW(parameters_feature,
    #                   lr=opt.learning_rate,
    #                   # momentum=opt.momentum,
    #                   # dampening=dampening,
    #                   weight_decay=opt.weight_decay,
    #                   # nesterov=opt.nesterov
    #                                 )
    assert opt.lr_scheduler in ['plateau', 'multistep']
    assert not (opt.lr_scheduler == 'plateau' and opt.no_val)
    # if opt.lr_scheduler == 'plateau':
    #     scheduler = lr_scheduler.ReduceLROnPlateau(
    #         optimizer, 'min', patience=opt.plateau_patience)
    #     scheduler_1 = lr_scheduler.ReduceLROnPlateau(
    #         optimizer_1, 'min', patience=opt.plateau_patience)
    # else:
    #     scheduler = lr_scheduler.MultiStepLR(optimizer,
    #                                          opt.multistep_milestones, gamma=0.1)
    #     scheduler_1 = lr_scheduler.MultiStepLR(optimizer_1,
    #                                            opt.multistep_milestones, gamma=0.1)

    # return (train_loader, train_sampler, train_logger, train_batch_logger,
    #         optimizer, optimizer_1, scheduler, scheduler_1, acc_logger)
    return train_loader, train_sampler, train_logger, train_batch_logger, acc_logger


def get_val_utils(opt, ratio):
    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)
    spatial_transform = [
        Resize(opt.sample_size),
        CenterCrop(opt.sample_size),
        ToTensor()
    ]
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        TemporalEvenCrop(opt.sample_duration, opt.n_val_samples))
    temporal_transform = TemporalCompose(temporal_transform)

    val_data, collate_fn = get_validation_data(
        video_path=opt.video_path,
        annotation_path=opt.annotation_path,
        dataset_name=opt.dataset,
        input_type=opt.input_type,
        file_type=opt.file_type,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        ratio=ratio,
        target_transform=None)
    if opt.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_data, shuffle=False)
    else:
        val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=(opt.batch_size //
                                                         opt.n_val_samples),
                                             shuffle=False,
                                             num_workers=opt.n_threads,
                                             pin_memory=True,
                                             sampler=val_sampler,
                                             worker_init_fn=worker_init_fn,
                                             collate_fn=collate_fn)

    if opt.is_master_node:
        val_logger = Logger(opt.result_path / 'val.log',
                            ['epoch', 'loss_total', 'acc'])
    else:
        val_logger = None

    return val_loader, val_logger


def get_inference_utils(opt, ratio):
    assert opt.inference_crop in ['center', 'nocrop']

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm,
                                     opt.no_std_norm)

    spatial_transform = [Resize(opt.sample_size)]
    if opt.inference_crop == 'center':
        spatial_transform.append(CenterCrop(opt.sample_size))
    spatial_transform.append(ToTensor())
    if opt.input_type == 'flow':
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(
        SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform)

    # inference_data, collate_fn = get_inference_data(
    #     opt.video_path, opt.annotation_path, opt.dataset, opt.input_type,
    #     opt.file_type, opt.inference_subset, spatial_transform,
    #     temporal_transform)
    inference_data, collate_fn = get_inference_data(
        video_path=opt.video_path,
        annotation_path=opt.annotation_path,
        dataset_name=opt.dataset,
        input_type=opt.input_type,
        file_type=opt.file_type,
        inference_subset=opt.inference_subset,
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        ratio=ratio,
        target_transform=None
    )

    inference_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=opt.inference_batch_size,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=collate_fn)

    return inference_loader, inference_data.class_names


def save_checkpoint(save_file_path, epoch, arch, model, optimizer, scheduler):
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    save_states = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(save_states, save_file_path)


def main_worker(index, opt):
    random.seed(opt.manual_seed)
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)

    if index >= 0 and opt.device.type == 'cuda':
        opt.device = torch.device(f'cuda:{index}')

    if opt.distributed:
        opt.dist_rank = opt.dist_rank * opt.ngpus_per_node + index
        dist.init_process_group(backend='nccl',
                                init_method=opt.dist_url,
                                world_size=opt.world_size,
                                rank=opt.dist_rank)
        opt.batch_size = int(opt.batch_size / opt.ngpus_per_node)
        opt.n_threads = int(
            (opt.n_threads + opt.ngpus_per_node - 1) / opt.ngpus_per_node)
    opt.is_master_node = not opt.distributed or opt.dist_rank == 0

    current_model = generate_model(opt)
    feature_model = generate_model(opt)
    # current_model = TimeSformer(img_size=112, num_classes=768, num_frames=16, attention_type="space_only", pretrained_model= "/media/cowinrio/datafile/code/11/3D-ResNets-PyTorch/models/vit/weight/TimeSformer_divST_96x4_224_K600.pyth")
    # feature_model = TimeSformer(img_size=112, num_classes=768, num_frames=16, attention_type="space_only", pretrained_model= "/media/cowinrio/datafile/code/11/3D-ResNets-PyTorch/models/vit/weight/TimeSformer_divST_96x4_224_K600.pyth")
    current_model = make_data_parallel(current_model, opt.distributed, opt.device)
    feature_model = make_data_parallel(feature_model, opt.distributed, opt.device)
    if opt.batchnorm_sync:
        assert opt.distributed, 'SyncBatchNorm only supports DistributedDataParallel.'
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if opt.pretrain_path:
        current_model = load_pretrained_model(current_model, opt.pretrain_path, opt.model,
                                              opt.n_finetune_classes)
        feature_model = load_pretrained_model(feature_model, opt.pretrain_path, opt.model,
                                              opt.n_finetune_classes)
    if opt.resume_path is not None:
        model = resume_model(opt.resume_path, opt.arch, model)
    # model = make_data_parallel(model, opt.distributed, opt.device)

    if opt.pretrain_path:
        parameters_current = get_fine_tuning_parameters(current_model, opt.ft_begin_module)
        # parameters_feature = get_fine_tuning_parameters(feature_model, opt.ft_begin_module)
    else:
        parameters_current = current_model.parameters()
        # parameters_feature = feature_model.parameters()

    # parameters_current = current_model.parameters()
    # parameters_feature = feature_model.parameters()
    if opt.is_master_node:
        print(current_model)

    criterion = CrossEntropyLoss().to(opt.device)

    if not opt.no_train:
        (train_loader, train_sampler, train_logger, train_batch_logger, acc_logger) = get_train_utils(opt=opt,
                                                                                       # model_parameters=parameters_current,
                                                                                       # parameters_feature=parameters_feature,
                                                                                       ratio=opt.ratio_current)
        # (train_loader_1, train_sampler_1, train_logger_1, train_batch_logger_1,
        #  optimizer_1, scheduler_1) = get_train_utils(opt, parameters_feature, opt.ratio_future)
        # if opt.resume_path is not None:
        #     opt.begin_epoch, optimizer, scheduler = resume_train_utils(
        #         opt.resume_path, opt.begin_epoch, optimizer, scheduler)
        #     if opt.overwrite_milestones:
        #         scheduler.milestones = opt.multistep_milestones
    if not opt.no_val:
        val_loader, val_logger = get_val_utils(opt=opt, ratio=opt.ratio_current)

    if opt.tensorboard and opt.is_master_node:
        from torch.utils.tensorboard import SummaryWriter
        if opt.begin_epoch == 1:
            tb_writer = SummaryWriter(log_dir=opt.result_path)
        else:
            tb_writer = SummaryWriter(log_dir=opt.result_path,
                                      purge_step=opt.begin_epoch)
    else:
        tb_writer = None

    prev_val_loss = None
    # TODO change the loss
    criterion_L2 = torch.nn.L1Loss()
    # criterion_L2 = torch.nn.MSELoss()
    # TODO change the cengshu of MLP
    dis = torch.nn.Sequential(

        # torch.nn.TransformerEncoderLayer(d_model=2048, nhead=8),
        encoder_fuser(in_dim=2048, dim=256, num_heads=8, num_layers=1),
        torch.nn.Linear(in_features=2048, out_features=1, bias=True),
        # torch.nn.ReLU(inplace=True),
        # torch.nn.Linear(in_features=2048, out_features=2048, bias=True)
    )
    dis.to(opt.device)
    # FOR Conv
    # MLP = U_Net(img_ch=2048,output_ch=2048)
    # MLP.to(opt.device)
    # FOR Transformer
    # MLP = torch.nn.TransformerEncoderLayer(d_model=2048,nhead=8)
    MLP = decoder_fuser(in_dim=2048, dim=256, num_heads=8, num_layers=3)
    # MLP = TransformerDecoder(num_layers=3)
    # MLP = torch.nn.Sequential(
    #     torch.nn.Linear(2048,2048)
    # )
    MLP.to(opt.device)

    querycoding = encoder_fuser(in_dim=2048, dim=2048, num_heads=8, num_layers=1)
    querycoding.to(opt.device)

    classifer = torch.nn.Sequential(
        # encoder_fuser(dim=2048, num_heads=8, num_layers=1),
        torch.nn.Linear(2048, 101),
        # torch.nn.ReLU(),
        # torch.nn.Linear(1024, 51),

    )
    classifer.to(opt.device)

    if opt.val:
        MLP = load_pretrained_model(MLP, opt.MLP_path, opt.model, opt.n_finetune_classes)
        classifer = load_pretrained_model(classifer, opt.cls_path, opt.model, opt.n_finetune_classes)
    optimizer_2 = torch.optim.AdamW([{'params': parameters_current},
                                     {'params': MLP.parameters()},
                                     {'params': dis.parameters()},
                                     {'params': classifer.parameters()}], lr=opt.learning_rate, weight_decay=opt.weight_decay)
    # scheduler_3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2, mode='max')
    scheduler_3 = lr_scheduler.MultiStepLR(optimizer_2, [15, 25, 35], gamma=0.1)
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        # TODO some
        # i = 135+i
        if not opt.no_train:
            if opt.distributed:
                train_sampler.set_epoch(i)
            current_lr = get_lr(optimizer_2)
            # TODO train_1,train:MLP,classifer,optimizer_2,scheduler_3 =
            train_epoch(
                epoch=i,
                data_loader=train_loader,
                current_model=current_model,
                future_model=feature_model,
                criterion=criterion,
                # optimizer=optimizer,
                # optimizer_1=optimizer_1,
                device=opt.device,
                current_lr=current_lr,
                epoch_logger=train_logger,
                batch_logger=train_batch_logger,
                tb_writer=tb_writer,
                distributed=opt.distributed,
                criterion_L2=criterion_L2,
                MLP=MLP,
                queryen=querycoding,
                dis=dis,
                classifer=classifer,
                optimizer_2=optimizer_2
            )
            # train_epoch(i, train_loader, current_model, feature_model,criterion, optimizer,
            #             opt.device, current_lr, train_logger,
            #             train_batch_logger, tb_writer, opt.distributed)
            # feature_future = train_epoch(i, train_loader_1, feature_model, criterion, optimizer_1,
            #             opt.device, current_lr, train_logger_1,
            #             train_batch_logger_1, tb_writer, opt.distributed)
            if i > -1:
                if opt.inference:
                    inference_loader, inference_class_names = get_inference_utils(opt, ratio=opt.ratio_current)
                    inference_result_path = opt.result_path / '{}_{}.json'.format(
                        opt.inference_subset, i)

                    inference.inference(inference_loader, current_model, MLP, querycoding, classifer, inference_result_path,
                                        inference_class_names, opt.inference_no_average,
                                        opt.output_topk)
                    accuracys = evaluate(
                        ground_truth_path=opt.ground_truth_path,
                        result_path=opt.result_path / 'val_{}.json'.format(i),
                        subset='validation',
                        top_k=1,
                        ignore=True)
                    if acc_logger is not None:
                        acc_logger.log(
                            {
                                'epoch': i,
                                'acc_epoch_test': accuracys
                            }
                        )
            if i % opt.checkpoint == 0 and opt.is_master_node and i > -1:
                save_file_path = opt.result_path / 'save_{}.pth'.format(i)
                save_checkpoint(save_file_path, i, opt.arch, current_model, optimizer_2,
                                scheduler_3)
                # save_file_path_4 = opt.result_path / 'save_future_{}.pth'.format(i)
                # save_checkpoint(save_file_path_4, i, opt.arch, feature_model, optimizer_1, scheduler_1)
                save_file_path_2 = opt.result_path / 'save_Trans_{}.pth'.format(i)
                save_checkpoint(save_file_path_2, i, 'MLP', MLP, optimizer_2, scheduler_3)
                save_file_path_3 = opt.result_path / 'save_cls_{}.pth'.format(i)
                save_checkpoint(save_file_path_3, i, 'cls', classifer, optimizer_2, scheduler_3)

        if not opt.no_val:
            prev_val_loss = val_epoch(
                epoch=i,
                data_loader=val_loader,
                current_model=current_model,
                MLP=MLP,
                criterion=criterion,
                classifer=classifer,
                device=opt.device,
                logger=val_logger,
                tb_writer=tb_writer,
                distributed=opt.distributed
            )
            # prev_val_loss = val_epoch(i, val_loader, model, criterion,
            #                           opt.device, val_logger, tb_writer,
            #                           opt.distributed)

        if not opt.no_train and opt.lr_scheduler == 'multistep':
            # scheduler.step()
            # scheduler_1.step()
            scheduler_3.step()
        elif not opt.no_train and opt.lr_scheduler == 'plateau':
            scheduler_3.step(prev_val_loss)

    # if opt.inference:
    #     inference_loader, inference_class_names = get_inference_utils(opt,ratio = opt.ratio_current)
    #     inference_result_path = opt.result_path / '{}.json'.format(
    #         opt.inference_subset)
    #
    #     inference.inference(inference_loader, current_model,MLP,classifer, inference_result_path,
    #                         inference_class_names, opt.inference_no_average,
    #                         opt.output_topk)


if __name__ == '__main__':
    opt = get_opt()

    opt.device = torch.device('cpu' if opt.no_cuda else 'cuda')
    if not opt.no_cuda:
        cudnn.benchmark = True
    if opt.accimage:
        torchvision.set_image_backend('accimage')

    opt.ngpus_per_node = torch.cuda.device_count()
    if opt.distributed:
        opt.world_size = opt.ngpus_per_node * opt.world_size
        mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt,))
    else:
        main_worker(-1, opt)
