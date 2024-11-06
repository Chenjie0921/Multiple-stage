import argparse
from pathlib import Path


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path',
                        default='/media/cj/新加卷/datasets/UCF101',
                        type=Path,
                        help='Root directory path')
    parser.add_argument('--video_path',
                        # default='ucf101_videos/jpg/ucf_101frame',
                        # default='/media/cowinrio/datafile/3D-ResNets-PyTorch/data/hmdb51/jpg/HMDB51_frame',
                        # default='/media/cowinrio/datafile/Dataset/HMDB51/opticalflow',
                        default='ucf_101frame',
                        type=Path,
                        help='Directory path of videos')
    parser.add_argument('--annotation_path',
                        # default='hmdb51_3.json',
                        default='ucf101_01.json',
                        type=Path,
                        help='Annotation file path')
    parser.add_argument('--result_path',
                        # default='/media/cowinrio/datafile/code/11/3D-ResNets-PyTorch/result/hmdb51/dis(encoder) gen(decoder)/50%',
                        # default='/media/cowinrio/datafile/code/11/3D-ResNets-PyTorch/result/ucf101/de=5 en=5/split 3/50%',
                        # default='/media/cowinrio/datafile/code/11/3D-ResNets-PyTorch/result/ucf101/de=3 en=5 back_lr=0.002 model_lr=0.001/cls=encoder+mlp',
                        default='/home/cj/cj_code/stage_2/result',
                        type=Path,
                        help='Result directory path')
    parser.add_argument(
        '--dataset',
        default='ucf101',
        type=str,
        help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    parser.add_argument(
        '--n_classes',
        default=101,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400 or 600, ucf101: 101, hmdb51: 51)'
    )
    parser.add_argument('--n_pretrain_classes',
                        default=101,
                        type=int,
                        help=('Number of classes of pretraining task.'
                              'When using --pretrain_path, this must be set.'))
    parser.add_argument('--pretrain_path',
                        # default ='results_resnext101_hmdb51/save_150.pth',
                        default='/media/cj/新加卷/datasets/UCF101/results_resnext101_ucf101/resnext-101-kinetics-ucf101_split1.pth',
                        # default = 'results/resnext-101-kinetics.pth',
                        # default='results_resnext101_hmdb51/resnext-101-kinetics-hmdb51_split3.pth',
                        # default='/media/cowinrio/datafile/code/11/3D-ResNets-PyTorch/models/vit/weight/TimeSformer_divST_96x4_224_K600.pyth',
                        type=Path,
                        help='Pretrained model path (.pth).')
    parser.add_argument(
        '--ft_begin_module',
        default='',
        type=str,
        help=('Module name of beginning of fine-tuning'
              '(conv1, layer1, fc, denseblock1, classifier, ...).'
              'The default means all layers are fine-tuned.'))
    parser.add_argument('--sample_size',
                        default=224,
                        type=int,
                        help='Height and width of inputs')
    parser.add_argument('--sample_duration',
                        default=16,
                        type=int,
                        help='Temporal duration of inputs')
    parser.add_argument('--ratio_current',
                        default=0.1,
                        type=float,
                        help='current_model input ratio of data')
    # parser.add_argument('--ratio_future',
    #                     default=1.0,
    #                     type=float,
    #                     help='future_model input ratio of data')
    parser.add_argument(
        '--sample_t_stride',
        default=1,
        type=int,
        help='If larger than 1, input frames are subsampled with the stride.')
    parser.add_argument(
        '--train_crop',
        default='random',
        type=str,
        help=('Spatial cropping method in training. '
              'random is uniform. '
              'corner is selection from 4 corners and 1 center. '
              '(random | corner | center)'))
    parser.add_argument('--train_crop_min_scale',
                        default=0.25,
                        type=float,
                        help='Min scale for random cropping in training')
    parser.add_argument('--train_crop_min_ratio',
                        default=0.75,
                        type=float,
                        help='Min aspect ratio for random cropping in training')
    parser.add_argument('--no_hflip',
                        action='store_true',
                        help='If true holizontal flipping is not performed.')
    parser.add_argument('--colorjitter',
                        action='store_true',
                        help='If true colorjitter is performed.')
    parser.add_argument('--train_t_crop',
                        default='random',
                        type=str,
                        help=('Temporal cropping method in training. '
                              'random is uniform. '
                              '(random | center)'))
    parser.add_argument('--learning_rate',
                        default=1e-06,
                        type=float,
                        help=('Initial learning rate'
                              '(divided by 10 while training by lr scheduler)'))
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--dampening',
                        default=0.0,
                        type=float,
                        help='dampening of SGD')
    parser.add_argument('--weight_decay',
                        default=1e-5,
                        type=float,
                        help='Weight Decay')
    parser.add_argument('--mean_dataset',
                        default='kinetics',
                        type=str,
                        help=('dataset for mean values of mean subtraction'
                              '(activitynet | kinetics | 0.5)'))
    parser.add_argument('--no_mean_norm',
                        action='store_true',
                        help='If true, inputs are not normalized by mean.')
    parser.add_argument(
        '--no_std_norm',
        action='store_true',
        help='If true, inputs are not normalized by standard deviation.')
    parser.add_argument(
        '--value_scale',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-1]. If 255, range of inputs is [0-255].')
    parser.add_argument('--nesterov',
                        action='store_true',
                        help='Nesterov momentum')
    parser.add_argument('--optimizer',
                        default='sgd',
                        type=str,
                        help='Currently only support SGD')
    parser.add_argument('--lr_scheduler',
                        default='multistep',
                        type=str,
                        help='Type of LR scheduler (multistep | plateau)')
    parser.add_argument(
        '--multistep_milestones',
        # default=[3,30,50],
        # default=[20, 40, 60],
        default=[15, 25, 35],
        type=int,
        nargs='+',
        help='Milestones of LR scheduler. See documentation of MultistepLR.')
    parser.add_argument(
        '--overwrite_milestones',
        action='store_true',
        help='If true, overwriting multistep_milestones when resuming training.'
    )
    parser.add_argument(
        '--plateau_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument('--batch_size',
                        default=4,
                        type=int,
                        help='Batch Size')
    parser.add_argument(
        '--inference_batch_size',
        default=1,
        type=int,
        help='Batch Size for inference. 0 means this is the same as batch_size.'
    )
    parser.add_argument(
        '--batchnorm_sync',
        action='store_true',
        help='If true, SyncBatchNorm is used instead of BatchNorm.')
    parser.add_argument('--val',
                        # default=True,
                        action='store_true',
                        help='ceshi val')
    parser.add_argument('--cls_path',
                        default='/home/lz/3D-ResNets-PyTorch/data/results_resnext101_hmdb51/save_cls_150.pth',
                        type=Path)
    parser.add_argument('--MLP_path',
                        default='/home/lz/3D-ResNets-PyTorch/data/results_resnext101_hmdb51/save_MLP_150.pth',
                        type=Path)
    parser.add_argument('--n_epochs',
                        default=100,
                        type=int,
                        help='Number of total epochs to run')
    parser.add_argument('--n_val_samples',
                        default=3,
                        type=int,
                        help='Number of validation samples for each activity')
    parser.add_argument('--resume_path',
                        default=None,
                        type=Path,
                        help='Save data (.pth) of previous training')
    parser.add_argument('--no_train',
                        # default=True,
                        action='store_true',
                        help='If true, training is not performed.')
    parser.add_argument('--no_val',
                        default=True,
                        action='store_true',
                        help='If true, validation is not performed.')
    parser.add_argument('--inference',
                        default=True,
                        action='store_true',
                        help='If true, inference is performed.')
    parser.add_argument('--inference_subset',
                        default='val',
                        type=str,
                        help='Used subset in inference (train | val | test)')
    parser.add_argument('--inference_stride',
                        default=16,
                        type=int,
                        help='Stride of sliding window in inference.')
    parser.add_argument(
        '--inference_crop',
        default='nocrop',
        type=str,
        help=('Cropping method in inference. (center | nocrop)'
              'When nocrop, fully convolutional inference is performed,'
              'and mini-batch consists of clips of one video.'))
    parser.add_argument(
        '--inference_no_average',
        action='store_true',
        help='If true, outputs for segments in a video are not averaged.')
    parser.add_argument('--ground_truth_path',
                        default='/media/cj/新加卷/datasets/UCF101/ucf101_01.json',
                        type=Path)
    parser.add_argument('--no_cuda',
                        action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--n_threads',
                        default=8,
                        type=int,
                        help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint',
                        default=1,
                        type=int,
                        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--model',
        default='resnext',
        type=str,
        help=
        '(resnet | resnet2p1d | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument('--model_depth',
                        default=101,
                        type=int,
                        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--conv1_t_size',
                        default=7,
                        type=int,
                        help='Kernel size in t dim of conv1.')
    parser.add_argument('--conv1_t_stride',
                        default=1,
                        type=int,
                        help='Stride in t dim of conv1.')
    parser.add_argument('--no_max_pool',
                        action='store_true',
                        help='If true, the max pooling after conv1 is removed.')
    parser.add_argument('--resnet_shortcut',
                        default='B',
                        type=str,
                        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--resnet_widen_factor',
        default=1.0,
        type=float,
        help='The number of feature maps of resnet is multiplied by this value')
    parser.add_argument('--wide_resnet_k',
                        default=2,
                        type=int,
                        help='Wide resnet k')
    parser.add_argument('--resnext_cardinality',
                        default=32,
                        type=int,
                        help='ResNeXt cardinality')
    parser.add_argument('--input_type',
                        default='rgb',
                        type=str,
                        help='(rgb | flow)')
    parser.add_argument('--manual_seed',
                        default=5,
                        type=int,
                        help='Manually set random seed')
    parser.add_argument('--accimage',
                        action='store_true',
                        help='If true, accimage is used to load images.')
    parser.add_argument('--output_topk',
                        default=5,
                        type=int,
                        help='Top-k scores are saved in json file.')
    parser.add_argument('--file_type',
                        default='jpg',
                        type=str,
                        help='(jpg | hdf5)')
    parser.add_argument('--tensorboard',
                        # default=True,
                        action='store_true',
                        help='If true, output tensorboard log file.')
    parser.add_argument(
        '--distributed',
        # default=True,
        action='store_true',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs.')
    parser.add_argument('--dist_url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world_size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')

    args = parser.parse_args()

    return args
