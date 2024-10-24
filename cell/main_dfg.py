import time
import argparse
import datetime
import json
import numpy as np
import os
import UNETR
import torch
import torch.backends.cudnn as cudnn
import util.misc as misc
import matplotlib.pyplot as plt
import util.lr_decay as lrd
import re

from torch.utils.tensorboard import SummaryWriter
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_dfg import train_one_epoch, evaluate
from datasets import DatasetTrain, DatasetVal
from pathlib import Path
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_


def get_args_parser():
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)
     # 批量大小
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # 轮数
    parser.add_argument('--epochs', default=100, type=int)
    # 累计步数
    parser.add_argument('--accum_iter', default=2, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')


    # Model parameters
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    # 尺寸
    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')
    # 随机丢弃网络中的某些路径，以防止过拟合的一种正则化技术。
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    

   # Optimizer parameters
    # 梯度在更新模型权重时会被限制在这个阈值以内，以防止梯度爆炸问题。如果设置为None，则表示不对梯度进行裁剪。
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    # 权重衰减
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # 学习率
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    # 基础学习率
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    # 层级衰减是一种优化策略，它允许模型在训练过程中对不同层使用不同的学习率。
    # 默认值0.75可能意味着每层的学习率是前一层的75%。
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')
    # 最小学习率
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    # 预热
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')



    # * Finetuning params
    # 在微调过程中，通常使用在大型数据集上预训练的模型，并在特定任务的小型数据集上对模型进行进一步的训练，以适应特定的任务。
    parser.add_argument('--finetune', default='/home/share/huadjyin/home/zhangzhenke/Code/models/mae_finetuned_vit_large.pth',
                        help='finetune from checkpoint')
    # 全局平均池化
    parser.add_argument('--global_pool', action='store_true')
    # 默认值
    parser.set_defaults(global_pool=False)
    # 如果用户在命令行中使用了--cls_token参数，那么global_pool将被设置为False，否则默认为True。
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')
    

    # Dataset parameters
    #数据
    parser.add_argument('--data_path', default='./data', type=str,
                        help='dataset path')
    # 种类
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')
    # 结果
    parser.add_argument('--output_dir', default='./zzk/02/',
                        help='path where to save, empty for no saving')
    # 日志
    parser.add_argument('--log_dir', default='./zzk/02/',
                        help='path where to tensorboard log')
    # GPU
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # --resume 参数通常用于指定一个检查点(checkpoint)文件，该文件包含了模型训练过程中的状态，包括模型的参数和优化器的状态。
    # 允许用户在训练MAE模型时，通过命令行指定一个检查点文件来恢复训练
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    # 起始
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # 用于指示程序是否只执行评估过程，而不进行训练。
    # 当用户在运行程序时添加了--eval参数，程序将只会执行模型的评估过程，而不是进行训练。
    # store_true意味着当命令行中包含--eval参数时，这个参数的值将被设置为True，而不需要用户提供一个值。
    # 如果命令行中没有这个参数，那么它的值将被默认设置为False。
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    # 它告诉用户这个参数的作用是启用分布式评估。分布式评估通常在训练期间推荐使用，因为它可以更快地监控训练进度。
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    # 线程数
    parser.add_argument('--num_workers', default=10, type=int)
    # 预存储
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    # 不预存储
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    # 默认值
    parser.set_defaults(pin_mem=True)


    # distributed training parameters
    # 它表示分布式训练中的进程数量。
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    # 它表示当前进程的本地排名。在分布式训练中，每个进程都会被分配一个唯一的本地排名，用于确定其角色和任务。
    parser.add_argument('--local_rank', default=-1, type=int)
    # 默认值为False。它表示是否在ITP（Intel® oneAPI Collective Communications Library）上使用分布式训练。
    parser.add_argument('--dist_on_itp', action='store_true')
    # 它表示用于设置分布式训练的URL。env://表示从环境变量中获取分布式训练的URL。
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    return parser




def main(args):
    # 分布式训练
    misc.init_distributed_mode(args)

    # 打印出工作目录和args的参数 。 
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    # 设备
    device = torch.device(args.device)

    # 固定随机种子, args.seed与进程排名相加可以确保每个进程有一个不同的随机种子
    seed = args.seed + misc.get_rank()
    # 以相同的方式生成随机数
    torch.manual_seed(seed)
    # 生成相同的随机数序列
    np.random.seed(seed)

    # 某些层加速
    cudnn.benchmark = True



    dataset_train = DatasetTrain(cityscapes_data_path="./data/cellpose/images/",
                             cityscapes_meta_path="./data/cellpose/annotations/")
    print(dataset_train)
    dataset_val = DatasetVal(cityscapes_data_path="./data/cellpose/images/",
                             cityscapes_meta_path="./data/cellpose/annotations/")
    print(dataset_val)


    num_train_batches = int(len(dataset_train)/args.batch_size)
    num_val_batches = int(len(dataset_val)/args.batch_size)
    print ("num_train_batches:", num_train_batches)
    print ("num_val_batches:", num_val_batches)


    # args.distributed:
    if args.distributed == True: 

        # 获取参与分布式训练的总进程数。这个值通常用于设置分布式训练的参数，如采样器中的num_replicas。
        num_tasks = misc.get_world_size()
        # 来获取当前进程的排名。这个值通常用于设置分布式训练的参数，如采样器中的rank。
        global_rank = misc.get_rank()

        # DistributedSampler对象，它是一个用于分布式训练的采样器。
        # dataset_train是训练数据集，num_replicas是参与分布式训练的总进程数，rank是当前进程的排名，shuffle=True表示在每次迭代开始时重新打乱数据。
        # 每个进程都将获得数据的一个子集，并且每个进程的数据都会被打乱，以确保整个数据集的多样性。
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        # 打印出当前设置的采样器对象sampler_train的详细信息。这有助于调试和了解采样器的设置。
        print("Sampler_train = %s" % str(sampler_train))
        

        # 是否启用了分布式评估（dist_eval）。
        # 在分布式训练中，数据集会被分割到多个进程（或节点）中，每个进程处理一部分数据。
        if args.dist_eval:
            # 如果不能整除，意味着数据不能均匀分配给每个进程。
            if len(dataset_val) % num_tasks != 0:
                # 为了确保每个进程都有相同数量的样本，可能会在数据集中添加额外的重复条目，这可能会略微改变验证结果。
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            # 每个进程都能获取到数据集的一个子集，并且这些子集加起来能覆盖整个数据集。
            # 参数num_replicas表示总的进程数，rank表示当前进程的编号，shuffle=True表示在每个epoch开始时会随机打乱数据。
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        else:
            # 设置了采样器，使得数据以顺序方式被采样。
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        # 打印出当前设置的采样器对象sampler_val的详细信息。这有助于调试和了解采样器的设置。
        print("Sampler_val = %s" % str(sampler_val))


    else:
        # RandomSampler对象，它是一个随机采样器。
        # 在这种情况下，数据不会被分布到不同的进程，因此使用随机采样器来随机打乱数据。
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)


    # 这行代码检查当前进程是否是主进程（global_rank == 0）并且是否指定了日志目录（args.log_dir is not None）。
    if global_rank == 0 and args.log_dir is not None:
        # 这行代码创建指定的日志目录。如果目录已经存在，则不会引发错误。
        os.makedirs(args.log_dir, exist_ok=True)
        # 创建一个SummaryWriter对象，它用于记录和保存TensorBoard日志。log_dir是日志保存的目录。
        # 只有当当前进程是主进程时，才会创建日志记录器。
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        # 只在主进程中创建日志记录器。
        log_writer = None



    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )


    # 实例化模型
    model = UNETR.__dict__[args.model](        
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        img_size = args.input_size
        )
    

    # 在一个预训练模型的基础上，继续训练以适应一个新的数据集或任务。
    if args.finetune and not args.eval:

        # 加载预训练模型的检查点。
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        # 从检查点中提取模型的状态字典
        checkpoint_model = checkpoint['model']

        # 获取所有的键
        keys = list(checkpoint_model.keys())
        print("checkpoint_model: " )
        
        # 打印所有的键
        """for key in keys:
            print(key)"""


        # 获取当前模型的状态字典。
        state_dict = model.state_dict()

        for k in ['head.weight', 'head.bias']:
            # 检查预训练模型检查点中的头部权重和偏置是否与当前模型的权重和偏置形状不同。
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                # 删除预训练模型中对应的键。
                del checkpoint_model[k]


        # 调整和适配模型的位置嵌入（position embedding）矩阵，以匹配模型期望的输入尺寸。
        # 这通常在加载预训练模型并将其应用于不同分辨率的输入时使用。
        interpolate_pos_embed(model, checkpoint_model)


        # 将预训练模型的状态字典加载到当前模型中。
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)


        # 对神经网络模型中全连接层（fully connected layer，通常简称为fc层）的权重进行初始化的操作。
        trunc_normal_(model.head.weight, std=2e-5)

    


    model.to(device)

    # ddp包裹一下，分布式的
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # 有效batch_size ，每个训练迭代中每个进程实际处理的数据量。
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    # 学习率基于上面调整
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 128

    print("base lr: %.2e" % (args.lr * 128 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)


    # 多机多卡
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module


    # 构建优化器的参数组
    param_groups = lrd.param_groups_lrd(
        model_without_ddp,  # 模型实例
        args.weight_decay,  # 权重衰减
        no_weight_decay_list=model_without_ddp.no_weight_decay(), # 返回一个列表，包含不应该应用权重衰减的参数名。通常，这包括批量归一化（Batch Normalization）层的参数和偏差项（biases）。
        layer_decay=args.layer_decay    # 层级学习率衰减的系数
    )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    # 在自动混合精度（AMP）训练中处理梯度的类。
    # 在AMP训练中，模型的部分操作（如矩阵乘法）可以在低精度（如半精度浮点数）下进行，以提高计算效率。
    # 但是，反向传播和参数更新仍然需要在全精度下进行。
    loss_scaler = NativeScaler()

    # 交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))
 
    # 是否resume导入模型，优化器参数
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)



    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # 绘图acc曲线
    plt_val_acc = []

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            # 可以确保在不同的周期中，每个进程看到的样本是不同的，从而使训练更加随机和多样化。
            data_loader_train.sampler.set_epoch(epoch)

        train_stats, reslusts = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args
        )


        if (epoch % 10 == 0 or epoch + 1 == args.epochs):
            if args.output_dir:
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")


        confmat = evaluate(model, data_loader_val, device=device)
        val_info = str(confmat)
        print(confmat)


        # 使用正则表达式提取数字
        match = re.search(r'\d+\.\d+', val_info)
        if match:
            number = float(match.group())
        else:
            print("No number found")



        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:
            with open(os.path.join(args.output_dir, "F1.txt"), mode="a", encoding="utf-8") as f:
                f.write(str(epoch) + "  " + str(number) + "\n")


        # 绘图
        plt_val_acc.append(number)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    # 使用Python的Matplotlib库来绘制一个曲线图。plt_train_loss列表中的值将作为x轴的值，而损失值将作为y轴的
    plt.plot(plt_val_acc)
    plt.title('miou')
    # 为图表添加一个图例，表示“train”数据集。
    plt.legend(['FG'])
    # 假设 args.output_dir 是一个已经存在的目录路径
    output_path = os.path.join(args.output_dir, 'cellpose_f1.png')
    plt.savefig(output_path)
    print('plot achieve')



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)