# 主入口
import argparse
import datetime
import json
import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm.optim.optim_factory as optim_factory
import util.misc as misc
import models_mae
import matplotlib.pyplot as plt


from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from engine_pretrain import train_one_epoch



def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    # 批次大小
    parser.add_argument('--batch_size', default=64, type=int, 
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # 轮数
    parser.add_argument('--epochs', default=200, type=int)
    # 累计几步算一次loss
    parser.add_argument('--accum_iter', default=1, type=int, 
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')


    # Model parameters
    # 模型
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL', 
                        help='Name of model to train')
    # 输入尺寸
    parser.add_argument('--input_size', default=224, type=int, 
                        help='images input size')
    # 遮盖比例
    parser.add_argument('--mask_ratio', default=0.75, type=float, 
                        help='Masking ratio (percentage of removed patches).')
    # 是否对像素归一化再算loss
    # 当用户在命令行中指定这个参数时，它的行为就像设置了一个布尔值True。
    parser.add_argument('--norm_pix_loss', action='store_true', 
                        help='Use (per-patch) normalized pixels as targets for computing loss')


    # Optimizer parameters
    # 它表示权重衰减，即在训练过程中减少模型的权重参数，以防止过拟合。
    parser.add_argument('--weight_decay', type=float, default=0.05, 
                        help='weight decay (default: 0.05)')
    # 它表示学习率，即优化器调整权重参数的速度。如果设置了这个参数，它将作为绝对学习率使用。默认值为None。
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    # 它表示基础学习率，即在计算绝对学习率时使用的基准值。绝对学习率是通过将基础学习率乘以总批量大小除以256来计算的。
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    # 它表示学习率的下限，即在周期性学习率调度器中学习率可以达到的最小值。
    parser.add_argument('--min_lr', type=float, default=0.0, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    # 它表示预热 epochs，即在训练开始时逐渐增加学习率，以帮助模型更好地适应训练过程。
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')


    # Dataset parameters
    # 数据目录
    parser.add_argument('--data_path', default='/home/share/huadjyin/home/zhangzhenke/Code/models/ShuJu/', type=str,
                        help='dataset path')
    # 输出目录
    parser.add_argument('--output_dir', default='./zzk/01/',
                        help='path where to save, empty for no saving')
    # 日志目录
    parser.add_argument('--log_dir', default='./zzk/01/',
                        help='path where to tensorboard log')
    # GPU
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # 随机种子
    parser.add_argument('--seed', default=0, type=int)
    # 它表示从哪个检查点恢复训练。设置了这个参数，程序将加载指定检查点的权重，并从指定的开始 epoch 继续训练。
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    # 它表示训练的开始 epoch。如果设置了--resume参数，程序将从指定的检查点继续训练，而不是从0开始。
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    # 它表示在数据加载器（DataLoader）中使用的线程数量。这个参数可以影响数据加载的速度和效率。
    parser.add_argument('--num_workers', default=10, type=int)
    # 设置为True，数据加载器会尝试将数据加载到固定内存中，以便更高效地将数据传输到GPU。
    # 这可以提高某些情况下的数据加载效率。
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    # 如果设置为True，则数据加载器（DataLoader）不会尝试将数据加载到固定内存中。
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    # 这意味着如果用户没有明确设置--pin_mem参数，程序将默认使用固定内存来加载数据。
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


    # 图片预处理
    transform_train = transforms.Compose([
            # 随机裁剪图像到指定的大小 args.input_size，
            # scale 参数指定了裁剪区域相对于原始图像的比例范围，原始图像大小的20%（0.2倍）到100%（1.0倍）之间随机选择
            # interpolation=3 表示使用三次插值方法进行缩放，这是一种高质量的图像缩放方法。
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            # 以一定的概率随机水平翻转图像。
            transforms.RandomHorizontalFlip(),
            # u8 --> float
            # 将 PIL 图像或 Numpy 数组转换为 FloatTensor，并将数值范围从 [0, 255] 转换到 [0.0, 1.0]。
            transforms.ToTensor(),
            # 对图像进行标准化（归一化），使用 ImageNet 数据集的均值和标准差。
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    # 正态分布后的数据
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)


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

    else:
        # RandomSampler对象，它是一个随机采样器。
        # 在这种情况下，数据不会被分布到不同的进程，因此使用随机采样器来随机打乱数据。
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

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
        # 训练数据集的实例，提供了获取数据样本的方法。
        dataset_train, 
        # 采样器实例，如何从数据集中采样元素。
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        # 当数据集的大小不能被批处理大小整除时，DataLoader会丢弃最后一个不完整的批次。
        drop_last=True,
    )
    

    # 实例化模型
    model = models_mae.__dict__[args.model]()

    pretrained_weights = torch.load('/home/share/huadjyin/home/zhangzhenke/Code/mae-main/mae_pretrain_vit_large.pth', map_location='cpu')

    checkpoint_model = pretrained_weights['model']

    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    model.to(device)

    # ddp包裹一下，分布式的
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

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
        # 模型将被分配到列表中的所有GPU上。
        # DDP将自动检测那些在分布式训练过程中没有被所有进程使用的参数，并将其移除，以减少通信开销。
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        # 将model_without_ddp重新设置为模型的内部模块。
        # 在DDP包装的模型中，model是包装模块的一个实例，而model.module是原始的模型模块。
        # 通过获取model.module，您可以访问原始的模型模块，而不是包装后的实例。
        model_without_ddp = model.module
    


    # optim_factory包含了一些优化器的创建和配置方法。
    # add_weight_decay用于在优化器的参数组中添加权重衰减。
    # model_without_ddp是一个指向原始模型模块的引用，它包含模型的参数。
    # 返回的param_groups是一个包含模型参数的列表，其中每个参数都应用了权重衰减。
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    # 创建了一个AdamW优化器实例。
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # 指定了weight_decay参数，优化器会自动创建一个包含权重衰减参数的参数组。
    print(optimizer)
    # 在自动混合精度（AMP）训练中处理梯度的类。
    # 在AMP训练中，模型的部分操作（如矩阵乘法）可以在低精度（如半精度浮点数）下进行，以提高计算效率。
    # 但是，反向传播和参数更新仍然需要在全精度下进行。
    loss_scaler = NativeScaler()

    # 是否resume导入模型，优化器参数
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    # 绘图loss曲线
    plt_train_loss = []

    # 每轮
    for epoch in range(args.start_epoch, args.epochs):
        # 
        if args.distributed:
            # 可以确保在不同的周期中，每个进程看到的样本是不同的，从而使训练更加随机和多样化。
            data_loader_train.sampler.set_epoch(epoch)

        # 一轮
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )

        # 存储
        if args.output_dir and (epoch % 25 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        # train_stats是一个字典，其中包含了一些训练过程中的统计信息。
        # 它遍历train_stats字典中的每个键值对，并将它们转换为一个新的字典。
        # 例如，如果train_stats中有键'loss'和值1.23，那么这个字典推导将会创建一个新的键'train_loss'和值1.23。
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        

        # 绘图
        # 将log_stats字典中'train_loss'键对应的值添加到plt_train_loss列表中。这通常是用来存储每个epoch的损失值，以便于绘制曲线。
        plt_train_loss.append(log_stats['train_loss'])


        
        # 只在主进程上进行写操作
        if args.rank in [-1, 0]:
            with open(os.path.join(args.output_dir, "loss.txt"), mode="a", encoding="utf-8") as f:
                f.write(str(epoch) + "  " + str(log_stats['train_loss']) + "\n")


        # 损失没有明显下降
        """if epoch > 50:
            if abs(plt_train_loss[epoch - 10] - plt_train_loss[epoch]) < 0.00005:
                break"""


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


    # Loss曲线
    # 使用Python的Matplotlib库来绘制一个曲线图。plt_train_loss列表中的值将作为x轴的值，而损失值将作为y轴的
    plt.plot(plt_train_loss)
    plt.title('train_loss')
    # 为图表添加一个图例，表示“train”数据集。
    plt.legend(['pretrain'])
    plt.savefig('./zzk/01/' + 'train_loss.png')
    print('plot achieve')
 


# 入口
if __name__ == '__main__':
    # 从命令行获取参数
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
