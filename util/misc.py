import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist
import sys

# from torch._six import inf

# 使用 sys.float_info.max 和 sys.float_info.min 来代替 inf
max_float = sys.float_info.max
min_float = sys.float_info.min


# SmoothedValue是一个辅助类，用于平滑地记录和打印值。
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


# MetricLogger是一个辅助类，通常用于跟踪和打印训练过程中的损失、精度等指标。
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)
    
    # 这个方法的作用是在多进程或多节点分布式训练中同步各个进程的统计信息。
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    # log_every 是一个迭代器，它可以在遍历一个数据加载器的迭代器时，定期打印出一些统计信息，如时间、内存使用情况等。
    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        # 开始遍历迭代器的时间。
        start_time = time.time()
        # 当前迭代结束的时间。
        end = time.time()
        # 平滑计算每个迭代的时间。
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        # 平滑计算数据加载的时间。
        data_time = SmoothedValue(fmt='{avg:.4f}')
        # 用于在迭代器索引和迭代器长度之间添加空格，以确保索引和长度之间的对齐。
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        # 一个列表，其中包含用于打印信息的格式化字符串。
        log_msg = [
            header,
            # 打印当前迭代索引和迭代器总长度。{0} 将被替换为当前迭代索引，{1} 将被替换为迭代器总长度。
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        # 将字节转换为兆字节（MB）
        MB = 1024.0 * 1024.0
        # 遍历迭代器中的每个元素。
        for obj in iterable:
            # 更新数据加载的时间。
            data_time.update(time.time() - end)
            # 生成迭代器中的每个元素。
            yield obj
            # 更新每个迭代的时间。
            iter_time.update(time.time() - end)
            # 刷新频率
            if i % print_freq == 0 or i == len(iterable) - 1:
                # 剩余迭代所需的时间。
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                # 剩余时间转换为可读的字符串格式。
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    # 打印包含时间、内存使用情况等统计信息的日志消息。
                    print(log_msg.format(
                        i, len(iterable), 
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), 
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    # 当前迭代索引、迭代器总长度、剩余时间、自定义统计信息（来自 self 对象）、
                    # 每个迭代的时间、数据加载的时间以及当前分配的最大CUDA内存的字符串。
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            # 更新当前迭代结束的时间。
            end = time.time()
        # 遍历整个迭代器所需的总时间。
        total_time = time.time() - start_time
        # 总时间转换为可读的字符串格式。
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # 打印包含总时间的日志消息。
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


# 分布式训练的初始化
def init_distributed_mode(args):

    # 如果是，这通常表示是否在ITP（Intel® oneAPI Collective Communications Library）上使用分布式训练。
    if args.dist_on_itp:
        # 进程的全局rank
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        # 总进程数
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        # 进程在本地的rank
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        # 用于进程间通信的TCP URL
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        # world_size GPU的数目
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    # LOCAL_RANK 第几个卡上
    # 但是环境变量中存在RANK和WORLD_SIZE，这通常意味着我们已经在分布式环境中运行，可能是使用其他的分布式通信框架。
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # 如果存在SLURM_PROCID环境变量，这通常意味着程序是在SLURM作业调度系统下运行的。
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    # 如果以上条件都不满足，说明没有使用分布式模式，函数会打印一条消息表示不使用分布式模式
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    # 如果设置了分布式训练模式，设置GPU设备，并初始化分布式进程组。这包括设置后端为nccl，并使用init_method指定的URL进行初始化。
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    # 等待所有的卡初始化完
    torch.distributed.barrier()
    # 如果args.rank == 0，调用setup_for_distributed函数，这通常是用于设置单机分布式训练环境。
    setup_for_distributed(args.rank == 0)



# 在自动混合精度（AMP）训练中处理梯度的类。
# 在AMP训练中，模型的部分操作（如矩阵乘法）可以在低精度（如半精度浮点数）下进行，以提高计算效率。
# 但是，反向传播和参数更新仍然需要在全精度下进行。
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        # 梯度缩放器，create_graph允许在梯度计算过程中创建一个计算图。
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            #  clip_grad 是一个可选参数，当它被设置为非 None 值时，表示用户希望启用梯度裁剪。
            if clip_grad is not None:
                # 确保了 parameters 被正确传递。
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                # 将模型参数的梯度范数裁剪到指定的最大值 clip_grad。
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                # 将优化器分配的参数的梯度进行反缩放
                # （在梯度缩放过程中，梯度被放大以允许更大的数值范围，这一步是必要的，以便在更新参数之前将梯度恢复到原来的规模）。
                self._scaler.unscale_(optimizer)
                # 计算参数梯度的范数，它可以根据指定的范数类型（默认为 L2 范数）来计算梯度的总范数。
                # 这个函数在梯度裁剪后被用来获取裁剪后的梯度范数。
                norm = get_grad_norm_(parameters)
            # 更新模型参数。
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == max_float or min_float :
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


# 保存权重
def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)


# 加载权重
def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        print("Resume checkpoint %s" % args.resume)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            
            # 尝试加载优化器状态
            optimizer_state_dict = checkpoint['optimizer']

            # 确保参数组数量匹配
            if len(optimizer.param_groups) != len(optimizer_state_dict['param_groups']):
                # 你可以选择保留预训练的参数组设置，也可以选择当前优化器的参数组设置
                # 这里我们选择当前优化器的参数组设置
                optimizer_state_dict['param_groups'] = optimizer_state_dict['param_groups'][:len(optimizer.param_groups)]

            # 手动加载学习率、权重衰减和动量
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = optimizer_state_dict['param_groups'][i]['lr']
                optimizer.param_groups[i]['weight_decay'] = optimizer_state_dict['param_groups'][i]['weight_decay']
                


            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


# 计算所有进程的平均损失值。
def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x