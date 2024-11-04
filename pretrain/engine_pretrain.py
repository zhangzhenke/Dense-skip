# 引擎
import math
import sys
import torch
import util.misc as misc
import util.lr_sched as lr_sched

from typing import Iterable


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):

    # 在训练模式下，模型会应用如dropout、batch normalization等训练时特有的行为。
    model.train(True)

    # 它用于记录和打印训练过程中的各种指标。跟踪和打印训练过程中的损失、精度等指标。
    # delimiter="  "表示在打印指标时使用的分隔符是两个空格。
    metric_logger = misc.MetricLogger(delimiter="  ")
    # 向MetricLogger对象添加了一个新的指标，名为’lr’，用于记录学习率。
    # window_size=1表示只记录最近的1个值，并使用这个值进行平滑计算。
    # fmt='{value:.6f}'表示打印时格式化学习率为六位小数。
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # 创建了一个字符串header，用于打印每个epoch的标题。
    header = 'Epoch: [{}]'.format(epoch)
    # 表示每20个iteration打印一次日志。
    print_freq = 20

    # accum_iter通常用于累积梯度，以减少显存的使用或提高训练效率。
    accum_iter = args.accum_iter

    # 将优化器中的所有梯度清零。在每次迭代开始时，通常需要将上一次迭代中累积的梯度清零，以便开始新的迭代。
    optimizer.zero_grad()

    # 日志记录器，用于将训练日志写入文件或TensorBoard。
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))


    # 自监督学习的训练循环
    # data_iter_step是当前迭代的计数器，自监督学习label可以不要。
    # (samples, _)是从数据加载器（data_loader）中得到的当前批次的数据和标签（在这里，标签是None，是自监督学习）。
    # log_every是一个迭代器，它每隔print_freq个批次打印一次日志，并返回当前迭代的计数器和当前批次的数据。
    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):


        # 检查当前迭代计数器data_iter_step是否达到了累积步数accum_iter的倍数。
        if data_iter_step % accum_iter == 0:
            # 这行代码调用学习率调度器（lr_sched）来调整优化器（optimizer）的学习率。
            # 学习率调整的时机是在每个累积步数（accum_iter）的倍数处，即每accum_iter个迭代。
            # data_iter_step / len(data_loader) + epoch表示当前累积步数在整体训练过程中的比例。
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        # 将当前批次的数据samples移动到指定的设备（device）上。
        # non_blocking=True表示如果设备当前忙碌，则允许异步操作，这可以提高效率。
        samples = samples.to(device, non_blocking=True)


        # autocast上下文管理器在进入和退出时自动将模型和计算图的运算精度从浮点32位转换为浮点16位，以减少内存使用和提高性能。
        with torch.cuda.amp.autocast():
            # 调用模型来计算损失。samples是当前批次的数据，mask_ratio是一个超参数，用于控制数据掩码的策略。
            # 函数model返回三个值：损失loss，以及其他可能的值，这里用下划线（_）表示这些值被忽略。
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        # 将计算出的损失值转换为浮点数，并存储在变量loss_value中。
        loss_value = loss.item()

        # 检查损失值是否为无穷大或未定义。如果是，这通常意味着模型遇到了一个不可训练的点或梯度爆炸，导致损失值不正常。
        if not math.isfinite(loss_value):
            # 如果损失值不正确，这行代码将打印一条错误消息并停止训练。
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 这行代码将损失值除以累积步数accum_iter，以便在每个累积步数中正确地累积梯度。
        loss /= accum_iter

        '''
        这行代码使用loss_scaler来处理混合精度训练中的梯度缩放。
        loss_scaler是一个辅助类, 通常用于处理异步混合精度训练中的数值稳定性问题。
        loss是当前批次计算出的损失值。
        optimizer是用于更新模型参数的优化器。
        parameters=model.parameters()表示缩放器的参数是模型的所有参数。
        update_grad=(data_iter_step + 1) % accum_iter == 0是一个条件, 用于确定是否需要更新梯度。
            只有在每个累积步数结束时, 才需要更新梯度。
        '''
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        # 优化器梯度是否置零, 前迭代计数器是否达到了累积步数accum_iter的倍数。
        if (data_iter_step + 1) % accum_iter == 0:
            # 这行代码将优化器中的所有梯度清零，以便开始新的迭代。
            optimizer.zero_grad()

        # 等待所有GPU上的操作完成，以确保异步操作的同步。
        torch.cuda.synchronize()

        # 更新MetricLogger对象中的损失值loss_value。
        metric_logger.update(loss=loss_value)

        # 从优化器的参数组中获取第一个参数组的学习率，并将其存储在变量lr中。
        lr = optimizer.param_groups[0]["lr"]
        # 更新MetricLogger对象中的学习率lr。
        metric_logger.update(lr=lr)

        # 辅助函数来计算所有进程的平均损失值。
        # 在分布式训练中，这有助于确保所有进程都使用相同的损失值。
        #loss_value_reduce = misc.all_reduce_mean(loss_value)

        
        # 检查是否有日志记录器可用，并且当前迭代计数器是否达到了累积步数accum_iter的倍数。
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ 
            这行代码表明, 他们使用“epoch_1000x”作为x轴来表示时间或每个epoch处理的数据量, 
            这样可以校准不同批次大小下的网络性能曲线。
            """
            # “1000x”可能是指将每个epoch的训练数据集扩展1000倍。
            # 换句话说，它可能是指在每个epoch中，网络会处理1000倍于原始训练数据集的数据量。
            #epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1)
            # 将计算出的损失值loss_value_reduce添加到TensorBoard的“train_loss”日志中。epoch_1000x是x轴的值。
            #log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            # 将当前的学习率lr添加到TensorBoard的“lr”日志中。epoch_1000x是x轴的值。
            #log_writer.add_scalar('lr', lr, epoch_1000x)

            '''
            add_scalar方法是TensorBoard日志记录器的一个方法, 用于记录标量值, 并将这些值添加到日志中。
            这些标量值是从metric_logger对象中获取的, 然后通过log_writer记录到TensorBoard中, 以便于可视化和分析。

            具体来说, log_writer.add_scalar方法通常会接收三个参数:

                日志名称(例如, 'train_loss' 或 'lr')
                标量值(例如, loss_value_reduce 或 lr)
                x轴的值(例如, epoch_1000x)在数据记录过程中发生了多少次迭代.TensorBoard会根据这个值在图形中绘制数据点。
                这个方法会将指定的标量值添加到TensorBoard日志中, 并使用提供的x轴值作为数据点的标签。
                它不会直接修改metric_logger对象中的值, 而是将这些值作为记录到TensorBoard中的数据点。
            '''
            continue
            



    # 在多进程或多节点分布式训练中，每个进程或节点可能会有自己的统计信息，例如损失值、精度等。
    # 使用这个方法可以将这些信息从各个进程或节点收集起来，并进行汇总，以便于查看整体的统计信息。
    metric_logger.synchronize_between_processes()

    # 这行代码打印出经过汇总后的统计信息。
    # 由于在多进程或多节点分布式训练中，每个进程或节点的统计信息可能会有所不同，
    # 因此打印出汇总后的统计信息可以帮助你了解整体训练情况。
    print("Averaged stats:", metric_logger)
    # 这行代码返回一个字典，其中包含了各个统计指标的平均值。
    # 这个字典是通过遍历metric_logger.meters字典生成的，其中meter.global_avg表示每个统计指标的平均值。
    # metric_logger.meters是一个字典，其中包含了各种统计指标，如损失值、精度等。
    # 通过遍历这个字典，我们可以得到每个统计指标的平均值，并将其存储在返回的字典中。
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}