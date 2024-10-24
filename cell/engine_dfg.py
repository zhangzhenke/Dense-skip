import numpy as np
import distributed_utils as utils
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import math
import sys
import torch.nn as nn

from use_able import compute_masks, labels_to_flows
from measures import evaluate_f1_score_cellseg
from typing import Iterable
from monai.metrics import CumulativeAverage
from monai.inferers import sliding_window_inference



# 损失函数
mse_loss = nn.MSELoss(reduction="mean")
bce_loss = nn.BCEWithLogitsLoss(reduction="mean")



# 组合损失
def mediar_criterion(device, outputs, labels_onehot_flows):
    """loss function between true labels and prediction outputs"""

    # Cell Recognition Loss
    cellprob_loss = bce_loss(
        outputs[:, -1],
        torch.from_numpy(labels_onehot_flows[:, 1] > 0.5).to(device).float(),
    )

    # Cell Distinction Loss
    gradient_flows = torch.from_numpy(labels_onehot_flows[:, 2:]).to(device)
    gradflow_loss = 0.5 * mse_loss(outputs[:, :2], 5.0 * gradient_flows)

    loss = cellprob_loss + gradflow_loss

    return loss


# 结果字典
def _update_results(phase_results, metric, metric_key, phase="train"):
    """Aggregate and flush metrics

        Args:
            phase_results (dict): base dictionary to log metrics
            metric (_type_): cumulated metrics
            metric_key (_type_): name of metric
            phase (str, optional): current phase name. Defaults to "train".

        Returns:
            dict: dictionary of metrics for the current phase
    """

    # Refine metrics name
    metric_key = "_".join([phase, metric_key]).title()


    # 聚合度量
    metric_item = round(metric.aggregate().item(), 4)

    # Log metrics to dictionary
    phase_results[metric_key] = metric_item

    # Flush metrics
    metric.reset()

    return phase_results



# 每轮
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None,
                    args=None):
    
    # 在训练模式下，模型会应用如dropout、batch normalization等训练时特有的行为。
    model.train(True)
    phase_results = {}
    
    # 它用于记录和打印训练过程中的各种指标。跟踪和打印训练过程中的损失、精度等指标。
    # delimiter="  "表示在打印指标时使用的分隔符是两个空格。
    metric_logger = misc.MetricLogger(delimiter="  ")

    loss_metric = CumulativeAverage()

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


    # data_iter_step是当前迭代的计数器
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # 检查当前迭代计数器data_iter_step是否达到了累积步数accum_iter的倍数。
        if data_iter_step % accum_iter == 0:
            # 这行代码调用学习率调度器（lr_sched）来调整优化器（optimizer）的学习率。
            # 学习率调整的时机是在每个累积步数（accum_iter）的倍数处，即每accum_iter个迭代。
            # data_iter_step / len(data_loader) + epoch表示当前累积步数在整体训练过程中的比例。
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)


        samples = samples.to(device, non_blocking=True)# (shape: (batch_size, 3, img_h, img_w))

        # [B, H, W]
        targets = targets.to(device, non_blocking=True)



        with torch.cuda.amp.autocast():
            with torch.set_grad_enabled(True):
                # [B, 20, 224, 224], 
                outputs = model(samples)# (shape: (batch_size, num_classes, img_h, img_w))

                labels_onehot_flows = labels_to_flows(targets, use_gpu=True, device=device)
                # 计算损失
                loss = mediar_criterion(device, outputs, labels_onehot_flows)

                loss_metric.append(loss)


        # 将计算出的损失值转换为浮点数，并存储在变量loss_value中。
        loss_value = loss.item()

        # 检查损失值是否为无穷大或未定义。如果是，这通常意味着模型遇到了一个不可训练的点或梯度爆炸，导致损失值不正常。
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 这行代码将损失值除以累积步数accum_iter，以便在每个累积步数中正确地累积梯度。
        loss /= accum_iter
        # 使用loss_scaler来处理混合精度训练中的梯度缩放。
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)

        # 优化器梯度是否置零, 前迭代计数器是否达到了累积步数accum_iter的倍数。
        if (data_iter_step + 1) % accum_iter == 0:
            # 这行代码将优化器中的所有梯度清零，以便开始新的迭代。
            optimizer.zero_grad()


        # 等待所有GPU上的操作完成，以确保异步操作的同步。
        torch.cuda.synchronize()

        # 更新MetricLogger对象中的损失值loss_value。
        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        # 从优化器的参数组中获取学习率，并将其存储在变量lr中。
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        # 更新MetricLogger对象中的学习率lr。
        metric_logger.update(lr=max_lr)

        # 辅助函数来计算所有进程的平均损失值。
        # 在分布式训练中，这有助于确保所有进程都使用相同的损失值。
        #loss_value_reduce = misc.all_reduce_mean(loss_value)

        # 检查是否有日志记录器可用，并且当前迭代计数器是否达到了累积步数accum_iter的倍数。
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            #epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1)
            #log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            #log_writer.add_scalar('lr', max_lr, epoch_1000x)
            continue


    # Update metrics
    phase_results = _update_results(
            phase_results, loss_metric, "dice_loss", "train"
    )

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
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, phase_results



# 后处理
def _post_process(outputs, labels, device):
    """Predict cell instances using the gradient tracking"""
    outputs = outputs.squeeze(0).cpu().numpy()
    gradflows, cellprob = outputs[:2], _sigmoid(outputs[-1])

    outputs = compute_masks(gradflows, cellprob, use_gpu=True, device=device)
    outputs = outputs[0]  # (1, C, H, W) -> (C, H, W)

    if labels is not None:
        labels = labels.squeeze(0).squeeze(0).cpu().numpy()

    return outputs, labels



def _sigmoid(z):
        """Sigmoid function for numpy arrays"""
        return 1 / (1 + np.exp(-z))

# f1分数
def _get_f1_metric(masks_pred, masks_true):
        f1_score = evaluate_f1_score_cellseg(masks_true, masks_pred)[-1]

        return f1_score


# 验证
def evaluate(model, data_loader, device):
    model.eval()

    phase_results = {}

    metric_logger = utils.MetricLogger(delimiter="  ")
    f1_metric = CumulativeAverage()

    header = 'Test:'


    for image, target in metric_logger.log_every(data_loader, 20, header):

        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
            

        with torch.no_grad():
            # [1, -, 224, 224], 
            outputs = model(image)

            outputs, labels = _post_process(outputs, target, device)

            f1_score = _get_f1_metric(outputs, labels)

            if f1_score == 0:
                f1_score = 0.0001

            f1_score = torch.tensor(f1_score, device=device)
            
            f1_metric.append(f1_score)


    # Update metrics
    phase_results = _update_results(phase_results, f1_metric, "f1_score", "val")
                    
    return phase_results

