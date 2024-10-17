import math

# 调整优化器的学习率，它采用了半周期余弦衰减的策略，并在预热期结束后开始应用。
def adjust_learning_rate(optimizer, epoch, args):

    # 当前的训练周期（epoch）小于预热期的epoch数
    if epoch < args.warmup_epochs:
        # 学习率设置为初始学习率（args.lr）与当前epoch的比例。这意味着在预热期内，学习率会线性增加。
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        # 学习率设置为最小学习率（args.min_lr）加上一个基于余弦函数的衰减部分。
        # 余弦函数的参数是当前epoch与预热期结束后的epoch数之差除以剩余epoch数与预热期结束后的epoch数之差。
        # 这实现了半周期余弦衰减，学习率会在预热期结束后的某个点达到最大值，然后开始逐渐减小。
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    # 遍历优化器中的每个参数组。
    for param_group in optimizer.param_groups:
        # 参数组中存在一个名为"lr_scale"的键
        # lr_scale 通常是一个缩放因子，用于调整学习率（learning rate）的大小。
        if "lr_scale" in param_group:
            # 将参数组的学习率设置为当前学习率（lr）乘以参数组中的"lr_scale"值。
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            # 将参数组的学习率设置为当前学习率（lr）。
            param_group["lr"] = lr
    # 返回当前的学习率
    return lr
