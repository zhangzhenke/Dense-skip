import os
import PIL


from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD



def build_dataset(is_train, args):

    # 预处理
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    # 数据存储方式有要求。
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset


# 图像预处理的转换序列
def build_transform(is_train, args):

    # 将图像数据标准化到一个特定的范围
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # 训练模式
    if is_train:
        # 转换序列
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',    # 双三次插值
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # 评估模式
    # 转换序列列表
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0  # 裁剪比例

    # 调整后的图像尺寸 size。   
    size = int(args.input_size / crop_pct)
    
    t.append(
        # 将图像大小调整到计算出的 size，使用双三次插值。
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    # 将图像裁剪到 args.input_size。
    t.append(transforms.CenterCrop(args.input_size))
    # 将 PIL 图像或 NumPy ndarray 转换为 Tensor。
    t.append(transforms.ToTensor())
    # 标准化图像数据，使用前面定义的 mean 和 std。
    t.append(transforms.Normalize(mean, std))

    # 转换序列合并为一个转换。
    return transforms.Compose(t)
