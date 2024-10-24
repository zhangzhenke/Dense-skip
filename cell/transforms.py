import numpy as np
import random
import torch

from torchvision import transforms as T
from torchvision.transforms import functional as F


# 填充
def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


# 打包
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# 调整最小边
class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):

        # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
        image = F.resize(image, (self.min_size, self.max_size), interpolation=T.InterpolationMode.BICUBIC)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        # 标签也缩放
        target = F.resize(target, (self.min_size, self.max_size), interpolation=T.InterpolationMode.NEAREST)
       
        return image, target


# 随机翻转
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


# 裁剪
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        # 检查图像的尺寸是否小于目标尺寸，并在必要时对图像进行填充（padding）以确保图像尺寸至少与目标尺寸一样大。
        image = pad_if_smaller(image, self.size)
        # 填充值被设置为255，这通常用于分割掩码，其中255可以是一个忽略的类标签（背景或未标注区域）
        target = pad_if_smaller(target, self.size)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


# 张量
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


# 标准化
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
