import torch
import torch.nn as nn
import timm.models.vision_transformer

from functools import partial
from collections import OrderedDict
from typing import List
from utils import Conv2DBlock, Deconv2DBlock



class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, 
                 extract_layers: List,   #提取层的列表。
                 **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)


        # 如果提取层的列表长度不是4，将引发一个错误。
        assert len(extract_layers) == 4, "Please provide 4 layers for skip connections"

        self.extract_layers = extract_layers


        # 解码器部分，其中包含了多个卷积和转置卷积块，以及跳过连接。
        # 跳过连接是一种技术，它允许来自较低层次的特征图在较高层次的解码器中进行融合。

        # 第一个 Conv2DBlock 的输入通道数为3（来自输入图像的通道数），输出通道数为32。
        # 第二个 Conv2DBlock 的输入通道数为32，输出通道数为64。
        # img用
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=0.1),
            Conv2DBlock(32, 64, 3, dropout=0.1),
        )  # 在位置编码之后添加 skip connection，形状应为 H, W, 64


        # 第一个 Deconv2DBlock 的输入通道数为embed_dim（来自编码器的输出），输出通道数为512。
        # 第二个 Deconv2DBlock 的输入通道数为 512，输出通道数为256。
        # 第三个 Deconv2DBlock 的输入通道数为 256，输出通道数为128。
        # transformer第一层用
        self.decoder1 = nn.Sequential(
            Deconv2DBlock(1024, 512, dropout=0.1),
            Deconv2DBlock(512, 256, dropout=0.1),
            Deconv2DBlock(256, 128, dropout=0.1),
        )  # skip connection 1  H/2, W/2, 128


        # 第一个 Deconv2DBlock 的输入通道数为 embed_dim，输出通道数为 512。
        # 第二个 Deconv2DBlock 的输入通道数为 512，输出通道数为256。
        # transformer第二层用
        self.decoder2 = nn.Sequential(
            Deconv2DBlock(1024, 512, dropout=0.1),
            Deconv2DBlock(512, 256, dropout=0.1),
        )  # skip connection 2   H/4, W/4, 256


        # Deconv2DBlock 的输入通道数为 embed_dim，输出通道数为 512。
        # transformer第三层用
        self.decoder3 = nn.Sequential(
            Deconv2DBlock(1024, 512, dropout=0.1)
        )  # skip connection 3   H/8, W/8, 512


        # 输出21通道
        self.decoder = self.create_upsampling_branch()


        # 上采样
        self.upsample3 = nn.ConvTranspose2d(
                in_channels=512,
                out_channels=256,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ) 
        
        self.upsample2 = nn.ConvTranspose2d(
                in_channels=256,
                out_channels=128,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            )
        
        self.upsample1 = nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            )
        
        
        self.cellprob_head = DeepSegmentationHead(
            in_channels=64, out_channels=1
        )
        self.gradflow_head = DeepSegmentationHead(
            in_channels=64, out_channels=2
        )
        



    # 上采样分支
    def create_upsampling_branch(self) -> nn.Module:

        # [1, 1024, 14, 14] -> [1, 512, 28, 28]
        # 使用了 stride=2 和 kernel_size=2，输出特征图的大小是输入特征图的 stride 倍
        # transformer得最后一层直接反卷积来上采样。
        bottleneck_upsampler = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
        )

        # 4层，[1, 512, 28, 28] -> [1, 256, 56, 56]
        decoder3_upsampler = nn.Sequential(
            Conv2DBlock(512 * 2, 512, dropout=0.0),
            Conv2DBlock(512, 512, dropout=0.0),
            Conv2DBlock(512, 512, dropout=0.0),
        )

        # 3层，[1, 256, 56, 56] -> [1, 128, 112, 112]
        decoder2_upsampler = nn.Sequential(
            Conv2DBlock(256 * 2, 256, dropout=0.0),
            Conv2DBlock(256, 256, dropout=0.0),
        )

        # 2层，[1, 128, 112, 112] -> [1, 64, 224, 224]
        decoder1_upsampler = nn.Sequential(
            Conv2DBlock(128 * 2, 128, dropout=0.0),
            Conv2DBlock(128, 128, dropout=0.0),
        )

        # 1层，[1, 64, 224, 224] -> [1, 3, 224, 224]
        decoder0_header = nn.Sequential(
            Conv2DBlock(64 * 2, 64, dropout=0.0),
            Conv2DBlock(64, 64, dropout=0.0),
        )
        # 包一下
        decoder = nn.Sequential(
            OrderedDict(
                [
                    ("bottleneck_upsampler", bottleneck_upsampler),
                    ("decoder3_upsampler", decoder3_upsampler),
                    ("decoder2_upsampler", decoder2_upsampler),
                    ("decoder1_upsampler", decoder1_upsampler),
                    ("decoder0_header", decoder0_header),
                ]
            )
        )

        return decoder



    # 上采样
    def _forward_upsample(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        z2: torch.Tensor,
        z3: torch.Tensor,
        z4: torch.Tensor,
        branch_decoder: nn.Sequential,  # 采样层
    ) -> torch.Tensor:
        
        # 这意味着 z0 将等于 x，而 z1、z2、z3 和 z4 将分别等于 z 中的前四个元素。
        # z4是最后一层输出直接反卷积
        # [1, 512, 28, 28]
        b4 = branch_decoder.bottleneck_upsampler(z4)

        # z3先Deconv2DBlock再和b4拼接做3次卷积
        # [1, 512, 28, 28]
        b3 = self.decoder3(z3)
        # [1, 256, 56, 56]，先卷积
        b3 = branch_decoder.decoder3_upsampler(torch.cat([b3, b4], dim=1))
        # 残差连接
        b3 = self.upsample3(b3 + b4)


        # [1, 256, 56, 56]
        b2 = self.decoder2(z2)
        # [1, 128, 112, 112]
        b2 = branch_decoder.decoder2_upsampler(torch.cat([b2, b3], dim=1))

        b4 = self.upsample3(b4)
        b2 = self.upsample2(b2 + b3 + b4) 


        # [1, 128, 112, 112]
        b1 = self.decoder1(z1)
        # [1, 64, 224, 224]
        b1 = branch_decoder.decoder1_upsampler(torch.cat([b1, b2], dim=1))

        b4 = self.upsample2(b4)
        b3 = self.upsample2(b3)
        b1 = self.upsample1(b1 + b2 + b3 + b4)


        # [1, 64, 224, 224]
        b0 = self.decoder0(z0)
        # [1, 3, 224, 224]
        branch_output = branch_decoder.decoder0_header(torch.cat([b0, b1], dim=1))

        b4 = self.upsample1(b4)
        b3 = self.upsample1(b3)
        b2 = self.upsample1(b2)
        branch_output = (b0 + b1 + b2 + b3 + b4)


        return branch_output



    # 编码器
    def forward_encoder(self, x):

        # 批次
        B = x.shape[0]
        # 得到patch_embedding的序列，timm的api提供。
        # BCHW -> BNC
        x = self.patch_embed(x)

        # B个
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # 拼接
        x = torch.cat((cls_tokens, x), dim=1)
        # 这种相加操作的含义是将每个图像块的位置信息与它自己的内容信息结合在一起。
        x = x + self.pos_embed
        # 在训练过程中随机丢弃输入数据中的某些位置
        x = self.pos_drop(x)


        extracted_layers = []


        # 遍历 self.blocks 列表中的每个 block，并对输入 x 进行处理。
        # 代码的目的是对输入 x 进行一系列的 block 操作，并在特定深度时提取中间层输出。
        # enumerate 函数用于在循环中同时获取索引和元素。
        for depth, blk in enumerate(self.blocks):
            x = blk(x)
            # 当前的 block 深度 depth + 1 是否在 self.extract_layers 列表中。
            # 如果深度在列表中，意味着应该从当前的 block 中提取中间层输出
            if depth + 1 in self.extract_layers:
                #  输出 x 添加到 extracted_layers 列表中。
                extracted_layers.append(x)
   
        x = self.norm(x)


        return extracted_layers
    

    # 解码器前向传播
    def forward_decoder(self, imgs, extracted_layers):

        # 这意味着 z1、z2、z3 和 z4 将分别等于 z 中的前四个元素。
        z1, z2, z3, z4 = extracted_layers[:4]

        # 定义了一个列表 patch_dim，它包含两个整数，分别代表输入特征 x 的最后一个维度（宽度）和
        # 倒数第二个维度（高度）除以 self.patch_size 的结果。self.patch_size 是一个类属性，代表模型中的补丁大小。
        patch_dim = [int(d / 16) for d in [imgs.shape[-2], imgs.shape[-1]]]
        # 它从 z4 中移除了第一个通道（可能是分类令牌），然后将剩余的通道进行转置，最后将结果展平为一维张量，
        # 其形状为 (-1, self.embed_dim, *patch_dim)。embed_dim 是模型的嵌入维度。
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, 1024, *patch_dim)
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, 1024, *patch_dim)
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, 1024, *patch_dim)
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, 1024, *patch_dim)


        out_dict = self._forward_upsample(imgs, z4 + z1 + z3 + z2, 
                                          z4 + z1 + z3 + z2, z4 + z1 + z2 + z3, 
                                          z1 + z4 + z3 + z2, self.decoder)



        return out_dict
    
    
    # 全局
    def forward(self, imgs):
        
        out = self.forward_encoder(imgs)
        
        x = self.forward_decoder(imgs, out)

        # Generate masks for cell probability and gradient flows
        cellprob_mask = self.cellprob_head(x)
        gradflow_mask = self.gradflow_head(x)

        # Concatenate the masks for output
        # Output shape is B x [grad y, grad x, cellprob] x H x W
        masks = torch.cat([gradflow_mask, cellprob_mask], dim=1)

        return masks




class DeepSegmentationHead(nn.Sequential):
    """Custom segmentation head for generating specific masks"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        # Define a sequence of layers for the segmentation head
        layers = [
            nn.Conv2d(
                in_channels,
                in_channels // 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(in_channels // 2),
            nn.Conv2d(
                in_channels // 2,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity(),
        ]
        super().__init__(*layers)
        


# 初始化
def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        extract_layers = [6, 12, 18, 24],
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

