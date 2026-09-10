"""EDSR 网络定义 - 严格匹配已训练权重的结构.

架构 (从权重文件反推):
- n_resblocks = 16, n_feats = 64
- 无 MeanShift (直接处理 [0,255] 或 [0,1])
- head: Conv2d(3, 64, 3)  裸卷积
- body: 16 个 ResBlock, 每个 ResBlock.layers = [Conv, ReLU, Conv], res_scale=0.1
- body_conv: Conv2d(64, 64, 3)  body 后的卷积
- tail: 依 scale 而定
    x2: [Conv(64,256,3), PixelShuffle(2), Conv(64,3,3)]
    x4: [Conv(64,256,3), PixelShuffle(2), Conv(64,256,3), PixelShuffle(2), Conv(64,3,3)]
"""
import math

import torch
import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size // 2), bias=bias)


class ResBlock(nn.Module):
    """残差块: layers = [Conv, ReLU, Conv], 输出 = layers(x) * res_scale + x."""

    def __init__(self, conv, n_feats, kernel_size, bias=True, act=nn.ReLU(True),
                 res_scale=0.1):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            conv(n_feats, n_feats, kernel_size, bias=bias),
            act,
            conv(n_feats, n_feats, kernel_size, bias=bias),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return self.layers(x) * self.res_scale + x


class EDSR(nn.Module):
    """EDSR 网络 (匹配训练权重结构)."""

    def __init__(self, n_resblocks=16, n_feats=64, scale=2, n_colors=3):
        super(EDSR, self).__init__()
        kernel_size = 3
        act = nn.ReLU(True)
        self.scale = scale

        # Head: 裸卷积
        self.head = default_conv(n_colors, n_feats, kernel_size)

        # Body: 16 个 ResBlock
        self.body = nn.Sequential(
            *[ResBlock(default_conv, n_feats, kernel_size, act=act)
              for _ in range(n_resblocks)]
        )

        # body 后的卷积 (用于残差连接)
        self.body_conv = default_conv(n_feats, n_feats, kernel_size)

        # Tail: PixelShuffle 上采样, 每级 Conv(64→256) + PixelShuffle(2)
        tail_modules = []
        num_upsample = int(round(math.log(scale, 2)))  # x2->1, x4->2, x8->3
        for _ in range(num_upsample):
            tail_modules.append(default_conv(n_feats, n_feats * 4, kernel_size))
            tail_modules.append(nn.PixelShuffle(2))
        # 最后一层 Conv 输出 3 通道
        tail_modules.append(default_conv(n_feats, n_colors, kernel_size))
        self.tail = nn.Sequential(*tail_modules)

    def forward(self, x):
        x = self.head(x)
        res = self.body_conv(self.body(x))
        x = x + res
        x = self.tail(x)
        return x


def create_edsr(scale=2, n_resblocks=16, n_feats=64):
    """工厂函数: 创建 EDSR 实例."""
    return EDSR(n_resblocks=n_resblocks, n_feats=n_feats, scale=scale)


def get_architecture_for_scale(scale):
    """根据放大倍率返回网络架构配置."""
    if scale in (2, 4, 8):
        return dict(n_resblocks=16, n_feats=64, scale=scale)
    raise ValueError(f"Unsupported scale: {scale}")
