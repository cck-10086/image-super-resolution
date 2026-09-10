"""RRDBNet (Real-ESRGAN) 网络定义与推理包装.

基于 Real-ESRGAN 论文 (Wang et al., 2021) 与官方实现.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np


def conv2d(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size,
                    stride=stride, padding=(kernel_size // 2), bias=bias)


class ResidualDenseBlock(nn.Module):
    """残差密集块 (RRDB 内部)."""

    def __init__(self, num_feat=64, num_grow_ch=32, scale=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = conv2d(num_feat, num_grow_ch, 3)
        self.conv2 = conv2d(num_feat + num_grow_ch, num_grow_ch, 3)
        self.conv3 = conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3)
        self.conv4 = conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3)
        self.conv5 = conv2d(num_feat + 4 * num_grow_ch, num_feat, 3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.scale = scale

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * self.scale + x


class RRDB(nn.Module):
    """残差中残差块 (Residual in Residual Dense Block)."""

    def __init__(self, num_feat, num_grow_ch=32, scale=0.2):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch, scale)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch, scale)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch, scale)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Real-ESRGAN 网络结构 (官方实现).

    上采样方式: nearest-neighbor 插值 + Conv (非 PixelShuffle).
    权重键名: conv_first, body.X.rdbY.convZ, conv_body, conv_up1, conv_up2, conv_hr, conv_last.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4,
                 num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale

        self.conv_first = conv2d(num_in_ch, num_feat, 3)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = conv2d(num_feat, num_feat, 3)

        # 上采样: nearest-neighbor 2x + Conv(64,64,3) → LReLU, 重复 log2(scale) 次
        self.conv_up1 = conv2d(num_feat, num_feat, 3)
        self.conv_up2 = conv2d(num_feat, num_feat, 3)
        self.conv_hr = conv2d(num_feat, num_feat, 3)
        self.conv_last = conv2d(num_feat, num_out_ch, 3)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.body(feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat

        # 2x nearest-neighbor → conv_up1 → LReLU
        out = self.lrelu(self.conv_up1(
            F.interpolate(feat, scale_factor=2, mode='nearest')))
        # 2x nearest-neighbor → conv_up2 → LReLU
        out = self.lrelu(self.conv_up2(
            F.interpolate(out, scale_factor=2, mode='nearest')))
        # conv_hr → LReLU → conv_last
        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        return out


class RealESRGANWrapper:
    """Real-ESRGAN 推理包装器，提供与 SuperResolutionModel 相同的 enhance 接口.

    支持大图分块处理，避免 GPU OOM.
    """

    def __init__(self, model, device='cpu', tile_size=512, tile_pad=16):
        self.model = model
        self.device = device
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.scale = getattr(model, 'scale', 4)
        self.model.eval()

    def _tensor_to_pil(self, tensor):
        """[1,C,H,W] float tensor in [0,1] -> PIL Image."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.clamp(0, 1).mul(255).round()
        arr = tensor.byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr, 'RGB')

    def _pil_to_tensor(self, img):
        """PIL Image -> [1,C,H,W] float tensor in [0,1]."""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _process_tile(self, tensor):
        """单次模型推理."""
        with torch.no_grad():
            tensor = tensor.to(self.device)
            out = self.model(tensor)
        return out

    def _tile_process(self, tensor):
        """分块推理，避免大图 OOM."""
        b, c, h, w = tensor.shape
        if h * w <= self.tile_size * self.tile_size:
            return self._process_tile(tensor)

        stride = self.tile_size - self.tile_pad
        out_h = h * self.scale
        out_w = w * self.scale
        out = torch.zeros((b, c, out_h, out_w), dtype=torch.float32, device='cpu')

        # 权重图用于融合重叠区域
        weight = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32, device='cpu')

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                y_start = max(0, y - self.tile_pad) if y > 0 else 0
                x_start = max(0, x - self.tile_pad) if x > 0 else 0

                tile = tensor[:, :, y_start:y_end, x_start:x_end]
                out_tile = self._process_tile(tile).cpu()

                out_y_start = y_start * self.scale
                out_x_start = x_start * self.scale
                out_y_end = y_end * self.scale
                out_x_end = x_end * self.scale

                valid_h = out_y_end - out_y_start
                valid_w = out_x_end - out_x_start

                out[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += \
                    out_tile[:, :, :valid_h, :valid_w]
                weight[:, :, out_y_start:out_y_end, out_x_start:out_x_end] += 1.0

        out = out / weight.clamp(min=1.0)
        return out

    def enhance(self, image_path):
        """对图像进行超分辨率放大.

        参数:
            image_path: 图像路径
        返回:
            PIL Image (RGB)
        """
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        tensor = self._pil_to_tensor(img)
        out = self._tile_process(tensor)
        return self._tensor_to_pil(out)
