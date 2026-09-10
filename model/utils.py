"""推理工具：双三次插值、超分模型包装、分块处理."""
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .edsr import EDSR, create_edsr, get_architecture_for_scale


def bicubic_interpolation(image_path, scale_factor):
    """双三次插值放大图像.

    参数:
        image_path: 图像路径
        scale_factor: 放大倍率
    返回:
        PIL Image (RGB)
    """
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    new_size = (img.width * scale_factor, img.height * scale_factor)
    return img.resize(new_size, Image.BICUBIC)


class SuperResolutionModel:
    """超分辨率模型包装器.

    提供 enhance(path) -> PIL Image 接口，自动加载/管理 EDSR 权重.
    """

    def __init__(self, scale_factor=2, weights_path=None, device=None):
        self.scale_factor = scale_factor
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 架构配置
        arch = get_architecture_for_scale(scale_factor)
        self.model = EDSR(
            n_resblocks=arch['n_resblocks'],
            n_feats=arch['n_feats'],
            scale=arch['scale']
        )

        # 自动定位权重文件
        self.weights_path = weights_path or self._find_weights()

        if self.weights_path and os.path.exists(self.weights_path):
            self._load_weights()
        else:
            # 没有权重也能运行，但效果差
            print(f"[Warning] No weights found for {scale_factor}x model, using random init.")

        self.model = self.model.to(self.device).eval()

    def _find_weights(self):
        """自动查找 EDSR 权重文件."""
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        weights_dir = os.path.join(project_dir, 'model_weights')

        candidates = [
            f'edsr_x{self.scale_factor}_best.pth',
            f'edsr_x{self.scale_factor}.pth',
            f'EDSR_x{self.scale_factor}.pth',
        ]

        for name in candidates:
            path = os.path.join(weights_dir, name)
            if os.path.exists(path):
                return path

        # 退一步查找任意 x2/x4 权重
        for f in os.listdir(weights_dir) if os.path.isdir(weights_dir) else []:
            if f.startswith(f'edsr_x{self.scale_factor}'):
                return os.path.join(weights_dir, f)

        return None

    def _load_weights(self):
        """加载 EDSR 权重 (训练时保存格式: model_state_dict)."""
        try:
            checkpoint = torch.load(self.weights_path, map_location=self.device,
                                     weights_only=False)
            if isinstance(checkpoint, dict):
                # 兼容多种保存格式: model_state_dict / state_dict / params / 裸 state_dict
                state_dict = checkpoint.get('model_state_dict',
                                            checkpoint.get('state_dict',
                                                           checkpoint.get('params',
                                                                          checkpoint.get('model',
                                                                                         checkpoint))))
            else:
                state_dict = checkpoint

            # 处理 DataParallel 前缀
            new_state = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '') if k.startswith('module.') else k
                new_state[new_k] = v

            try:
                self.model.load_state_dict(new_state, strict=True)
            except Exception:
                self.model.load_state_dict(new_state, strict=False)
            print(f"[Info] Loaded EDSR x{self.scale_factor} weights: {self.weights_path}")
        except Exception as e:
            print(f"[Warning] Failed to load EDSR weights: {e}")

    def _pil_to_tensor(self, img):
        """PIL -> [1,C,H,W] tensor in [0,255]."""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        arr = np.array(img).astype(np.float32)
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor

    def _tensor_to_pil(self, tensor):
        """[1,C,H,W] tensor -> PIL."""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.clamp(0, 255).round()
        arr = tensor.byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(arr, 'RGB')

    def _process_tile(self, tensor):
        """单次模型推理."""
        with torch.no_grad():
            tensor = tensor.to(self.device)
            out = self.model(tensor)
        return out

    def _tile_process(self, tensor, tile_size=512, tile_pad=16):
        """分块推理，避免大图 OOM."""
        b, c, h, w = tensor.shape
        if h * w <= tile_size * tile_size:
            return self._process_tile(tensor)

        stride = tile_size - tile_pad
        out_h = h * self.scale_factor
        out_w = w * self.scale_factor
        out = torch.zeros((b, c, out_h, out_w), dtype=torch.float32)

        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y_start = max(0, y - tile_pad) if y > 0 else 0
                x_start = max(0, x - tile_pad) if x > 0 else 0
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)

                tile = tensor[:, :, y_start:y_end, x_start:x_end]
                out_tile = self._process_tile(tile)

                out_y_start = y_start * self.scale_factor
                out_x_start = x_start * self.scale_factor
                out_y_end = y_end * self.scale_factor
                out_x_end = x_end * self.scale_factor

                valid_h = out_y_end - out_y_start
                valid_w = out_x_end - out_x_start
                out[:, :, out_y_start:out_y_end, out_x_start:out_x_end] = \
                    out_tile[:, :, :valid_h, :valid_w]

        return out.clamp(0, 255)

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
