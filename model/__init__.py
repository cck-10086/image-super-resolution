"""超分辨率模型模块.

导出:
- SuperResolutionModel: EDSR 推理包装器 (提供 enhance 接口)
- bicubic_interpolation: 双三次插值函数
- EDSR, create_edsr, get_architecture_for_scale: EDSR 网络相关
"""
from .edsr import EDSR, create_edsr, get_architecture_for_scale
from .utils import SuperResolutionModel, bicubic_interpolation

__all__ = [
    'SuperResolutionModel',
    'bicubic_interpolation',
    'EDSR',
    'create_edsr',
    'get_architecture_for_scale',
]
