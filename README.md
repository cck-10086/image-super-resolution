﻿# 图像超分辨率放大系统

基于 EDSR + Real-ESRGAN 的深度学习图像超分辨率系统，支持 2x / 4x / 8x 放大。

## 项目结构

```
├── backend/                  # Flask 后端服务
│   └── app.py                # API 入口
├── frontend/                 # Web 前端
│   ├── templates/index.html
│   └── static/
│       ├── css/style.css
│       └── js/main.js
├── model/                    # 模型模块
│   ├── edsr.py               # EDSR 网络定义
│   ├── rrdbnet.py            # RRDBNet (Real-ESRGAN) 网络定义
│   ├── utils.py              # 推理工具 (含分块处理)
│   └── __init__.py
├── model_weights/            # 模型权重
│   ├── edsr_x2_best.pth      # EDSR 2x (DIV2K 自训练)
│   ├── edsr_x4_best.pth      # EDSR 4x (DIV2K 自训练)
│   └── RealESRGAN_x4plus.pth # Real-ESRGAN 4x (官方预训练)
├── data_new/                 # 训练数据集
│   ├── train/                # 800 张
│   └── val/                  # 100 张
├── train.py                  # 训练脚本
├── prepare_data.py           # 数据准备脚本
├── test_system.py            # 系统测试
├── config.ini                # 配置文件
├── requirements.txt          # 依赖
└── README.md
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
cd backend
python app.py
```

### 3. 访问界面

打开浏览器访问：http://localhost:5000

## 放大方案

| 倍率 | 模型 | 说明 |
|------|------|------|
| **2x** | EDSR | DIV2K 训练，PSNR 33.24 |
| **4x** | Real-ESRGAN x4plus | 官方预训练权重，效果最佳 |
| **8x** | Real-ESRGAN 4x + EDSR 2x | 级联放大，复用优质模型 |

## 训练模型

```bash
# 训练 2x 模型
python train.py --data_dir ./data_new --scale 2 --epochs 100 --batch_size 16

# 训练 4x 模型
python train.py --data_dir ./data_new --scale 4 --epochs 100 --batch_size 16
```

## 数据准备

```bash
# 从原始 DIV2K 数据准备训练集
python prepare_data.py --input ./archive --output ./data_new --train_count 800 --val_count 100
```

## 系统测试

```bash
python test_system.py
```

## API 接口

### 图像增强

**POST** `/api/enhance`

| 参数 | 类型 | 说明 |
|------|------|------|
| image | File | 上传图像 |
| scale | int | 放大倍率 (2/4/8) |
| method | str | 处理方法 (edsr/bicubic) |

### 服务状态

**GET** `/api/status`

返回设备信息、模型类型、加载状态。

## 技术栈

- **框架**: Flask + PyTorch
- **模型**: EDSR + Real-ESRGAN (RRDBNet)
- **前端**: HTML5 + CSS3 + JavaScript (暗色主题)
- **GPU**: CUDA 加速

## 注意事项

1. 首次启动时会自动加载模型权重，约需 3-5 秒
2. 大尺寸图像 8x 放大使用分块推理，可能需要 20-30 秒
3. 建议使用 CUDA 设备以获得最佳性能