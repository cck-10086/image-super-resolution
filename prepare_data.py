"""准备 DIV2K 训练集.

从 archive 目录随机抽取图像，下采样后生成 LR/HR 对，
划分为 train / val 两个子集，保存到 data_new/.
"""
import os
import argparse
import random
from pathlib import Path

from PIL import Image
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DIV2K training data")
    parser.add_argument("--input", type=str, default="./archive",
                        help="原始数据集目录")
    parser.add_argument("--output", type=str, default="./data_new",
                        help="输出目录")
    parser.add_argument("--train_count", type=int, default=800,
                        help="训练集样本数量")
    parser.add_argument("--val_count", type=int, default=100,
                        help="验证集样本数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def collect_images(input_dir):
    """递归收集所有图像文件."""
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
    images = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                images.append(os.path.join(root, f))
    return images


def downsample_save(src_path, dst_dir, idx, scale=2):
    """生成 HR 图像及其 LR 降采样版本."""
    try:
        img = Image.open(src_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 控制最大尺寸
        max_dim = 1024
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # 这里只复制 HR 图像；训练时 train.py 内部会动态采样 LR patch
        dst_path = os.path.join(dst_dir, f"{idx:04d}.png")
        img.save(dst_path, 'PNG')
        return True
    except Exception as e:
        print(f"[Skip] {src_path}: {e}")
        return False


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.input):
        print(f"[Error] Input dir not found: {args.input}")
        return

    images = collect_images(args.input)
    if not images:
        print(f"[Error] No images found in {args.input}")
        return

    print(f"[Info] Found {len(images)} images in {args.input}")
    random.shuffle(images)

    train_dir = os.path.join(args.output, 'train')
    val_dir = os.path.join(args.output, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    total_needed = args.train_count + args.val_count
    if len(images) < total_needed:
        print(f"[Warning] Need {total_needed} but only {len(images)} available. "
              f"Will use all available and reuse if needed.")

    # 训练集
    print(f"[Info] Generating train set: {args.train_count} images...")
    idx = 1
    saved = 0
    while saved < args.train_count:
        for src_path in images[:args.train_count]:
            if saved >= args.train_count:
                break
            if downsample_save(src_path, train_dir, idx):
                saved += 1
            idx += 1
        if saved < args.train_count:
            # 数据不足，重新洗牌再利用
            random.shuffle(images)

    # 验证集
    print(f"[Info] Generating val set: {args.val_count} images...")
    saved = 0
    while saved < args.val_count:
        for src_path in images[-args.val_count:]:
            if saved >= args.val_count:
                break
            if downsample_save(src_path, val_dir, idx):
                saved += 1
            idx += 1
        if saved < args.val_count:
            random.shuffle(images)

    print(f"[Done] Train: {len(os.listdir(train_dir))} images")
    print(f"[Done] Val: {len(os.listdir(val_dir))} images")


if __name__ == "__main__":
    main()
