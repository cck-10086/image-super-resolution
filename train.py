"""EDSR 训练脚本.

使用 DIV2K 数据集训练 EDSR 超分辨率模型，支持 2x/4x 倍率.
采用 L1 损失，Adam 优化器，StepLR 学习率调度.

示例:
    python train.py --data_dir ./data_new --scale 2 --epochs 100 --batch_size 16
    python train.py --data_dir ./data_new --scale 4 --epochs 100 --batch_size 16
"""
import os
import sys
import time
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from PIL import Image

# 让本脚本能直接导入 model 包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.edsr import EDSR, create_edsr, get_architecture_for_scale


class DIV2KDataset(Dataset):
    """DIV2K 风格数据集：从 HR 图像动态采样 LR/HR patch 对."""

    def __init__(self, hr_dir, scale=2, patch_size=96, augment=True):
        self.hr_dir = hr_dir
        self.scale = scale
        self.patch_size = patch_size
        self.augment = augment
        self.image_files = sorted([
            f for f in os.listdir(hr_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        if not self.image_files:
            raise RuntimeError(f"No images found in {hr_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.hr_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')

        # 确保尺寸足够
        hr_p = self.patch_size * self.scale
        if img.width < hr_p or img.height < hr_p:
            img = img.resize((max(hr_p, img.width), max(hr_p, img.height)),
                             Image.BICUBIC)

        # 随机裁剪 HR patch
        x = random.randint(0, img.width - hr_p)
        y = random.randint(0, img.height - hr_p)
        hr_patch = img.crop((x, y, x + hr_p, y + hr_p))

        # 下采样得到 LR patch
        lr_patch = hr_patch.resize(
            (self.patch_size, self.patch_size), Image.BICUBIC
        )

        # 数据增强
        if self.augment:
            hr_patch, lr_patch = self._augment(hr_patch, lr_patch)

        # 转 tensor [0, 255]
        hr_t = self._to_tensor(hr_patch)
        lr_t = self._to_tensor(lr_patch)
        return lr_t, hr_t

    def _augment(self, hr, lr):
        # 90 度旋转 + 水平翻转
        k = random.randint(0, 3)
        hr = hr.rotate(90 * k, expand=False)
        lr = lr.rotate(90 * k, expand=False)
        if random.random() < 0.5:
            hr = hr.transpose(Image.FLIP_LEFT_RIGHT)
            lr = lr.transpose(Image.FLIP_LEFT_RIGHT)
        return hr, lr

    @staticmethod
    def _to_tensor(img):
        arr = np.array(img).astype(np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train EDSR")
    parser.add_argument("--data_dir", type=str, default="./data_new")
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--n_resblocks", type=int, default=32)
    parser.add_argument("--n_feats", type=int, default=256)
    parser.add_argument("--res_scale", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="model_weights")
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--lr_step", type=int, default=30)
    parser.add_argument("--lr_gamma", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Info] Device: {device}")

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')

    if not os.path.isdir(train_dir):
        print(f"[Error] Train dir not found: {train_dir}")
        sys.exit(1)

    # 数据集
    train_set = DIV2KDataset(train_dir, scale=args.scale,
                              patch_size=args.patch_size)
    val_set = None
    if os.path.isdir(val_dir):
        val_set = DIV2KDataset(val_dir, scale=args.scale,
                               patch_size=args.patch_size, augment=False)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device == 'cuda'),
        drop_last=True
    )
    print(f"[Info] Train samples: {len(train_set)}, batches: {len(train_loader)}")

    # 模型
    model = EDSR(n_resblocks=args.n_resblocks, n_feats=args.n_feats,
                 res_scale=args.res_scale, scale=args.scale).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] EDSR x{args.scale} parameters: {n_params/1e6:.2f}M")

    # 优化器 & 调度器 & 损失
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    criterion = nn.L1Loss()

    os.makedirs(args.save_dir, exist_ok=True)
    best_psnr = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (lr, hr) in enumerate(train_loader):
            lr = lr.to(device, non_blocking=True)
            hr = hr.to(device, non_blocking=True)

            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}")

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0
        print(f"[Epoch {epoch}/{args.epochs}] avg_loss={avg_loss:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e} time={elapsed:.1f}s")

        # 验证
        if val_set is not None and epoch % 5 == 0:
            psnr = evaluate(model, val_set, args, device)
            print(f"  [Val] PSNR: {psnr:.2f} dB")
            if psnr > best_psnr:
                best_psnr = psnr
                save_path = os.path.join(args.save_dir,
                                          f'edsr_x{args.scale}_best.pth')
                torch.save(model.state_dict(), save_path)
                print(f"  [Save] Best model -> {save_path} (PSNR={psnr:.2f})")

        # 定期保存
        if epoch % args.save_interval == 0:
            save_path = os.path.join(args.save_dir,
                                      f'edsr_x{args.scale}_epoch{epoch}.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  [Save] Checkpoint -> {save_path}")

        scheduler.step()

    # 训练结束保存最终模型
    final_path = os.path.join(args.save_dir, f'edsr_x{args.scale}_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"[Done] Final model -> {final_path}")
    if best_psnr > 0:
        print(f"[Done] Best PSNR: {best_psnr:.2f} dB")


def evaluate(model, val_set, args, device, max_batches=20):
    """计算 PSNR."""
    model.eval()
    psnr_sum = 0.0
    count = 0

    with torch.no_grad():
        for i in range(min(len(val_set), max_batches)):
            lr, hr = val_set[i]
            lr = lr.unsqueeze(0).to(device)
            hr = hr.unsqueeze(0).to(device)
            sr = model(lr).clamp(0, 255)

            mse = torch.mean((sr - hr) ** 2).item()
            if mse < 1e-10:
                psnr = 100.0
            else:
                psnr = 10 * np.log10(255 * 255 / mse)
            psnr_sum += psnr
            count += 1

    model.train()
    return psnr_sum / max(count, 1)


if __name__ == "__main__":
    main()
