"""系统测试脚本.

测试项:
1. 依赖包检查
2. 模型权重加载
3. 双三次插值
4. EDSR 推理（2x / 4x / 8x）
5. API 服务状态（若已启动）
"""
import os
import sys
import time
import io

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SuperResolutionModel, bicubic_interpolation


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE = os.path.join(PROJECT_DIR, 'data_new', 'val', '0001.png')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'test_images')


def print_section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_dependencies():
    print_section("1. 依赖检查")
    deps = {}
    try:
        import torch
        deps['torch'] = (torch.__version__, torch.cuda.is_available())
    except Exception as e:
        deps['torch'] = (None, str(e))

    try:
        import flask
        deps['flask'] = (flask.__version__, None)
    except Exception as e:
        deps['flask'] = (None, str(e))

    try:
        import PIL
        deps['pillow'] = (PIL.__version__, None)
    except Exception as e:
        deps['pillow'] = (None, str(e))

    try:
        import numpy
        deps['numpy'] = (numpy.__version__, None)
    except Exception as e:
        deps['numpy'] = (None, str(e))

    for name, (version, info) in deps.items():
        if version is None:
            print(f"  [FAIL] {name}: {info}")
        else:
            extra = f" (CUDA={info})" if info is not None else ""
            print(f"  [ OK ] {name} {version}{extra}")


def test_weights():
    print_section("2. 权重文件检查")
    weights_dir = os.path.join(PROJECT_DIR, 'model_weights')
    if not os.path.isdir(weights_dir):
        print(f"  [WARN] 权重目录不存在: {weights_dir}")
        return

    expected = ['edsr_x2_best.pth', 'edsr_x4_best.pth', 'RealESRGAN_x4plus.pth']
    for f in expected:
        path = os.path.join(weights_dir, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024 / 1024
            print(f"  [ OK ] {f} ({size:.1f} MB)")
        else:
            print(f"  [WARN] {f} 不存在")

    # 列出其他权重
    others = [f for f in os.listdir(weights_dir) if f.endswith('.pth')
              and f not in expected]
    for f in others:
        path = os.path.join(weights_dir, f)
        size = os.path.getsize(path) / 1024 / 1024
        print(f"  [INFO] 额外权重: {f} ({size:.1f} MB)")


def make_test_image():
    """生成测试图像（如果原始测试图不存在）."""
    if os.path.exists(TEST_IMAGE):
        return TEST_IMAGE

    os.makedirs(os.path.dirname(TEST_IMAGE), exist_ok=True)
    arr = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    Image.fromarray(arr, 'RGB').save(TEST_IMAGE, 'PNG')
    return TEST_IMAGE


def test_bicubic(image_path):
    print_section("3. 双三次插值测试")
    try:
        img = Image.open(image_path)
        print(f"  输入尺寸: {img.size}")

        t0 = time.time()
        out = bicubic_interpolation(image_path, 2)
        elapsed = time.time() - t0
        print(f"  2x 输出: {out.size}, 用时: {elapsed*1000:.1f}ms")
        save_path = os.path.join(OUTPUT_DIR, 'bicubic_2x.png')
        out.save(save_path)
        print(f"  保存: {save_path}")

        t0 = time.time()
        out = bicubic_interpolation(image_path, 4)
        elapsed = time.time() - t0
        print(f"  4x 输出: {out.size}, 用时: {elapsed*1000:.1f}ms")
        save_path = os.path.join(OUTPUT_DIR, 'bicubic_4x.png')
        out.save(save_path)
        print(f"  保存: {save_path}")

        print("  [ OK ] 双三次插值测试通过")
    except Exception as e:
        print(f"  [FAIL] {e}")


def test_edsr(image_path):
    print_section("4. EDSR 推理测试")
    try:
        for scale in [2, 4]:
            print(f"\n  --- EDSR x{scale} ---")
            t0 = time.time()
            model = SuperResolutionModel(scale_factor=scale)
            t_load = time.time() - t0
            print(f"  模型加载: {t_load:.2f}s, device={model.device}")

            t0 = time.time()
            out = model.enhance(image_path)
            t_infer = time.time() - t0
            print(f"  推理: {t_infer:.2f}s, 输出尺寸: {out.size}")

            save_path = os.path.join(OUTPUT_DIR, f'edsr_x{scale}.png')
            out.save(save_path)
            print(f"  保存: {save_path}")

            del model
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()


def test_api():
    print_section("5. API 服务测试")
    try:
        import requests
    except ImportError:
        print("  [SKIP] requests 未安装，跳过 API 测试")
        return

    try:
        r = requests.get('http://127.0.0.1:5000/api/status', timeout=5)
        if r.status_code == 200:
            data = r.json()
            print(f"  [ OK ] 服务在线")
            print(f"        device: {data.get('device')}")
            print(f"        models_loaded: {data.get('models_loaded')}")
            print(f"        model_type: {data.get('model_type')}")
        else:
            print(f"  [WARN] 状态码: {r.status_code}")
    except Exception as e:
        print(f"  [INFO] 服务未启动: {e}")
        print("         请先运行: cd backend && python app.py")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print_section("图像超分辨率系统 - 测试")
    print(f"项目目录: {PROJECT_DIR}")

    test_dependencies()
    test_weights()

    image_path = make_test_image()
    print(f"\n测试图像: {image_path}")

    test_bicubic(image_path)
    test_edsr(image_path)
    test_api()

    print_section("测试完成")


if __name__ == "__main__":
    main()
