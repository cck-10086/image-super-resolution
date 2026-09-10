"""下载 Real-ESRGAN x4plus 官方预训练权重.

权重来源: https://github.com/xinntao/Real-ESRGAN/releases
保存至: ./model_weights/RealESRGAN_x4plus.pth
"""
import os
import sys
import urllib.request

# Real-ESRGAN x4plus 官方权重下载地址
URLS = [
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "https://download.imfile.cn/RealESRGAN_x4plus.pth",
]

OUTPUT_PATH = os.path.join("model_weights", "RealESRGAN_x4plus.pth")


def download():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if os.path.exists(OUTPUT_PATH):
        size = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
        print(f"[Info] Already exists: {OUTPUT_PATH} ({size:.1f} MB)")
        return

    for url in URLS:
        try:
            print(f"[Info] Trying: {url}")
            urllib.request.urlretrieve(url, OUTPUT_PATH)
            size = os.path.getsize(OUTPUT_PATH) / 1024 / 1024
            print(f"[OK] Downloaded to {OUTPUT_PATH} ({size:.1f} MB)")
            return
        except Exception as e:
            print(f"[Fail] {url}: {e}")
            if os.path.exists(OUTPUT_PATH):
                try:
                    os.remove(OUTPUT_PATH)
                except Exception:
                    pass

    print("[Error] All download sources failed.")
    sys.exit(1)


if __name__ == "__main__":
    download()
