#!/bin/bash
# =========================================================
# Jetson Nano install script (JetPack 4.6.x / Ubuntu 18.04)
# Run:  bash jetson/install.sh
# Needs internet on the Jetson first.
# =========================================================
set -e

echo "== [1/5] System packages =="
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-numpy \
    libopencv-dev python3-opencv nano git wget \
    libopenblas-base libopenmpi-dev

echo "== [2/5] Cython =="
pip3 install Cython

echo "== [3/5] PyTorch (NVIDIA Jetson wheel — NOT pip torch) =="
if python3 -c "import torch" 2>/dev/null; then
    echo "PyTorch already installed, skipping."
else
    wget -nc https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl \
        -O torch-1.10.0-cp36-cp36m-linux_aarch64.whl
    pip3 install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
fi

echo "== [4/5] Ultralytics (YOLO) =="
pip3 install ultralytics || {
    echo "Full install failed, trying minimal install..."
    pip3 install "ultralytics==8.0.145" --no-deps
    pip3 install pyyaml tqdm requests pillow
}

echo "== [5/5] Verify =="
python3 jetson/check_gpu.py

echo ""
echo "Install complete. Next:  python3 jetson/benchmark.py"
