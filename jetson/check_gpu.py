"""
Jetson GPU sanity check — run this FIRST on the Nano.
    python3 jetson/check_gpu.py
"""

import sys

print("Python :", sys.version.split()[0])

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA   :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU    :", torch.cuda.get_device_name(0))
    else:
        print("WARNING: CUDA not available — YOLO will run on CPU (slow).")
        print("Did you install NVIDIA's Jetson PyTorch wheel (not pip torch)?")
except ImportError:
    print("PyTorch is NOT installed. Install the Jetson wheel first.")

try:
    import cv2
    print("OpenCV :", cv2.__version__)
except ImportError:
    print("OpenCV is NOT installed. Run: sudo apt-get install python3-opencv")

try:
    import ultralytics
    print("Ultralytics:", ultralytics.__version__)
except ImportError:
    print("Ultralytics is NOT installed. Run: pip3 install ultralytics")
