"""
YOLO inference benchmark on the Jetson Nano.
    python3 jetson/benchmark.py
Reports average latency and FPS over 20 runs.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.detector import VehicleDetector

RUNS = 20


def main():

    print("Loading YOLO (%s) on %s..." % (config.YOLO_MODEL_PATH, config.DEVICE))
    detector = VehicleDetector(config.YOLO_MODEL_PATH, device=config.DEVICE)
    detector.load_model()
    print("Loaded. Benchmarking %d runs at %dx%d..."
          % (RUNS, config.IMAGE_SIZE, config.IMAGE_SIZE))

    dummy = np.random.randint(
        0, 255, (config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8
    )

    times = []
    for i in range(RUNS):
        t0 = time.time()
        detector.detect(dummy)
        ms = (time.time() - t0) * 1000
        times.append(ms)
        print("  run %2d: %6.1f ms" % (i + 1, ms))

    avg = sum(times) / len(times)
    print("-" * 30)
    print("Average: %.1f ms  =  %.1f FPS" % (avg, 1000.0 / avg))
    print("Min/Max: %.1f / %.1f ms" % (min(times), max(times)))
    print("\nFor traffic control, anything above ~5 FPS is enough")
    print("(a signal decision is only needed every few seconds).")


if __name__ == "__main__":
    main()
