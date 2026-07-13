# Deployment configuration
"""
=============================================================
Traffic AI using YOLOv11 + DQN + SUMO
Deployment Configuration

Target Platform:
    NVIDIA Jetson Nano

Author:
    Team Project

Description:
    Global configuration used by every module in the project.
=============================================================
"""

from pathlib import Path
import torch

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent

# ============================================================
# DIRECTORIES
# ============================================================

MODELS_DIR = ROOT / "models"
VIDEOS_DIR = ROOT / "videos"
OUTPUT_DIR = ROOT / "outputs"

SUMO_DIR = ROOT / "sumo"

SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"

# Automatically create output folders
OUTPUT_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)

# ============================================================
# MODEL FILES
# ============================================================

YOLO_MODEL_PATH = MODELS_DIR / "best.pt"

DQN_MODEL_PATH = MODELS_DIR / "dqn_traffic_model.pth"

# Optional
CLASS_FILE = MODELS_DIR / "classes.txt"

# ============================================================
# VIDEO
# ============================================================

INPUT_VIDEO = VIDEOS_DIR / "traffic.mp4"

OUTPUT_VIDEO = OUTPUT_DIR / "output.mp4"

SIMULATION_VIDEO = OUTPUT_DIR / "simulation.mp4"

LOG_FILE = OUTPUT_DIR / "traffic_log.csv"

# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# YOLO CONFIGURATION
# ============================================================

IMAGE_SIZE = 640

CONFIDENCE_THRESHOLD = 0.30

IOU_THRESHOLD = 0.45

USE_HALF_PRECISION = True

# ============================================================
# VIDEO CONFIGURATION
# ============================================================

FRAME_SKIP = 1

DISPLAY_OUTPUT = True

SAVE_OUTPUT = True

SHOW_FPS = True

# ============================================================
# DQN CONFIGURATION
# ============================================================

STATE_SIZE = 6

ACTION_SIZE = 2

MAX_GREEN_TIME = 60

MIN_GREEN_TIME = 5

YELLOW_TIME = 4

# ============================================================
# SUMO CONFIGURATION
# ============================================================

SUMO_CONFIG = SUMO_DIR / "sumo_config.sumocfg"

SUMO_BINARY = "sumo-gui"

USE_SUMO_GUI = False

TRAFFIC_LIGHT_ID = "center"

# Incoming edges
INCOMING_EDGES = {
    "north": "n_in",
    "south": "s_in",
    "east": "e_in",
    "west": "w_in"
}

# Outgoing edges
OUTGOING_EDGES = {
    "north": "n_out",
    "south": "s_out",
    "east": "e_out",
    "west": "w_out"
}

# ============================================================
# PERFORMANCE
# ============================================================

MAX_FPS = 15

BENCHMARK = False

SAVE_SCREENSHOTS = False

# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = "INFO"

PRINT_DETECTIONS = False

PRINT_DQN_STATE = False

PRINT_SIGNAL_CHANGES = True

# ============================================================
# LANE REGIONS  (x1, y1, x2, y2) in pixels
#
# Defined for the REFERENCE resolution below; scaled
# automatically to the actual frame size by LaneCounter.
# "center" = the junction box, used for spillback.
#
# IMPORTANT:
# These are DEFAULT grid values. Update them after the
# traffic video is placed inside videos/traffic.mp4
# (or once the real camera is mounted).
# ============================================================

REFERENCE_WIDTH = 1280

REFERENCE_HEIGHT = 720

LANE_REGIONS = {

    "north": (426, 0, 853, 240),

    "south": (426, 480, 853, 720),

    "east": (853, 240, 1280, 480),

    "west": (0, 240, 426, 480),

    "center": (426, 240, 853, 480)

}

# ============================================================
# COLORS
# ============================================================

GREEN = (0,255,0)

RED = (0,0,255)

BLUE = (255,0,0)

YELLOW = (0,255,255)

WHITE = (255,255,255)

BLACK = (0,0,0)

# ============================================================
# FONT
# ============================================================

FONT_SCALE = 0.6

FONT_THICKNESS = 2

# ============================================================
# STARTUP CHECK
# ============================================================

def validate_paths():

    required_files = [
        YOLO_MODEL_PATH,
        DQN_MODEL_PATH,
        INPUT_VIDEO,
        SUMO_CONFIG
    ]

    missing = []

    for file in required_files:
        if not file.exists():
            missing.append(file)

    return missing


def print_configuration():

    print("\n==============================================")
    print(" Traffic AI Deployment Configuration")
    print("==============================================")

    print(f"Device           : {DEVICE}")
    print(f"YOLO Model       : {YOLO_MODEL_PATH}")
    print(f"DQN Model        : {DQN_MODEL_PATH}")
    print(f"Input Video      : {INPUT_VIDEO}")
    print(f"SUMO Config      : {SUMO_CONFIG}")

    print("==============================================\n")