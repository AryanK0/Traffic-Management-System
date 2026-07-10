# Traffic Management AI (Jetson Deployment Pipeline)

This repository contains the full hardware-in-the-loop Smart Traffic Light Controller using **YOLOv11** (vehicle detection) and **Deep Q-Network (DQN)** reinforcement learning (signal control) on the **NVIDIA Jetson Nano**.

## Project Architecture & Structure

The repository has been fully restructured for deployment. All experimental, training, and data-processing scripts have been archived safely.

```text
.
├── core/                   # Core AI modules
│   ├── detector.py         # YOLOv11 wrapper and vision handling
│   ├── dqn_model.py        # PyTorch DQN neural network architecture
│   ├── dqn_controller.py   # State formulation and AI logic
│   └── traffic_manager.py  # Orchestrator
├── sumo/                   # Simulation interface (Laptop/Host side)
│   ├── traci_client.py     # TraCI communication
│   ├── runner.py           # SUMO simulation launcher
│   └── sumo_config.sumocfg # Traffic network definition
├── jetson/                 # Hardware specific deployment scripts
│   ├── install.sh          # JetPack dependencies
│   ├── check_gpu.py        # CUDA validation
│   └── benchmark.py        # TensorRT / ONNX benchmarking
├── utils/                  # Logging and helper utilities
├── models/                 # Pretrained weights (best.pt, dqn_traffic_model.pth)
├── videos/                 # Video feed sources (e.g. traffic.mp4)
├── outputs/                # Logs and annotated videos
├── data/                   # Large datasets (e.g., master_dataset_640)
├── research_and_training/  # Archive of original training, data processing, & presentation scripts
├── main.py                 # Live vision pipeline entry point (Jetson)
├── main_sumo.py            # Hardware-in-the-loop demo entry point (Jetson)
└── system_controller.py    # Standalone system controller script
```

## Setup on Jetson Nano (JetPack 4.6, Python 3.6+)

1. **Install dependencies:**
```bash
bash jetson/install.sh
```
2. **Verify CUDA and TensorRT:**
```bash
python3 jetson/check_gpu.py     # MUST output CUDA: True
python3 jetson/benchmark.py     # Verify inference latency
```

## Running the System

### 1. Standalone Vision Pipeline
To run just the vision pipeline (reading from `videos/traffic.mp4` or falling back to live camera):
```bash
python3 main.py
```

### 2. Full Hardware-in-the-Loop Simulation (DQN + SUMO)
This connects the Jetson hardware running the PyTorch DQN agent directly to a SUMO simulation (typically running on a laptop/workstation).

1. **Start the simulation environment (Laptop):**
```bash
python3 sumo/runner.py --jetson
```
2. **Launch the AI agent (Jetson):**
```bash
python3 main_sumo.py <laptop-ip>  # e.g., 192.168.55.100 if using USB tethering
```

To run everything locally on one machine for testing without the Jetson hardware:
```bash
python3 sumo/runner.py
```

## Reinforcement Learning Details (DQN)

The state representation and actions have been explicitly mapped to the custom SUMO environment and must not be altered unless the model is retrained:

- **State Space (6 dimensions):** 
  - `[North, South, East, West]` queue lengths (normalized by 100).
  - Intersection spillback count (normalized by 100).
  - Time elapsed in current green phase (normalized by 60s max limit).
- **Action Space (2 discrete actions):**
  - `0`: KEEP current phase.
  - `1`: SWITCH to next phase.

*Safety Failsafes:* Hardcoded overrides guarantee a minimum green time of 5s, a maximum green time of 60s, and standard 4s yellow transition phases.

## Re-Training & Archival Scripts
The `research_and_training/` folder contains the legacy scripts used to build and evaluate this system:
- `train_yolov11.py` & `train_microbatch.py`
- `resize_dataset.py`, `health_scrubber.py`, etc.
- `demos/presentation_demo.py` (The sequential presentation demo)

Use these scripts if you intend to collect more data and retrain the models from scratch.
