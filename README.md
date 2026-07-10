# 🚦 Smart Traffic Management System (YOLO + DQN)

Welcome to the **Smart Traffic Management System**, a full hardware-in-the-loop traffic control pipeline combining state-of-the-art computer vision (YOLO) and Reinforcement Learning (Deep Q-Network). This project is specifically designed to handle the complexities of Indian traffic (heavy occlusion, high volume of 2-wheelers and 3-wheelers) and dynamically optimize traffic signal timings in a SUMO simulation environment. The inference pipeline is highly optimized for edge deployment on devices like the **NVIDIA Jetson Nano** (TensorRT FP16 compatible).

---

## 📖 Project Overview

Traditional traffic lights operate on fixed timers, leading to inefficient traffic flow, congestion, and increased emissions. This project introduces a dynamic, AI-driven approach:
1. **Perception**: A highly customized YOLO architecture processes live video feeds to detect and classify vehicles (2-wheelers, 3-wheelers, 4-wheelers) in real-time.
2. **Decision**: A Deep Q-Network (DQN) reinforcement learning agent takes the aggregated vehicle counts and queue lengths as state inputs to dynamically decide whether to `KEEP` or `SWITCH` the current traffic signal phase.
3. **Simulation**: The system interfaces directly with the SUMO (Simulation of Urban MObility) engine to simulate real-world traffic intersections and evaluate the agent's performance.

---

## 📊 Dataset

The core dataset used to train the vision models is heavily balanced to prioritize traditionally occluded objects like 2-wheelers and 3-wheelers. Images have been scaled to a highly optimized 640x640 resolution to maximize mAP while ensuring real-time edge performance.

- **Download Link**: [Traffic Dataset (640px) - Google Drive](https://drive.google.com/file/d/14agFUtavCnNSWXXxJ13kIy3gMMHv5hMI/view?usp=sharing)
- **Size**: ~25.1 GB (uncompressed)
- **Format**: YOLO format (Images + `.txt` labels)

*Note: Extract the dataset into the `data/master_dataset_640/` directory before running the training scripts.*

---

## 🧠 Architecture & Methodology

### 1. Vision Architecture (Custom YOLO11)
To solve the challenge of detecting small, heavily occluded vehicles, we implemented a custom YOLO11 architecture (`models/yolo11-custom.yaml`):
- **P2 Detection Head**: Added a 4th detection head at the P2 level (high-resolution feature map) specifically engineered to detect extremely small 2-wheelers.
- **CBAM Injection**: Convolutional Block Attention Modules (CBAM) are injected into the backbone and neck to help the network focus on salient spatial and channel features. These are implemented using standard PyTorch operations to guarantee TensorRT compilation.
- **Knowledge Distillation (KD)**: During training (`train_kd.py`), we use KL divergence to distill knowledge from a larger teacher model to a highly efficient student model.

### 2. Inference Pipeline Enhancements
- **Dynamic SAHI ROI**: Instead of passing the full 1080p frame, the detector dynamically crops the bottom 60% Region of Interest (ROI) and applies Slicing Aided Hyper Inference (SAHI) tiled processing.
- **Vector-based Soft-NMS**: Implemented Gaussian Soft-NMS on the collected tile detections before re-mapping coordinates back to the global frame.

### 3. Reinforcement Learning (DQN)
The traffic controller is an autonomous RL agent interacting via TraCI (Traffic Control Interface).
- **State Space (6-dim)**: 
  - `[North, South, East, West]` lane queue lengths (normalized by 100).
  - Intersection spillback count (normalized by 100).
  - Time elapsed in the current green phase (normalized by 60s max limit).
- **Action Space (2-dim)**:
  - `0`: KEEP current phase.
  - `1`: SWITCH to next phase.
- **Reward Function**: Heavily penalizes high queue lengths, total waiting time, and frequent unnecessary signal switches.

---

## 📁 Repository Structure

```text
.
├── core/                   # Core AI modules (detector, state_builder, dqn_model, traffic_manager)
├── sumo/                   # Simulation interface (TraCI client, SUMO config, network files)
├── jetson/                 # Hardware specific deployment scripts (install, benchmark)
├── models/                 # Pretrained weights (best.pt, dqn_traffic_model.pth)
├── data/                   # Large datasets directory (Extract dataset here)
├── research_and_training/  # Training scripts (YOLO, DQN, Distillation, Evaluation)
├── utils/                  # Logging and helper utilities
├── main.py                 # Live vision pipeline entry point (Jetson / PC)
├── main_sumo.py            # Hardware-in-the-loop demo entry point (DQN + SUMO)
└── system_controller.py    # Standalone system controller script
```

---

## 🚀 How to Run

### Setup Dependencies
1. Clone the repository and install the Python requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. If deploying on an **NVIDIA Jetson Nano**:
   ```bash
   bash jetson/install.sh
   python3 jetson/check_gpu.py
   ```

### 1. Standalone Vision Pipeline
To test the YOLO vehicle detection locally on a test video or live feed:
```bash
python3 main.py
```

### 2. Full Hardware-in-the-Loop Simulation (DQN + SUMO)
This connects the PyTorch DQN agent directly to the SUMO simulation.

1. **Start the SUMO simulation environment:**
   ```bash
   python3 sumo/runner.py
   ```
2. **Launch the AI agent:**
   *(In a new terminal window)*
   ```bash
   python3 main_sumo.py
   ```
   *Note: If running the simulation on a host PC and the AI on a Jetson, pass the host IP address to `main_sumo.py`.*

### 3. Training the Models
All training and evaluation scripts have been carefully organized in the `research_and_training/` directory. Ensure you have downloaded and extracted the Google Drive dataset to `data/master_dataset_640/` before initiating a training run.

- **Train Custom YOLO**: `python3 research_and_training/train_yolov11.py`
- **Train via Knowledge Distillation**: `python3 research_and_training/train_kd.py`
- **Train DQN Agent**: `python3 research_and_training/train_dqn.py`

---
*Built to optimize traffic flow, reduce emissions, and run efficiently at the edge.*
