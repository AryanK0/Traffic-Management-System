# IoT-Driven Smart Traffic Management System Data Pipeline

This repository contains a modular Python data pipeline designed to ingest, unify, and augment multiple heterogeneous datasets (IDD, FGVD, i2wdd, DATS 2022) into a single YOLOv8/v10/v11 format dataset. It is optimized for Jetson Nano edge deployment by consolidating complex taxonomies into 5 macro categories.

## Architecture and Execution Order

The pipeline is split into clean, modular `.py` files to maintain production readiness without excessive commenting.

### 1. `config.py`
**Purpose**: Centralized configuration file.
- Contains the absolute paths to input directories and the unified output structure (`master_dataset/`).
- Defines the core taxonomy mapping (`CLASS_MAPPING`) used to condense hundreds of granular dataset labels into 5 key YOLO classes.

### 2. `parsers.py`
**Purpose**: Universal Bounding Box Parsing.
- Contains `parse_xml()`, `parse_json()`, and `parse_txt()` functions.
- Extracts disparate annotation formats (e.g., Pascal VOC, YOLO TXT, JSON) into a unified Python dictionary: `{"label": str, "bbox": [xmin, ymin, xmax, ymax]}`.

### 3. `yolo_utils.py`
**Purpose**: YOLO Math and Normalization.
- Converts absolute bounding box coordinates into YOLO normalized coordinates `[center_x, center_y, width, height]`.
- Implements safety limits by clipping values between `0.0` and `1.0` to prevent out-of-bounds training errors.

### 4. `taxonomy.py`
**Purpose**: Taxonomy Standardization & Active Learning Loop.
- Maps raw, chaotic string labels from datasets (like `car_MarutiSuzuki_Ciaz`) to exactly one of 5 macro categories using intelligent substring matching.
- **Active Learning Component**: If a string doesn't match known definitions, it defaults to class `4` (unknown). It then dynamically logs the image path and raw label into a `unknown_classes_log.csv` file for manual review later.

### 5. `augment.py`
**Purpose**: Data Augmentation mimicking Indian roads.
- Uses `albumentations` to apply Random Brightness, Contrast, Advanced Blur, Motion Blur, Fog, and Rain.
- Only transforms images in the `train` split and calculates the precise updated bounding boxes after spatial alterations.

### 6. `main.py`
**Purpose**: Master Execution Script.
- Iterates through the root datasets, finds matches for images and annotations, and invokes parsers.
- Preforms an 80/20 Train/Validation Split.
- Moves and renames images with their parent dataset prefix (e.g., `idd_img001.jpg`) to prevent file-overwrite collisions.
- Writes the converted YOLO `.txt` labels to the output directories.
- Finally, triggers `augment_dataset()` on the master training set.

---

## Taxonomy Rules

Indian traffic is chaotic. The bounding box labels are standardized as follows:
- **0: `2_wheeler`** (motorcycle, bike, scooter, etc.)
- **1: `3_wheeler`** (autorickshaw, auto, e-rickshaw)
- **2: `4_wheeler`** (car, taxi, suv, van)
- **3: `heavy_vehicle`** (bus, truck, tractor)
- **4: `unknown`** (CRITICAL: Any unrecognizable string becomes class 4, preventing `-1` errors in YOLO training and facilitating RL feedback loops).

---

## How to Run on a Windows Terminal

1. **Install Prerequisites**:
   Ensure you have Python installed, then install the required dependencies:
   ```powershell
   pip install opencv-python albumentations
   ```

2. **Prepare Datasets**:
   Place your raw datasets inside a `dataset/` directory in the root of this project:
   - `dataset/idd/`
   - `dataset/fgvd/`
   - `dataset/i2wdd/`
   - `dataset/dats/`

3. **Execute the Pipeline**:
   Run the master orchestrator script.
   ```powershell
   python main.py
   ```

4. **Review Outputs**:
   - The unified dataset will be built in the `master_dataset/` folder.
   - You can use the included `data.yaml` to begin your YOLO training immediately.
   - Review the `unknown_classes_log.csv` for any edge-case classifications that need manual tagging.
