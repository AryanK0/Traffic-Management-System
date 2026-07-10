"""
=========================================================
Vehicle Detector
=========================================================

YOLOv11 Detection Module with SAHI ROI & Soft-NMS

Author : Traffic AI Team
Platform : NVIDIA Jetson Nano
=========================================================
"""

import time
import numpy as np
from ultralytics import YOLO
import config
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def soft_nms(boxes, scores, class_ids, sigma=0.5, Nt=0.3, threshold=0.001, method=2):
    """
    Soft-NMS (Gaussian or Linear) to prevent dense clusters
    of motorcycles from suppressing each other.
    """
    if len(boxes) == 0:
        return boxes, scores, class_ids
        
    boxes = np.array(boxes).astype(float)
    scores = np.array(scores).astype(float)
    class_ids = np.array(class_ids).astype(int)
    
    N = boxes.shape[0]
    for i in range(N):
        maxpos = scores[i:N].argmax()
        maxpos += i
        
        boxes[[i, maxpos]] = boxes[[maxpos, i]]
        scores[[i, maxpos]] = scores[[maxpos, i]]
        class_ids[[i, maxpos]] = class_ids[[maxpos, i]]
        
        tx1, ty1, tx2, ty2 = boxes[i]
        
        pos = i + 1
        while pos < N:
            x1, y1, x2, y2 = boxes[pos]
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + 
                               (x2 - x1 + 1) * (y2 - y1 + 1) - iw * ih)
                    ov = iw * ih / ua
                    if method == 1: # linear
                        weight = 1 - ov if ov > Nt else 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else:
                        weight = 0 if ov > Nt else 1
                        
                    scores[pos] = scores[pos] * weight
                    if scores[pos] < threshold:
                        boxes[[pos, N-1]] = boxes[[N-1, pos]]
                        scores[[pos, N-1]] = scores[[N-1, pos]]
                        class_ids[[pos, N-1]] = class_ids[[N-1, pos]]
                        N -= 1
                        pos -= 1
            pos += 1
    return boxes[:N], scores[:N], class_ids[:N]

class VehicleDetector:

    def __init__(self, model_path, device="cpu"):
        self.model_path = str(model_path)
        self.device = device
        self.detection_model = None
        self.class_names = {0: "2_wheeler", 1: "3_wheeler", 2: "4_wheeler", 3: "unknown"}
        self.loaded = False
        self.inference_time = 0.0
        self.total_frames = 0
        self.total_detections = 0

    def load_model(self):
        # We load via SAHI AutoDetectionModel for slicing
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=self.model_path,
            confidence_threshold=0.01, # Low threshold to let Soft-NMS handle filtering
            device=self.device,
        )
        self.loaded = True

    def detect(self, frame):
        if not self.loaded:
            raise RuntimeError("YOLO model has not been loaded.")

        start = time.time()
        
        # Dynamic SAHI ROI: Crop bottom 60% of the frame to save Jetson memory (road area)
        img_h, img_w = frame.shape[:2]
        crop_y_start = int(img_h * 0.4)
        roi_frame = frame[crop_y_start:, :]
        
        # SAHI Sliced Inference
        result = get_sliced_prediction(
            roi_frame,
            self.detection_model,
            slice_height=320,
            slice_width=320,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type="NMS", 
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.9 # High IOU threshold to pass overlaps to our custom Soft-NMS
        )
        
        boxes = []
        scores = []
        class_ids = []
        for obj in result.object_prediction_list:
            x1, y1, x2, y2 = obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy
            
            # Map coordinates back to full 1080p spatial resolution
            y1 += crop_y_start
            y2 += crop_y_start
            
            boxes.append([x1, y1, x2, y2])
            scores.append(obj.score.value)
            class_ids.append(obj.category.id)
            
        detections = []
        
        if len(boxes) > 0:
            # Apply Soft-NMS
            boxes_nms, scores_nms, class_ids_nms = soft_nms(
                boxes, scores, class_ids, 
                sigma=0.5, Nt=0.3, 
                threshold=config.CONFIDENCE_THRESHOLD, 
                method=2 # Gaussian
            )
            
            for box, score, cls_id in zip(boxes_nms, scores_nms, class_ids_nms):
                x1, y1, x2, y2 = map(int, box)
                
                # HEURISTIC OVERRIDE: Force unknown classes (3) to 2_wheeler (0)
                if cls_id == 3:
                    cls_id = 0
                
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width // 2
                center_y = y1 + height // 2
                
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.class_names.get(cls_id, "unknown"),
                    "confidence": float(score),
                    "bbox": (x1, y1, x2, y2),
                    "center": (center_x, center_y),
                    "width": width,
                    "height": height
                })

        self.inference_time = (time.time() - start)
        self.total_frames += 1
        self.total_detections += len(detections)

        return detections

    def get_inference_time(self):
        return self.inference_time

    def get_total_frames(self):
        return self.total_frames

    def get_total_detections(self):
        return self.total_detections

    def reset_statistics(self):
        self.total_frames = 0
        self.total_detections = 0
        self.inference_time = 0.0

    def reload_model(self):
        self.load_model()

    def is_loaded(self):
        return self.loaded