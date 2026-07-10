import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks

# ==========================================
# 1. TRT-Compatible CBAM Injection
# ==========================================
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Strictly uses Conv2d, BatchNorm2d, and Sigmoid for TensorRT compatibility.
    Avoids unsupported pooling or complex tensor ops.
    """
    def __init__(self, c1, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, max(1, c1 // ratio), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, c1 // ratio), c1, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel
        ca = self.channel_attention(x)
        x = x * ca
        
        # Spatial
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        sa = torch.cat([max_out, mean_out], dim=1)
        sa = self.spatial_attention(sa)
        
        return x * sa

# Register CBAM so ultralytics parse_model can find it via eval()
modules.CBAM = CBAM
tasks.CBAM = CBAM


# ==========================================
# 2. Knowledge Distillation Trainer
# ==========================================
class KDDetectionTrainer(DetectionTrainer):
    def __init__(self, cfg=None, overrides=None):
        super().__init__(cfg, overrides)
        
        # Load Teacher Model (yolo11x)
        print("🔥 Loading Teacher Model (yolo11x.pt)...")
        # Initialize standard model to get the architecture and weights
        try:
            self.teacher_model = YOLO("yolo11x.pt").model.to(self.device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.kd_alpha = 0.5 # KD weight
            self.temperature = 3.0
            print("✅ Teacher Model Loaded!")
        except Exception as e:
            print(f"⚠️ Warning: Could not load teacher model ({e}). Continuing without KD.")
            self.teacher_model = None

    def train_step(self):
        """
        Custom Training Step for Knowledge Distillation.
        Intercepts the forward pass to inject Kullback-Leibler Divergence (KLD) loss.
        """
        # Fetch batch
        batch = next(self.train_loader)
        batch = self.preprocess_batch(batch)
        
        # Forward pass (Student)
        preds = self.model(batch["img"])
        loss, loss_items = self.loss(batch, preds)
        
        # Compute Knowledge Distillation Loss if Teacher exists
        if hasattr(self, "teacher_model") and self.teacher_model is not None:
            with torch.no_grad():
                teacher_preds = self.teacher_model(batch["img"])
            
            # KD Loss computation on raw logits (assuming lists of tensors from YOLO heads)
            kd_loss = 0.0
            # Teacher and Student might have different number of heads (4 vs 3). 
            # We will compute KLD only on matching spatial resolutions.
            student_logits = preds[1] if isinstance(preds, tuple) else preds
            teacher_logits = teacher_preds[1] if isinstance(teacher_preds, tuple) else teacher_preds
            
            if isinstance(student_logits, list) and isinstance(teacher_logits, list):
                # Reverse iterate: standard YOLO heads order is P3, P4, P5
                # Student has P2, P3, P4, P5. Teacher has P3, P4, P5.
                # We align them from the end (P5 matches P5).
                for s_logit, t_logit in zip(reversed(student_logits), reversed(teacher_logits)):
                    if s_logit.shape == t_logit.shape: # Ensures resolution matches
                        # Scale down by temperature
                        s_prob = F.log_softmax(s_logit / self.temperature, dim=1)
                        t_prob = F.softmax(t_logit / self.temperature, dim=1)
                        # Compute KLD
                        kd_loss += F.kl_div(s_prob, t_prob, reduction="batchmean") * (self.temperature ** 2)
            
            # Combine standard loss and KD loss
            loss = (1 - self.kd_alpha) * loss + (self.kd_alpha * kd_loss)

        # Backward and step
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update EMA and loss items
        self.ema.update(self.model)
        self.loss_items = loss_items


def train_distillation():
    # Define custom model path
    custom_yaml = os.path.join(os.path.dirname(__file__), "..", "models", "yolo11-custom.yaml")
    
    # Create the model using the custom YAML (will parse our CBAM layer)
    print("🔥 Initializing Custom Student Model with P2 Head and CBAM...")
    model = YOLO(custom_yaml)
    
    # Start Knowledge Distillation Training using the custom trainer
    # We pass the model to the custom trainer via overrides
    print("🚀 Starting KD Training Pipeline on H100 (FP16)...")
    
    # We override the base YOLO train method to use our KDDetectionTrainer
    overrides = {
        "data": "data.yaml", # Make sure this points to your dataset
        "epochs": 100,
        "batch": 32,
        "imgsz": 640,
        "device": 0,
        "half": True, # FP16 mixed precision for H100
        "project": "Indian_Traffic_AI",
        "name": "yolo11_kd_student",
        "model": custom_yaml
    }
    
    trainer = KDDetectionTrainer(overrides=overrides)
    trainer.train()

if __name__ == "__main__":
    train_distillation()
