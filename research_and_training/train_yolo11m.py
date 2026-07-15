import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn import modules, tasks

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Strictly uses Conv2d, BatchNorm2d, and Sigmoid for TensorRT compatibility.
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

def main():
    print("🚀 Initializing Custom YOLO11m Model...")
    # Initialize the model using the custom YOLO11m architecture layout
    model = YOLO("../models/yolo11m-custom.yaml").load("yolo11m.pt")
    
    print("🔥 Starting High-Resolution Training...")
    # Execute the train() function with specified hyperparameter overrides
    model.train(
        data="data.yaml", 
        epochs=150,
        imgsz=1280,
        batch=16,
        project="runs/detect",
        name="train-medium-highres",
        device="0" if torch.cuda.is_available() else "cpu"
    )

if __name__ == "__main__":
    main()
