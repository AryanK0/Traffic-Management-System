from ultralytics import YOLO
import os

def main():
    print("🚀 Initializing YOLOv11-nano Training Pipeline on Server...")
    
    # 1. Load the base model (the 'weights' downloaded by setup_yolo.py)
    # yolo11n.pt is the nano architecture optimized for Jetson Edge devices
    model = YOLO("yolo11n.pt") 

    # 2. Define the exact path to your data.yaml
    # Note: Make sure this points to wherever you upload the dataset on the Thapar server
    dataset_config = "master_dataset/data.yaml"
    
    # 3. Define the path to the custom Indian Traffic hyperparameters
    hyperparameters = "indian_traffic_hyp.yaml"

    # 4. Start the Training Engine
    # batch=64 : Takes advantage of the 32GB VRAM to train incredibly fast
    # epochs=100 : Good starting point; YOLO has early stopping if it finishes learning early
    # imgsz=640 : Standard resolution for YOLOv11
    # device=0 : Tells it to use the first available NVIDIA GPU
    
    print("🔥 Beginning Training Loop...")
    results = model.train(
        data=dataset_config,
        cfg=hyperparameters, # Injects the heavy mixup/mosaic augmentations
        epochs=100,
        batch=64, # Change to 128 if the 32GB GPU allows it without out-of-memory errors
        imgsz=640,
        device=0, 
        project="Indian_Traffic_AI",
        name="yolo11n_uvh26_idd",
        save=True, # Saves the best weights
        plots=True, # Generates confusion matrices and F1-score charts
        image_weights=True # Applies class weighting by sampling rare classes more often
    )
    
    print("✅ Training Complete! Weights saved to Indian_Traffic_AI/yolo11n_uvh26_idd/weights/best.pt")

if __name__ == '__main__':
    main()