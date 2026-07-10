from ultralytics import YOLO

def main():
    print("🚀 Initializing YOLOv11 CPU Overnight Training Run...")
    
    # Load the base nano model
    model = YOLO("yolo11n.pt") 

    # FIXED: data.yaml is in your root folder, not inside master_dataset
    dataset_config = "data.yaml" 

    print("🔥 Beginning Overnight Training Loop on CPU (~6 to 8 hours)...")
    results = model.train(
        data=dataset_config,
        epochs=10,           # Train for 10 full passes of the subset
        batch=4,             # Keep batch tiny so Windows RAM doesn't crash
        imgsz=640,           # FULL resolution! We want to see real accuracy now.
        device="cpu",        # Force the Intel i7 to do the work
        fraction=0.10,       # Use 10% of the dataset (~8,500 images)
        project="CPU_Overnight_Run",
        name="yolo11n_indian_traffic",
        plots=True           # Crucial: This generates the performance graphs!
    )
    
    print("✅ Overnight training complete! Check the 'CPU_Overnight_Run' folder for your graphs and best.pt!")

if __name__ == '__main__':
    main()