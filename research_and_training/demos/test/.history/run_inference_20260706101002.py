from ultralytics import YOLO

def main():
    print("🧠 Loading the local Indian Traffic model...")
    # Point directly to the weights file in your test folder
    model = YOLO("best.onnx")

    print("🎥 Popping open the video window...")
    
    # Run the prediction
    results = model.predict(
        source="test_video.mp4", # Make sure this matches your video file name
        show=True,               # 👈 MAGIC: Pops open a window to watch it live!
        save=True,               # Still saves a final copy just in case
        conf=0.25,               
        line_width=2             
    )
    
if __name__ == '__main__':
    main()