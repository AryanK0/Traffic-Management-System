import cv2
from ultralytics import YOLO

# --- Configuration ---
# Adjusted paths to use the files in the 'test' directory
MODEL_PATH = "test/best.onnx" 
VIDEO_PATH = "test/test_video.mp4"
OUTPUT_PATH = "presentation_demo.mp4"

def main():
    # 1. Load the YOLO ONNX model
    # Ultralytics natively supports loading .onnx files
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH, task="detect")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 2. Extract class ID mapping from the model
    # model.names is a dictionary mapping integer class IDs to string class names
    names_dict = model.names
    
    # Reverse the dictionary to easily look up IDs by class name
    name_to_id = {name: cls_id for cls_id, name in names_dict.items()}
    
    print(f"Available model classes: {names_dict}")
    
    unknown_id = name_to_id.get("unknown", -1)
    two_wheeler_id = name_to_id.get("2_wheeler", -1)
    
    if unknown_id == -1 or two_wheeler_id == -1:
        print("Warning: 'unknown' or '2_wheeler' class not found in model names! Relabeling will not work.")
    else:
        print(f"Relabeling class ID {unknown_id} ('unknown') -> {two_wheeler_id} ('2_wheeler')")
    
    # 3. Open the video file
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return
        
    # Get video properties for the VideoWriter
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Fallback for NaN or 0 fps
        fps = 30.0
        
    # Initialize the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    print(f"Starting inference on {VIDEO_PATH}... Press 'q' in the video window to quit early.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 4. Run YOLO prediction on the current frame
        results = model(frame, verbose=False)
        result = results[0]
        
        # 5. Intercept and modify the detection tensors in real-time
        if result.boxes is not None and len(result.boxes) > 0:
            
            if unknown_id != -1 and two_wheeler_id != -1:
                # Clone the underlying data tensor to safely modify it outside InferenceMode
                modified_data = result.boxes.data.clone()
                
                # The class ID is the 6th column (index 5) in the bounding box data [x1, y1, x2, y2, conf, cls]
                cls_column = modified_data[:, 5]
                cls_column[cls_column == unknown_id] = two_wheeler_id
                
                # Assign the cloned and modified tensor back to the boxes object
                result.boxes.data = modified_data
                
        # 6. After the tensor is modified, draw the corrected boxes
        annotated_frame = result.plot()
        
        # 7. Write the annotated frame to the output file
        out.write(annotated_frame)
        
        # 8. Display the live cv2 window
        cv2.imshow("Real-Time YOLO Relabeling Demo", annotated_frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting early...")
            break
            
    # Clean up all resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete! Corrected video saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
