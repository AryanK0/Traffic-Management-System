import os
import random
import cv2

CLASS_MAP = {
    0: "2-Wheeler",
    1: "3-Wheeler",
    2: "4-Wheeler",
    3: "Heavy_Vehicle",
    4: "Unknown"
}

def unnormalize_yolo(bbox, img_w, img_h):
    cx, cy, w, h = bbox
    cx_abs = cx * img_w
    cy_abs = cy * img_h
    w_abs = w * img_w
    h_abs = h * img_h
    
    xmin = int(cx_abs - w_abs / 2.0)
    ymin = int(cy_abs - h_abs / 2.0)
    xmax = int(cx_abs + w_abs / 2.0)
    ymax = int(cy_abs + h_abs / 2.0)
    
    return xmin, ymin, xmax, ymax

def main():
    images_dir = os.path.join("master_dataset", "images", "train")
    labels_dir = os.path.join("master_dataset", "labels", "train")
    out_dir = "sanity_check_output"
    
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(images_dir):
        print(f"Images directory {images_dir} does not exist.")
        return

    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        print("No images found in the training directory.")
        return
        
    sample_images = random.sample(all_images, min(30, len(all_images)))
    
    for img_name in sample_images:
        img_path = os.path.join(images_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w, _ = img.shape
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]
                    
                    xmin, ymin, xmax, ymax = unnormalize_yolo(bbox, w, h)
                    
                    label_str = CLASS_MAP.get(cls_id, "Unknown")
                    
                    # Draw bbox
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    # Draw label background
                    (tw, th), _ = cv2.getTextSize(label_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(img, (xmin, ymin - th - 5), (xmin + tw, ymin), (0, 255, 0), -1)
                    # Draw label text
                    cv2.putText(img, label_str, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        out_path = os.path.join(out_dir, img_name)
        cv2.imwrite(out_path, img)
        print(f"Saved sanity check image to {out_path}")

if __name__ == "__main__":
    main()
