import os
import cv2

def scrub_dataset(base_dir="master_dataset"):
    corrupted_images = 0
    empty_labels = 0
    
    for split in ['train', 'val']:
        images_dir = os.path.join(base_dir, 'images', split)
        labels_dir = os.path.join(base_dir, 'labels', split)
        
        if not os.path.exists(images_dir):
            continue
            
        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(images_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            label_path = os.path.join(labels_dir, base_name + ".txt")
            
            # 1. Check if image is corrupted (0 bytes)
            if os.path.getsize(img_path) == 0:
                print(f"Deleting 0-byte image: {img_path}")
                os.remove(img_path)
                if os.path.exists(label_path):
                    os.remove(label_path)
                corrupted_images += 1
                continue
                
            # 2. Check if label file is missing or empty
            is_empty_label = False
            if not os.path.exists(label_path):
                is_empty_label = True
            elif os.path.getsize(label_path) == 0:
                is_empty_label = True
            else:
                # Optionally check if all lines are empty
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        is_empty_label = True
                        
            if is_empty_label:
                print(f"Deleting image with empty/missing label: {img_path}")
                os.remove(img_path)
                if os.path.exists(label_path):
                    os.remove(label_path)
                empty_labels += 1

    print(f"Scrubbed {corrupted_images} corrupted images and {empty_labels} empty labels.")

if __name__ == "__main__":
    scrub_dataset()
