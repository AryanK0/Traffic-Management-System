import os
import cv2
import shutil
import concurrent.futures
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = 'master_dataset'
DEST_DIR = 'master_dataset_640'
TARGET_SIZE = (640, 640)
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg')

def process_image(img_path):
    """
    Reads, resizes, and saves a single image while maintaining folder structure.
    Returns (success_boolean, error_message) to handle corrupted files gracefully.
    """
    try:
        # Reconstruct the original folder structure in the new destination
        rel_path = os.path.relpath(img_path, SOURCE_DIR)
        dest_path = os.path.join(DEST_DIR, rel_path)
        
        # Ensure the subdirectories exist
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            return False, f"Warning: Skipped corrupted or unreadable image: {img_path}"
            
        # Resize using INTER_AREA, which is highly optimized for downsampling without losing features
        resized_img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # Save the optimized image
        cv2.imwrite(dest_path, resized_img)
        return True, None
        
    except Exception as e:
        return False, f"Warning: Exception encountered on {img_path}: {e}"

def main():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory '{SOURCE_DIR}' not found.")
        return

    print(f"Scanning '{SOURCE_DIR}' for images...")
    image_paths = []
    
    # Recursively find all images
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(VALID_EXTENSIONS):
                image_paths.append(os.path.join(root, file))

    total_images = len(image_paths)
    if total_images == 0:
        print("No images found to process. Exiting.")
        return

    print(f"Found {total_images} images. Beginning high-speed resizing to {TARGET_SIZE}...")
    
    fatal_error = False
    failed_count = 0

    try:
        # Utilize all CPU cores using ProcessPoolExecutor
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Wrap with tqdm for a beautiful progress bar and ETA estimation
            results = list(tqdm(
                executor.map(process_image, image_paths), 
                total=total_images, 
                desc="Resizing Dataset",
                unit="img"
            ))
            
        # Log any warnings from corrupted images without crashing the script
        for success, msg in results:
            if not success:
                print(msg)
                failed_count += 1
                
        print(f"\nResizing Complete! Successfully squashed {total_images - failed_count}/{total_images} images.")
        
    except Exception as e:
        print(f"\nFATAL ERROR during multiprocessing: {e}")
        fatal_error = True

    # --- Auto-Cleanup (Space Saver Phase) ---
    if not fatal_error:
        print("\nSafety check passed: No fatal pool errors detected.")
        print(f"Deleting original '{SOURCE_DIR}' folder to reclaim 140GB of storage...")
        try:
            shutil.rmtree(SOURCE_DIR)
            print(f"SUCCESS: Original dataset '{SOURCE_DIR}' has been safely deleted.")
        except Exception as e:
            print(f"Failed to delete directory: {e}")
    else:
        print(f"\nSafety check FAILED. Retaining '{SOURCE_DIR}' to prevent data loss.")

if __name__ == '__main__':
    main()
