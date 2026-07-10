import os
import yaml

def download_yolo_weights():
    try:
        from ultralytics import YOLO
        print("Downloading yolo11n.pt base weights...")
        # Instantiating the model will automatically download the weights if not present
        model = YOLO('yolo11n.pt')
        print("Successfully downloaded and initialized yolo11n.pt")
    except ImportError:
        print("Error: ultralytics library is not installed. Please run: pip install ultralytics")
    except Exception as e:
        print(f"Error initializing YOLO: {e}")

def create_hyperparameter_config():
    hyp_config = {
        'mosaic': 1.0,
        'mixup': 0.1,
        'degrees': 10.0,
        'hsv_s': 0.5,
        'hsv_v': 0.4
    }
    
    file_path = "indian_traffic_hyp.yaml"
    with open(file_path, 'w') as f:
        yaml.dump(hyp_config, f, default_flow_style=False)
        
    print(f"Successfully generated hyperparameter config at {file_path}")

def main():
    print("Setting up YOLO environment...")
    download_yolo_weights()
    create_hyperparameter_config()
    print("Setup complete.")

if __name__ == "__main__":
    main()
