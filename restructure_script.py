import os, shutil

def move_safe(src, dst):
    if os.path.exists(src):
        print(f'Moving {src} to {dst}')
        try:
            shutil.move(src, dst)
        except Exception as e:
            print(f'Error moving {src}: {e}')

def remove_safe(src):
    if os.path.exists(src):
        print(f'Removing {src}')
        try:
            if os.path.isdir(src):
                shutil.rmtree(src)
            else:
                os.remove(src)
        except Exception as e:
            print(f'Error removing {src}: {e}')

os.makedirs('research_and_training/demos', exist_ok=True)
os.makedirs('data', exist_ok=True)

# Move datasets
move_safe('master_dataset', 'data/master_dataset')
move_safe('master_dataset_640', 'data/master_dataset_640')
move_safe('master_dataset_640.zip', 'data/master_dataset_640.zip')

scripts = [
    'train_yolov11.py', 'train_microbatch.py', 'resize_dataset.py', 
    'health_scrubber.py', 'parsers.py', 'verify_labels.py', 
    'evaluate_models.py', 'evaluate_model.py', 'setup_yolo.py', 
    'taxonomy.py', 'yolo_utils.py', 'config.py', 'data.yaml', 
    'indian_traffic_hyp.yaml', 'DATA_PIPELINE_README.md', 'yolo11n.pt'
]
for s in scripts:
    move_safe(s, f'research_and_training/{s}')

move_safe('runs', 'research_and_training/runs')
move_safe('presentation_demo.py', 'research_and_training/demos/presentation_demo.py')
move_safe('presentation_demo.mp4', 'research_and_training/demos/presentation_demo.mp4')
move_safe('test', 'research_and_training/demos/test')

junk = ['unknown_classes_log.csv', 'sanity_check_output', 'main.py', 'main_sumo.py', 'system_controller.py', 'dqn_agent.py', 'sumo_env', 'dqn_traffic_model.pth']
for j in junk:
    remove_safe(j)

src_jetson = 'traffic_ai_jetson_v2 2/traffic_ai_jetson_v2'
if os.path.exists(src_jetson):
    for item in os.listdir(src_jetson):
        move_safe(os.path.join(src_jetson, item), os.path.join('.', item))
    remove_safe('traffic_ai_jetson_v2 2')

print("Done restructuring!")
