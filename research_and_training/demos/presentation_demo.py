import os
import sys
import torch
import numpy as np
import time
import warnings
import cv2
import traci

warnings.filterwarnings("ignore")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("Please install ultralytics: pip install ultralytics")

from dqn_agent import DQN

def run_phase_1_yolo():
    print("\n=======================================================")
    print("🎥 PHASE 1: YOLO Vision Detection Pipeline")
    print("=======================================================")
    
    yolo_path = "test/best.onnx"
    video_path = "test/test_video.mp4"
    
    print(f"Loading YOLO model from {yolo_path}...")
    try:
        model = YOLO(yolo_path, task="detect")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    names_dict = model.names
    name_to_id = {name: id for id, name in names_dict.items()}
    unknown_id = name_to_id.get("unknown", -1)
    two_wheeler_id = name_to_id.get("2_wheeler", -1)
    
    print(f"Opening video {video_path} and skipping first 16 seconds...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
        
    # Skip the first 16000 milliseconds (16 seconds)
    cap.set(cv2.CAP_PROP_POS_MSEC, 16000)
    
    print("Playing Video! Press 'q' on the video window to stop Phase 1 and proceed to Phase 2.")
    
    while True:
        # Read two frames, process the second one (effectively 2x speed)
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break
            
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break
            
        results = model(frame, verbose=False)
        result = results[0]
        
        # In-place relabeling
        if result.boxes is not None and len(result.boxes) > 0:
            if unknown_id != -1 and two_wheeler_id != -1:
                modified_data = result.boxes.data.clone()
                cls_column = modified_data[:, 5]
                cls_column[cls_column == unknown_id] = two_wheeler_id
                result.boxes.data = modified_data
                
        annotated_frame = result.plot()
        cv2.imshow("PHASE 1: Live YOLO Inference", annotated_frame)
        
        # Use a slightly longer waitKey to slow down playback if it's too fast, 
        # or 1ms for max speed processing
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User interrupted Phase 1.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("Phase 1 Complete.\n")

def get_sumo_state(time_in_phase):
    """ Get exact queue lengths from SUMO to feed the DQN """
    state = []
    for edge in ["n_in", "s_in", "e_in", "w_in"]:
        try:
            halted_count = traci.edge.getLastStepHaltingNumber(edge)
        except:
            halted_count = 0
        state.append(float(halted_count) / 100.0)
        
    spillback_count = 0
    for edge in ["n_out", "s_out", "e_out", "w_out"]:
        try:
            spillback_count += traci.edge.getLastStepHaltingNumber(edge)
        except:
            pass
    state.append(float(spillback_count) / 100.0)
    state.append(float(time_in_phase) / 60.0)
        
    return np.array(state, dtype=np.float32)

def run_phase_2_sumo():
    print("\n=======================================================")
    print("🚦 PHASE 2: DQN Reinforcement Learning Control")
    print("=======================================================")
    
    dqn_path = "dqn_traffic_model.pth"
    print(f"Loading DQN model from {dqn_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = DQN().to(device)
    dqn.load_state_dict(torch.load(dqn_path, map_location=device, weights_only=True))
    dqn.eval()
    
    sumocfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_env", "sumo_config.sumocfg")
    
    sumo_cmd = [
        "sumo-gui",  # We want the GUI for the demo
        "-c", sumocfg_path, 
        "--start"
    ]
    
    print("Starting TraCI and SUMO-GUI...")
    try:
        traci.close()
    except:
        pass
        
    traci.start(sumo_cmd)
    tl_id = traci.trafficlight.getIDList()[0]
    
    try:
        num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
    except:
        num_phases = 8
        
    step_count = 0
    max_steps = 1800
    time_in_current_phase = 0
    
    print("SUMO Simulation Running. Watch the graphical interface!")
    
    try:
        while step_count < max_steps:
            state = get_sumo_state(time_in_current_phase)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = dqn(state_tensor)
                action = q_values.argmax().item()
                
            steps_to_sim = 5
            
            # Failsafe
            if action == 0:  
                time_in_current_phase += steps_to_sim
                if time_in_current_phase >= 60: 
                    action = 1  
                    action_str = "FORCED SWITCH (Max Green)"
                    time_in_current_phase = 0
                else:
                    action_str = "KEEP Phase"
            else:  
                action_str = "SWITCH Phase"
                time_in_current_phase = 0
                
            print(f"Time {step_count:4}s | Camera Queues [N:{int(state[0]*100):3} S:{int(state[1]*100):3} E:{int(state[2]*100):3} W:{int(state[3]*100):3}] | AI Action: {action_str}")
                
            if action == 1:
                current_phase = traci.trafficlight.getPhase(tl_id)
                yellow_phase = (current_phase + 1) % num_phases
                green_phase = (current_phase + 2) % num_phases
                
                traci.trafficlight.setPhase(tl_id, yellow_phase)
                for _ in range(4):
                    traci.simulationStep()
                    step_count += 1
                    
                traci.trafficlight.setPhase(tl_id, green_phase)
            else:
                current_phase = traci.trafficlight.getPhase(tl_id)
                traci.trafficlight.setPhase(tl_id, current_phase)
                
            for _ in range(steps_to_sim):
                traci.simulationStep()
                step_count += 1
                
            time.sleep(0.2)
            
    except traci.exceptions.FatalTraCIError:
        print("SUMO window closed by user.")
        
    try:
        traci.close()
    except:
        pass
    print("Phase 2 Complete.")

if __name__ == "__main__":
    run_phase_1_yolo()
    run_phase_2_sumo()
