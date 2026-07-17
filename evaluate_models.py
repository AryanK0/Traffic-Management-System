import os
import sys
import random
import numpy as np
import torch
import traci

# Ensure imports work from the current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from core.dqn_controller import DQNController
from sumo.traci_client import TraciClient
from sumo.signal_controller import SignalController

MAX_STEPS = 3600  # Equivalent to runner.py

def evaluate_model(model_path, env_seed=42):
    """
    Evaluates a DQN model in SUMO for a fixed number of steps.
    Returns a dictionary of metrics.
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None

    # 1. Model Initialization & Inference Mode
    # ---------------------------------------------------------
    print(f"[{model_path}] Loading model...")
    # Inspect checkpoint to determine if this is the old 6-state or new 9-state model
    state_dict = torch.load(model_path, map_location="cpu")
    input_size = state_dict['fc1.weight'].shape[1]
    
    dqn = DQNController(model_path)
    
    # Manually initialize the model to bypass DQN's default input_size parameter
    # which is frozen at import time based on the config.
    from core.dqn_model import DQN
    dqn.model = DQN(input_size=input_size).to(dqn.device)
    dqn.model.load_state_dict(torch.load(model_path, map_location=dqn.device))
    dqn.loaded = True
    
    # Explicitly force eval mode to prevent memory leaks and freeze weights
    dqn.model.eval()
    
    # Explicitly set epsilon to 0.0 for pure exploitation (DQNController implicitly does this)
    epsilon = 0.0

    client = TraciClient()
    
    # 2. Strict Apples-to-Apples Comparison
    # ---------------------------------------------------------
    # Pass --seed to SUMO to ensure deterministic vehicle generation
    client.start(extra_args=["--seed", str(env_seed)])
    signal = SignalController(client)
    
    # Metric Trackers
    vehicle_wait_times = {}  # Map vehicle ID -> max accumulated wait time
    arrived_vehicles = set() # Track unique vehicles that finished their route
    max_queue_length = 0
    cumulative_reward = 0.0

    print(f"[{model_path}] Starting evaluation (Max Steps: {MAX_STEPS})...")
    
    try:
        while client.step_count < MAX_STEPS:
            state = client.get_state(signal.time_in_phase)
            
            # Reconstruct the old 6-dim state if evaluating the old baseline
            if input_size == 6:
                old_state = []
                for edge in client.incoming:
                    try:
                        halted = traci.edge.getLastStepHaltingNumber(edge)
                    except Exception:
                        halted = 0
                    old_state.append(float(halted) / 100.0)
                
                spill = 0
                for edge in client.outgoing:
                    try:
                        spill += traci.edge.getLastStepHaltingNumber(edge)
                    except Exception:
                        pass
                old_state.append(float(spill) / 100.0)
                old_state.append(float(signal.time_in_phase) / 60.0)
                
                state = old_state
            
            # Forward pass (DQNController.decide internally uses torch.no_grad())
            action, _ = dqn.decide(np.array(state, dtype="float32"))
            
            # Apply action and step environment
            result = signal.apply(action)
            client.step(5)  # Steps forward by 5 ticks (standard in runner.py)
            
            # --- 3. Metric Tracking ---
            
            # Cumulative Episode Reward
            reward = client.calculate_reward()
            cumulative_reward += reward
            
            # Total Throughput (Arrived Vehicles)
            arrived_vehicles.update(traci.simulation.getArrivedIDList())
            
            # Max Queue Length (across all approaches)
            current_max_queue = 0
            for edge in client.incoming:
                queue = traci.edge.getLastStepHaltingNumber(edge)
                if queue > current_max_queue:
                    current_max_queue = queue
            if current_max_queue > max_queue_length:
                max_queue_length = current_max_queue
                
            # Average Waiting Time
            # We track the accumulated waiting time for all active vehicles at every step
            for edge in client.incoming:
                for vid in traci.edge.getLastStepVehicleIDs(edge):
                    vehicle_wait_times[vid] = traci.vehicle.getAccumulatedWaitingTime(vid)
                    
    except Exception as e:
        print(f"[{model_path}] Error during simulation: {e}")
    finally:
        # Ensures traci closes cleanly even if simulation hits an error
        client.close()
        
    # Calculate final average wait time per vehicle
    avg_wait_time = sum(vehicle_wait_times.values()) / max(1, len(vehicle_wait_times))
    total_throughput = len(arrived_vehicles)
    
    print(f"[{model_path}] Evaluation Complete.\n")
    
    return {
        "avg_wait_time": avg_wait_time,
        "throughput": total_throughput,
        "max_queue": max_queue_length,
        "reward": cumulative_reward
    }

def print_comparison(old_metrics, new_metrics):
    """
    Automated Comparative Analytics Output
    """
    if not old_metrics or not new_metrics:
        print("Missing metrics for comparison.")
        return

    print("=" * 75)
    print(f"{'Metric':<25} | {'Old Baseline':<15} | {'New Model':<15} | {'Change':<15}")
    print("=" * 75)

    def calc_pct(old_val, new_val, lower_is_better):
        if old_val == 0:
            return "N/A"
        
        diff = new_val - old_val
        pct = (diff / abs(old_val)) * 100
        
        if lower_is_better:
            if pct < 0:
                return f"Improved ({abs(pct):.2f}%)"
            elif pct > 0:
                return f"Degraded ({abs(pct):.2f}%)"
            else:
                return "No Change"
        else:
            if pct > 0:
                return f"Improved ({abs(pct):.2f}%)"
            elif pct < 0:
                return f"Degraded ({abs(pct):.2f}%)"
            else:
                return "No Change"

    # Average Waiting Time (Lower is better)
    m1_old, m1_new = old_metrics['avg_wait_time'], new_metrics['avg_wait_time']
    m1_diff = calc_pct(m1_old, m1_new, lower_is_better=True)
    print(f"{'Average Wait Time (s)':<25} | {m1_old:<15.2f} | {m1_new:<15.2f} | {m1_diff:<15}")

    # Total Throughput (Higher is better)
    m2_old, m2_new = old_metrics['throughput'], new_metrics['throughput']
    m2_diff = calc_pct(m2_old, m2_new, lower_is_better=False)
    print(f"{'Total Throughput (vehs)':<25} | {m2_old:<15} | {m2_new:<15} | {m2_diff:<15}")

    # Maximum Queue Length (Lower is better)
    m3_old, m3_new = old_metrics['max_queue'], new_metrics['max_queue']
    m3_diff = calc_pct(m3_old, m3_new, lower_is_better=True)
    print(f"{'Max Queue Length':<25} | {m3_old:<15} | {m3_new:<15} | {m3_diff:<15}")

    # Cumulative Episode Reward (Higher is better)
    m4_old, m4_new = old_metrics['reward'], new_metrics['reward']
    m4_diff = calc_pct(m4_old, m4_new, lower_is_better=False)
    print(f"{'Cumulative Reward':<25} | {m4_old:<15.2f} | {m4_new:<15.2f} | {m4_diff:<15}")
    print("=" * 75)
    
    # Formulated natural language insights
    print("\n[Insights]")
    if m1_old > 0:
        pct_wait = ((m1_new - m1_old) / m1_old) * 100
        direction = "decreased" if pct_wait <= 0 else "increased"
        print(f" -> New model {direction} average waiting time by {abs(pct_wait):.2f}%.")
    
    if m2_old > 0:
        pct_thru = ((m2_new - m2_old) / m2_old) * 100
        direction = "increased" if pct_thru >= 0 else "decreased"
        print(f" -> New model {direction} total network throughput by {abs(pct_thru):.2f}%.")


def main():
    # 2. Strict Apples-to-Apples Comparison
    # Fix all standard random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    model_old = "models/dqn_traffic_model.pth"
    model_new = "models/dqn_traffic_model_new.pth"
    
    print("\n" + "#"*50)
    print(" DQN Traffic Management - Head-to-Head Evaluation")
    print("#"*50 + "\n")
    
    # Run Baseline
    metrics_old = evaluate_model(model_old, env_seed=42)
    
    # Run New Model
    metrics_new = evaluate_model(model_new, env_seed=42)
    
    # Output Automated Comparative Analytics
    if metrics_old and metrics_new:
        print_comparison(metrics_old, metrics_new)
    else:
        print("Evaluation failed. One or both models returned no metrics.")

if __name__ == "__main__":
    main()
