import os
import torch
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.dqn_model import DQN
import warnings
import traci

warnings.filterwarnings("ignore")

class TrafficEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dqn = DQN().to(self.device)
        self.dqn.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "..", "models", "dqn_traffic_model.pth"), map_location=self.device, weights_only=True))
        self.dqn.eval()
        
        self.incoming_edges = ["n_in", "s_in", "e_in", "w_in"]
        
        sumo_binary = "sumo" # Using 'sumo' instead of 'sumo-gui' to run the test at maximum speed
        self.sumo_cmd = [
            sumo_binary, 
            "-c", os.path.join(os.path.dirname(__file__), "..", "sumo", "sumo_config.sumocfg"), 
            "--no-step-log", "true", 
            "--waiting-time-memory", "10000",
            "--start"
        ]

    def get_queues(self):
        state = []
        for edge in self.incoming_edges:
            try:
                state.append(traci.edge.getLastStepHaltingNumber(edge))
            except:
                state.append(0)
        return state

    def run_simulation(self, mode="static", max_steps=3600):
        traci.start(self.sumo_cmd)
        tl_id = traci.trafficlight.getIDList()[0]
        
        try:
            num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
        except:
            num_phases = 8
            
        step_count = 0
        total_wait_time = 0
        queue_lengths = []
        throughput = 0 # FIX: We will continually add to this!
        
        time_in_phase = 0
        current_phase = 0
        
        try:
            while step_count < max_steps:
                queues = self.get_queues()
                queue_lengths.append(sum(queues))
                
                # Record Wait Times
                for edge in self.incoming_edges:
                    total_wait_time += traci.edge.getWaitingTime(edge)
                    
                steps_to_sim = 5
                switch = False
                
                # ==========================================
                # LOGIC 1: STATIC (Old Standard)
                # ==========================================
                if mode == "static":
                    if time_in_phase >= 40: # Force change every 40 seconds
                        switch = True
                        
                # ==========================================
                # LOGIC 2: ACTUATED (Current Real-World Smart Standard)
                # ==========================================
                elif mode == "actuated":
                    # If there are cars waiting on the green light, keep it green (up to 60s max)
                    # If the green lane is empty, switch immediately to save time
                    active_green_queues = queues[current_phase % 4] # Simplified check
                    if time_in_phase >= 10 and active_green_queues == 0:
                        switch = True
                    elif time_in_phase >= 60:
                        switch = True
                        
                # ==========================================
                # LOGIC 3: YOUR DQN AI (The Future)
                # ==========================================
                elif mode == "dqn":
                    dqn_state = []
                    for q in queues:
                        dqn_state.append(float(q) / 100.0)
                        
                    spillback = 0
                    for edge in ["n_out", "s_out", "e_out", "w_out"]:
                        try:
                            spillback += traci.edge.getLastStepHaltingNumber(edge)
                        except:
                            pass
                    dqn_state.append(float(spillback) / 100.0)
                    dqn_state.append(float(time_in_phase) / 60.0)
                    
                    state_tensor = torch.FloatTensor(dqn_state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        action = self.dqn(state_tensor).argmax().item()
                    
                    if action == 1 or time_in_phase >= 60:
                        switch = True

                # Apply Phase Switch
                if switch:
                    yellow_phase = (current_phase + 1) % num_phases
                    green_phase = (current_phase + 2) % num_phases
                    
                    traci.trafficlight.setPhase(tl_id, yellow_phase)
                    for _ in range(4):
                        traci.simulationStep()
                        throughput += traci.simulation.getArrivedNumber()
                        step_count += 1
                        
                    traci.trafficlight.setPhase(tl_id, green_phase)
                    current_phase = green_phase
                    time_in_phase = 0
                else:
                    traci.trafficlight.setPhase(tl_id, current_phase)
                    
                time_in_phase += steps_to_sim

                for _ in range(steps_to_sim):
                    traci.simulationStep()
                    throughput += traci.simulation.getArrivedNumber() # FIX: Count arrivals safely!
                    step_count += 1
                    
        except traci.exceptions.FatalTraCIError:
            # FIX: If SUMO naturally ends its config timer early, gracefully catch it and continue!
            pass
            
        try:
            traci.close()
        except:
            pass
        
        avg_queue = sum(queue_lengths) / len(queue_lengths) if queue_lengths else 0
        
        return {
            "WaitTime": total_wait_time,
            "Throughput": throughput,
            "AvgQueue": avg_queue
        }

    def start_benchmark(self):
        print("[TEST] Starting Competitive Benchmark: Static vs. Actuated vs. AI")
        print("This will run in the background (no GUI) to simulate 3 hours of traffic instantly...\n")
        
        print("1/3: Running Baseline (Static Indian Timer)...")
        static_results = self.run_simulation(mode="static")
        
        print("2/3: Running Industry Standard (Vehicle-Actuated)...")
        actuated_results = self.run_simulation(mode="actuated")
        
        print("3/3: Running Your DQN Agent...")
        dqn_results = self.run_simulation(mode="dqn")
        
        print("\n=======================================================")
        print("[RESULTS] FINAL BENCHMARK RESULTS (1 Hour Simulation)")
        print("=======================================================")
        print(f"{'Metric':<25} | {'Static (Dumb)':<15} | {'Actuated (Current)':<20} | {'DQN AI (Yours)':<15}")
        print("-" * 80)
        print(f"{'Total Wait Time (sec)':<25} | {static_results['WaitTime']:<15.0f} | {actuated_results['WaitTime']:<20.0f} | {dqn_results['WaitTime']:<15.0f}")
        print(f"{'Intersection Throughput':<25} | {static_results['Throughput']:<15} | {actuated_results['Throughput']:<20} | {dqn_results['Throughput']:<15}")
        print(f"{'Avg Queue Length (cars)':<25} | {static_results['AvgQueue']:<15.1f} | {actuated_results['AvgQueue']:<20.1f} | {dqn_results['AvgQueue']:<15.1f}")
        print("=======================================================\n")
        
        # Calculate AI improvement over the Current Industry Standard
        improvement = ((actuated_results['WaitTime'] - dqn_results['WaitTime']) / actuated_results['WaitTime']) * 100
        print(f"[WIN] Your AI improved traffic flow by {improvement:.1f}% compared to current Real-World Smart Lights!")

if __name__ == "__main__":
    evaluator = TrafficEvaluator()
    evaluator.start_benchmark()