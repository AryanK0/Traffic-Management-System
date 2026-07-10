import os
import sys
import torch
import numpy as np
import time

# Import the network and environment classes from your training script
from dqn_agent import DQN, SumoEnv

def evaluate():
    print("Loading the trained DQN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate the model architecture
    model = DQN().to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load("dqn_traffic_model.pth", map_location=device, weights_only=True))
    model.eval() # Set model to evaluation mode (no learning, purely making decisions)
    
    # Setup sumo_cmd to use sumo-gui so you can watch the AI
    sumo_binary = "sumo-gui" 
    sumocfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sumo_env", "sumo_config.sumocfg")
    
    # We remove --quit-on-end so the GUI stays open at the end for you to see the final state
    sumo_cmd = [
        sumo_binary, 
        "-c", sumocfg_path, 
        "--no-step-log", "true", 
        "--waiting-time-memory", "10000",
        "--start"
    ]
    
    env = SumoEnv(sumo_cmd, max_steps=1000)
    
    print("Starting evaluation episode. Watch the SUMO GUI!")
    # Let's test it on a standard traffic scale of 1.0 (or you can try 1.5 to test heavy traffic)
    state = env.reset(scale=1.0)
    total_reward = 0
    done = False
    
    while not done:
        # In evaluation, there is no Epsilon-Greedy randomness. 
        # The AI purely EXPLOITS its learned knowledge.
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = q_values.argmax().item() # Choose the action with the highest Q-value
            
        next_state, reward, done = env.step(action)
        
        # Add a slight delay so you can comfortably watch the AI manage the traffic light
        time.sleep(0.1) 
        
        state = next_state
        total_reward += reward
        
    env.close()
    print(f"Evaluation Complete! Total Reward for this episode: {total_reward:.2f}")

if __name__ == "__main__":
    evaluate()
