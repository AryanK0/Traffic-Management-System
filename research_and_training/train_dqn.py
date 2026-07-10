import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.dqn_model import DQN
from sumo.traci_client import TraciClient
from sumo.signal_controller import SignalController

class SumoEnv:
    def __init__(self, sumo_cmd, max_steps=3600):
        self.sumo_cmd = sumo_cmd
        self.client = TraciClient()
        self.signal = SignalController(self.client)
        self.max_steps = max_steps

    def reset(self, scale=1.0):
        # We start sumo via traci client
        self.client.start()
        self.signal.time_in_phase = 0
        return self.client.get_state(self.signal.time_in_phase)

    def step(self, action):
        result = self.signal.apply(action)
        self.client.step(5)
        
        # Continuous Delay Penalty Reward Function
        reward = self.client.get_squared_delay_reward()
        
        next_state = self.client.get_state(self.signal.time_in_phase)
        done = self.client.step_count >= self.max_steps
        
        return next_state, reward, done

    def close(self):
        self.client.close()


def train_dqn():
    print("🚀 Starting DQN Training Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sumo_binary = "sumo" 
    sumocfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sumo", "sumo_config.sumocfg")
    
    sumo_cmd = [
        sumo_binary, 
        "-c", sumocfg_path, 
        "--no-step-log", "true", 
        "--waiting-time-memory", "10000"
    ]
    
    env = SumoEnv(sumo_cmd, max_steps=3600)
    
    model = DQN().to(device)
    target_model = DQN().to(device)
    target_model.load_state_dict(model.state_dict())
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    memory = deque(maxlen=50000)
    batch_size = 64
    gamma = 0.99
    
    total_epochs = 100
    exploration_phase = 0.35 * total_epochs
    
    for epoch in range(total_epochs):
        state = env.reset()
        total_reward = 0
        done = False
        
        # Epsilon Decay Schedule: 
        # High exploration rate, decaying slowly over the first 35% of epochs.
        if epoch < exploration_phase:
            epsilon = 1.0 - (epoch / exploration_phase) * 0.95  # Decays from 1.0 to 0.05
        else:
            epsilon = 0.05  # Shifts to exploitation

        while not done:
            if random.random() < epsilon:
                action = random.choice([0, 1])
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()
                    
            next_state, reward, done = env.step(action)
            memory.append((state, action, reward, next_state, done))
            
            state = next_state
            total_reward += reward
            
            # Training Step
            if len(memory) > batch_size:
                batch = random.sample(memory, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                states_t = torch.FloatTensor(np.array(states)).to(device)
                actions_t = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states_t = torch.FloatTensor(np.array(next_states)).to(device)
                dones_t = torch.FloatTensor(dones).unsqueeze(1).to(device)
                
                q_values = model(states_t).gather(1, actions_t)
                next_q_values = target_model(next_states_t).max(1)[0].unsqueeze(1)
                target_q_values = rewards_t + gamma * next_q_values * (1 - dones_t)
                
                loss = criterion(q_values, target_q_values.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Update target network occasionally
        if epoch % 5 == 0:
            target_model.load_state_dict(model.state_dict())
            
        print(f"Epoch {epoch+1}/{total_epochs} | Epsilon: {epsilon:.2f} | Total Reward: {total_reward:.2f}")

    env.close()
    
    # Save the trained model
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "models"), exist_ok=True)
    save_path = os.path.join(os.path.dirname(__file__), "..", "models", "dqn_traffic_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete! Weights saved to {save_path}")

if __name__ == "__main__":
    train_dqn()
