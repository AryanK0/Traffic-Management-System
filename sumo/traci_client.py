"""
=========================================================
TraCI Client
=========================================================

Thin wrapper around TraCI: starts/stops SUMO and reads
the training-format state from the simulation.

Runs on the machine that has SUMO installed (usually the
laptop, not the Jetson).

Author : Traffic AI Team
=========================================================
"""

import os
import sys

if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))

import traci

import config


class TraciClient:

    def __init__(self):

        self.incoming = list(config.INCOMING_EDGES.values())
        self.outgoing = list(config.OUTGOING_EDGES.values())
        self.tl_id = None
        self.num_phases = 8
        self.step_count = 0
        self.last_wait_time = 0.0

    ########################################################
    # Start / Stop
    ########################################################

    def start(self, extra_args=None):

        binary = config.SUMO_BINARY if config.USE_SUMO_GUI else "sumo"

        cmd = [
            binary,
            "-c", str(config.SUMO_CONFIG),
            "--no-step-log", "true",
            "--waiting-time-memory", "10000",
            "--start",
        ]
        if extra_args:
            cmd.extend(extra_args)

        try:
            traci.close()
        except Exception:
            pass

        traci.start(cmd)

        self.tl_id = traci.trafficlight.getIDList()[0]
        try:
            self.num_phases = len(
                traci.trafficlight.getAllProgramLogics(self.tl_id)[0].phases
            )
        except Exception:
            self.num_phases = 8

        self.step_count = 0
        print("SUMO started. Traffic light: %s (%d phases)"
              % (self.tl_id, self.num_phases))

    def close(self):

        try:
            traci.close()
        except Exception:
            pass

    ########################################################
    # State (EXACT training format)
    ########################################################

    def get_state(self, time_in_phase):

        state = []

        for edge in self.incoming:
            try:
                veh_count = traci.edge.getLastStepVehicleNumber(edge)
                halting_count = traci.edge.getLastStepHaltingNumber(edge)
                
                lane_count = traci.edge.getLaneNumber(edge)
                edge_len = traci.lane.getLength(f"{edge}_0")
                max_capacity = (edge_len / 7.5) * lane_count
                
                density = min(1.0, veh_count / max_capacity) if max_capacity > 0 else 0.0
                queue_ratio = (halting_count / veh_count) if veh_count > 0 else 0.0
                
                state.extend([density, queue_ratio])
            except Exception:
                state.extend([0.0, 0.0])

        state.append(float(time_in_phase) / 60.0)

        return state

    ########################################################
    # Stepping / Metrics
    ########################################################

    def step(self, n=1):

        for _ in range(n):
            traci.simulationStep()
            self.step_count += 1

    def total_waiting_time(self):

        return sum(traci.edge.getWaitingTime(e) for e in self.incoming)

    def calculate_reward(self):
        """
        Differential Wait Time Reward: R_t = W_{t-1} - W_t
        Positive reward if wait time decreases (intersection clears).
        Applies a severe penalty for teleporting vehicles (massive jam).
        """
        current_wait_time = 0.0
        
        for edge in self.incoming:
            try:
                vehicles = traci.edge.getLastStepVehicleIDs(edge)
                for vid in vehicles:
                    current_wait_time += traci.vehicle.getWaitingTime(vid)
            except Exception:
                pass
                
        reward = self.last_wait_time - current_wait_time
        
        try:
            if traci.simulation.getStartingTeleportNumber() > 0:
                reward -= 10.0
        except Exception:
            pass

        self.last_wait_time = current_wait_time
        
        return reward

