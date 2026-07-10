"""
=========================================================
State Builder
=========================================================

Packs lane counts into the EXACT 6-dim state vector the
DQN was trained on (see dqn_agent.py _get_state):

    [0] north queue / 100
    [1] south queue / 100
    [2] east  queue / 100
    [3] west  queue / 100
    [4] junction spillback / 100
    [5] time in current phase / 60

Author : Traffic AI Team
Platform : NVIDIA Jetson Nano
=========================================================
"""

import numpy as np

MAX_LANE_CAPACITY = 100.0
TIME_NORM = 60.0

class StateBuilder:

    def build(self, lane_counts, time_in_phase):
        """
        lane_counts : dict with keys north/south/east/west/center
        time_in_phase : seconds the current green has been active
        """

        state = np.array([
            np.clip(lane_counts.get("north", 0) / MAX_LANE_CAPACITY, 0.0, 1.0),
            np.clip(lane_counts.get("south", 0) / MAX_LANE_CAPACITY, 0.0, 1.0),
            np.clip(lane_counts.get("east", 0) / MAX_LANE_CAPACITY, 0.0, 1.0),
            np.clip(lane_counts.get("west", 0) / MAX_LANE_CAPACITY, 0.0, 1.0),
            np.clip(lane_counts.get("center", 0) / MAX_LANE_CAPACITY, 0.0, 1.0),
            np.clip(float(time_in_phase) / TIME_NORM, 0.0, 1.0),
        ], dtype=np.float32)

        return state
