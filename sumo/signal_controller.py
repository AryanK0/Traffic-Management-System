"""
=========================================================
Signal Controller (SUMO side)
=========================================================

Applies KEEP/SWITCH decisions to the SUMO traffic light
EXACTLY the way the DQN experienced it during training:

    SWITCH: yellow phase for 4 steps, then next green
    KEEP  : re-assert current phase (resets SUMO's timer)
    Failsafe: force SWITCH after 60 s of green

Author : Traffic AI Team
=========================================================
"""

import traci

import config


class SignalController:

    def __init__(self, traci_client):

        self.client = traci_client
        self.time_in_phase = 0
        self.total_switches = 0

    def apply(self, action):
        """
        action: 0 = KEEP, 1 = SWITCH.
        Returns string describing what happened.
        Also advances the simulation through the yellow phase
        when switching (4 steps), matching training.
        """

        tl = self.client.tl_id
        num_phases = self.client.num_phases
        result = "KEEP"

        # Max-green failsafe (same as training)
        if action == 0 and self.time_in_phase >= config.MAX_GREEN_TIME:
            action = 1
            result = "FORCED_SWITCH"

        if action == 1:

            if result == "KEEP":
                result = "SWITCH"

            current = traci.trafficlight.getPhase(tl)

            traci.trafficlight.setPhase(tl, (current + 1) % num_phases)
            self.client.step(config.YELLOW_TIME)

            traci.trafficlight.setPhase(tl, (current + 2) % num_phases)

            self.time_in_phase = 0
            self.total_switches += 1

        else:
            current = traci.trafficlight.getPhase(tl)
            traci.trafficlight.setPhase(tl, current)

        self.time_in_phase += 5

        return result
