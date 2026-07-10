"""
=========================================================
Traffic Manager
=========================================================

The signal phase state machine.

Applies the DQN's KEEP/SWITCH decision safely:
    • MIN_GREEN_TIME  — ignore SWITCH if green just started
    • MAX_GREEN_TIME  — force SWITCH after 60 s (failsafe,
      same rule the DQN saw during training)
    • YELLOW_TIME     — 4 s yellow between greens

Hardware hook: _set_hardware() is where Jetson GPIO /
relay control goes when driving physical lights.

Author : Traffic AI Team
Platform : NVIDIA Jetson Nano
=========================================================
"""

import time

import config

PHASES = ["NS_GREEN", "EW_GREEN"]


class TrafficManager:

    def __init__(self):

        self.phase_index = 0
        self.phase_start = time.time()

        self.in_yellow = False
        self.yellow_start = 0.0

        self.total_switches = 0

    ########################################################
    # Properties
    ########################################################

    @property
    def current_phase(self):

        if self.in_yellow:
            return "YELLOW"

        return PHASES[self.phase_index]

    def time_in_phase(self):

        return time.time() - self.phase_start

    ########################################################
    # Apply DQN Action
    ########################################################

    def apply(self, action):
        """
        action: 0 = KEEP, 1 = SWITCH
        Returns a human-readable string of what happened.
        """

        if self.in_yellow:
            return "IN_YELLOW"

        elapsed = self.time_in_phase()

        # Failsafe: force switch after MAX_GREEN_TIME
        if action == 0 and elapsed >= config.MAX_GREEN_TIME:
            self._start_yellow()
            return "FORCED_SWITCH (max green)"

        # Safety: refuse switch during minimum green
        if action == 1 and elapsed < config.MIN_GREEN_TIME:
            return "KEEP (min green not reached)"

        if action == 1:
            self._start_yellow()
            return "SWITCH"

        return "KEEP"

    ########################################################
    # Yellow Transition
    ########################################################

    def _start_yellow(self):

        self.in_yellow = True
        self.yellow_start = time.time()
        self._set_hardware("YELLOW")

    def update(self):
        """Call every loop — finishes yellow -> next green."""

        if self.in_yellow and \
                time.time() - self.yellow_start >= config.YELLOW_TIME:

            self.in_yellow = False
            self.phase_index = (self.phase_index + 1) % len(PHASES)
            self.phase_start = time.time()
            self.total_switches += 1
            self._set_hardware(PHASES[self.phase_index])

    ########################################################
    # Hardware Hook
    ########################################################

    def _set_hardware(self, phase):
        """
        Physical light control goes here, e.g.:

            import Jetson.GPIO as GPIO
            GPIO.output(NS_GREEN_PIN, phase == "NS_GREEN")
        """

        if config.PRINT_SIGNAL_CHANGES:
            print("    [SIGNAL] -> %s" % phase)
