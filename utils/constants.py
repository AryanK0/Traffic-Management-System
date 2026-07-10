"""
Project-wide constants (things that never change,
as opposed to config.py which holds tunable settings).
"""

# Vehicle taxonomy (must match training data.yaml order)
CLASS_NAMES = {
    0: "2_wheeler",
    1: "3_wheeler",
    2: "4_wheeler",
    3: "heavy_vehicle",
    4: "unknown",
}

# DQN actions
ACTION_KEEP = 0
ACTION_SWITCH = 1

# Signal phases
PHASE_NS_GREEN = "NS_GREEN"
PHASE_EW_GREEN = "EW_GREEN"
PHASE_YELLOW = "YELLOW"

# State vector layout (indexes)
STATE_NORTH = 0
STATE_SOUTH = 1
STATE_EAST = 2
STATE_WEST = 3
STATE_SPILLBACK = 4
STATE_TIME_IN_PHASE = 5
