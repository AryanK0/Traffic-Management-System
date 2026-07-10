"""
=========================================================
DQN Controller
=========================================================

Loads the trained DQN and turns a 6-dim state into a
KEEP / SWITCH decision.

Runs on CPU on purpose — the Jetson Nano GPU is kept
free for YOLO inference. DQN inference is < 1 ms on CPU.

Author : Traffic AI Team
Platform : NVIDIA Jetson Nano
=========================================================
"""

import torch

import config
from core.dqn_model import DQN

ACTION_KEEP = 0
ACTION_SWITCH = 1

ACTION_NAMES = {
    ACTION_KEEP: "KEEP",
    ACTION_SWITCH: "SWITCH",
}


class DQNController:

    def __init__(self, model_path=None):

        self.model_path = str(model_path or config.DQN_MODEL_PATH)
        self.device = torch.device("cpu")
        self.model = None
        self.loaded = False
        self.total_decisions = 0

    ########################################################
    # Load Model
    ########################################################

    def load_model(self):

        self.model = DQN().to(self.device)

        # NOTE: no weights_only=True — the Jetson's torch 1.10
        # does not support that argument.
        state_dict = torch.load(
            self.model_path,
            map_location=self.device
        )

        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.loaded = True

    ########################################################
    # Decide
    ########################################################

    def decide(self, state):
        """
        state : np.ndarray shape (6,), already normalized.
        Returns (action, q_values_list).
        """

        if not self.loaded:
            raise RuntimeError("DQN model has not been loaded.")

        tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.model(tensor).squeeze(0)

        action = int(q_values.argmax())
        self.total_decisions += 1

        if config.PRINT_DQN_STATE:
            print("DQN state: %s -> %s" % (state, ACTION_NAMES[action]))

        return action, q_values.tolist()
