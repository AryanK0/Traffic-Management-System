"""
=========================================================
SUMO Runner — run on the LAPTOP that has SUMO installed.
=========================================================

Two modes:

LOCAL mode (everything on this machine):
    python3 sumo/runner.py
    -> loads the DQN here and controls the SUMO light.

JETSON mode (hardware-in-the-loop, the real demo):
    python3 sumo/runner.py --jetson
    -> waits for the Jetson to connect over the network,
       sends it each traffic state, and applies whatever
       KEEP/SWITCH decision the Jetson's DQN sends back.
       Watch the SUMO GUI obey the Jetson.

    On the Jetson, connect with:
        python3 main_sumo.py <this laptop's IP>
    (over the USB cable the laptop is 192.168.55.100)

Run from the project ROOT folder:
    python3 sumo/runner.py [--jetson]
=========================================================
"""

import os
import sys
import json
import socket
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from sumo.traci_client import TraciClient
from sumo.signal_controller import SignalController

PORT = 5555
MAX_STEPS = 3600


def get_action_local(dqn, state):

    import numpy as np
    action, _ = dqn.decide(np.array(state, dtype="float32"))
    return action


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--jetson", action="store_true",
                        help="wait for Jetson over network (HIL mode)")
    args = parser.parse_args()

    client = TraciClient()
    client.start()
    signal = SignalController(client)

    conn = None
    f = None
    dqn = None

    if args.jetson:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", PORT))
        srv.listen(1)
        print("Waiting for Jetson on port %d ..." % PORT)
        conn, addr = srv.accept()
        f = conn.makefile("rw")
        print("Jetson connected from %s — it is now in control!" % addr[0])
    else:
        from core.dqn_controller import DQNController
        dqn = DQNController()
        dqn.load_model()
        print("LOCAL mode — DQN running on this machine.")

    total_wait = 0.0

    try:
        while client.step_count < MAX_STEPS:

            state = client.get_state(signal.time_in_phase)

            if args.jetson:
                f.write(json.dumps({"state": state,
                                    "step": client.step_count}) + "\n")
                f.flush()
                line = f.readline()
                if not line:
                    print("Jetson disconnected.")
                    break
                action = int(json.loads(line)["action"])
            else:
                action = get_action_local(dqn, state)

            result = signal.apply(action)
            client.step(5)
            total_wait += client.total_waiting_time()

            q = [int(s * 100) for s in state[:4]]
            print("t=%4ds | N:%3d S:%3d E:%3d W:%3d | %s"
                  % (client.step_count, q[0], q[1], q[2], q[3], result))

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        if f is not None:
            f.close()
        if conn is not None:
            conn.close()
        client.close()

    print("\nRESULT — accumulated waiting time: %.0f s" % total_wait)
    print("(Run once in --jetson mode and once with a fixed timer "
          "to compare KPIs for your report.)")


if __name__ == "__main__":
    main()
