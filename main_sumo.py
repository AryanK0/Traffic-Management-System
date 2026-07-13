"""
=========================================================
main_sumo.py — RUNS ON THE JETSON (hardware-in-the-loop)
=========================================================

The laptop runs SUMO + `python3 sumo/runner.py --jetson`.
This script connects to it, receives each traffic state,
lets the Jetson's DQN decide KEEP/SWITCH, and sends the
decision back. The SUMO traffic light obeys the Jetson.

Run:
    python3 main_sumo.py 192.168.55.100
    (192.168.55.100 = laptop's IP over the USB cable)
=========================================================
"""

import sys
import json
import socket

import numpy as np

from core.dqn_controller import DQNController, ACTION_NAMES

PORT = 5555


def main():

    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"

    print("[LOADING] DQN model...")
    dqn = DQNController()
    dqn.load_model()
    print("          OK")

    print("[CONNECT] SUMO bridge at %s:%d ..." % (host, PORT))
    sock = socket.create_connection((host, PORT), timeout=30)
    f = sock.makefile("rw")
    print("          Connected — Jetson now controls the SUMO light!\n")

    try:
        while True:

            line = f.readline()
            if not line:
                print("SUMO simulation ended.")
                break

            state = np.array(json.loads(line)["state"], dtype=np.float32)

            action, q = dqn.decide(state)

            f.write(json.dumps({"action": action}) + "\n")
            f.flush()

            qn = [int(s * 100) for s in state[:4]]
            print("Decision %4d | N:%3d S:%3d E:%3d W:%3d | Spill:%3d | "
                  "Q[keep=%.2f switch=%.2f] -> %s"
                  % (dqn.total_decisions, qn[0], qn[1], qn[2], qn[3],
                     int(state[4] * 100), q[0], q[1], ACTION_NAMES[action]))

    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        f.close()
        sock.close()

    print("Total decisions made by Jetson: %d" % dqn.total_decisions)


if __name__ == "__main__":
    main()
