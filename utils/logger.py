"""
Traffic Logger — writes every DQN decision to a CSV file.
Use outputs/traffic_log.csv for your KPI analysis
(waiting time, switches per hour, queue trends).
"""

import csv
import time

import config


class TrafficLogger:

    HEADER = ["timestamp", "north", "south", "east", "west",
              "spillback", "phase", "action", "q_keep", "q_switch",
              "yolo_ms"]

    def __init__(self):

        self.file = open(str(config.LOG_FILE), "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.HEADER)
        self.rows = 0

    def log(self, counts, phase, action_str, q_values, yolo_seconds):

        self.writer.writerow([
            round(time.time(), 2),
            counts.get("north", 0),
            counts.get("south", 0),
            counts.get("east", 0),
            counts.get("west", 0),
            counts.get("center", 0),
            phase,
            action_str,
            round(q_values[0], 4),
            round(q_values[1], 4),
            round(yolo_seconds * 1000, 1),
        ])
        self.rows += 1

        if self.rows % 20 == 0:
            self.file.flush()

    def close(self):

        self.file.close()
