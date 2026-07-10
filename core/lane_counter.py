"""
=========================================================
Lane Counter
=========================================================

Converts YOLO detections into vehicle counts for each
approach (north / south / east / west) plus the junction
box (center, used for spillback).

Uses config.LANE_REGIONS — pixel rectangles calibrated
to the camera / video view.

Author : Traffic AI Team
Platform : NVIDIA Jetson Nano
=========================================================
"""

import config


class LaneCounter:

    LANES = ["north", "south", "east", "west", "center"]

    def __init__(self):

        self.reset()

    ########################################################
    # Reset Counts
    ########################################################

    def reset(self):

        self.counts = {lane: 0 for lane in self.LANES}
        self.counts["total"] = 0

    ########################################################
    # Count Vehicles
    ########################################################

    def count(self, detections, frame_shape=None):
        """
        detections : list of dicts from VehicleDetector.detect()
        frame_shape : (h, w) of the frame, used to scale the
                      LANE_REGIONS (which are defined for the
                      reference resolution in config).
        """

        self.reset()

        sx = sy = 1.0
        if frame_shape is not None:
            h, w = frame_shape[:2]
            sx = w / float(config.REFERENCE_WIDTH)
            sy = h / float(config.REFERENCE_HEIGHT)

        for detection in detections:

            if detection["confidence"] < config.CONFIDENCE_THRESHOLD:
                continue

            cx, cy = detection["center"]

            for lane in self.LANES:

                region = config.LANE_REGIONS.get(lane)
                if region is None:
                    continue

                x1, y1, x2, y2 = region

                if (x1 * sx <= cx <= x2 * sx and
                        y1 * sy <= cy <= y2 * sy):
                    self.counts[lane] += 1
                    break

        self.counts["total"] = sum(
            self.counts[lane] for lane in self.LANES
        )

        return self.counts.copy()

    ########################################################
    # Get Counts
    ########################################################

    def get_counts(self):

        return self.counts.copy()

    ########################################################
    # Print Counts
    ########################################################

    def print_counts(self):

        print("--------------------------------")
        print("Lane Counts")
        print("--------------------------------")

        for lane, value in self.counts.items():
            print("%-10s: %d" % (lane, value))

        print("--------------------------------")
