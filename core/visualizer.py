"""
=========================================================
Visualizer
=========================================================

Draws everything a human needs to trust the system:
    • lane region rectangles + live counts
    • YOLO bounding boxes + class + confidence
    • current signal phase, DQN decision, FPS

Author : Traffic AI Team
Platform : NVIDIA Jetson Nano
=========================================================
"""

import cv2

import config


class Visualizer:

    def annotate(self, frame, detections, lane_counts,
                 phase, action_str, fps=0.0):

        h, w = frame.shape[:2]
        sx = w / float(config.REFERENCE_WIDTH)
        sy = h / float(config.REFERENCE_HEIGHT)

        # Lane regions + counts
        for lane, region in config.LANE_REGIONS.items():

            if region is None:
                continue

            x1, y1, x2, y2 = region
            p1 = (int(x1 * sx), int(y1 * sy))
            p2 = (int(x2 * sx), int(y2 * sy))

            cv2.rectangle(frame, p1, p2, config.BLUE, 2)

            cv2.putText(
                frame,
                "%s: %d" % (lane, lane_counts.get(lane, 0)),
                (p1[0] + 5, p1[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                config.YELLOW,
                config.FONT_THICKNESS
            )

        # Detections
        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            cv2.rectangle(frame, (x1, y1), (x2, y2), config.GREEN, 2)

            cv2.putText(
                frame,
                "%s %.2f" % (det["class_name"], det["confidence"]),
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                config.GREEN,
                1
            )

        # Status bar
        status = "Phase: %s | DQN: %s" % (phase, action_str)

        if config.SHOW_FPS:
            status += " | FPS: %.1f" % fps

        cv2.putText(
            frame, status, (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, config.RED, 2
        )

        return frame

    ########################################################
    # Screenshot
    ########################################################

    def save_screenshot(self, frame, name):

        if not config.SAVE_SCREENSHOTS:
            return

        path = config.SCREENSHOT_DIR / ("%s.jpg" % name)
        cv2.imwrite(str(path), frame)
