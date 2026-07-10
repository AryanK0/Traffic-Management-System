"""
=========================================================
Video I/O
=========================================================

Handles all frame input/output:
    • Read from a video file (videos/traffic.mp4)
    • Or from a live camera (/dev/video0)
    • Optionally write the annotated output video

Author : Traffic AI Team
Platform : NVIDIA Jetson Nano
=========================================================
"""

import cv2

import config


class VideoIO:

    def __init__(self):

        self.cap = None
        self.writer = None
        self.source = None
        self.frame_count = 0

    ########################################################
    # Open Input
    ########################################################

    def open(self):
        """
        Tries the input video file first; falls back to the
        live camera. Returns the source name or None.
        """

        if config.INPUT_VIDEO.exists():
            self.cap = cv2.VideoCapture(str(config.INPUT_VIDEO))
            if self.cap.isOpened():
                self.source = "video:%s" % config.INPUT_VIDEO.name
                return self.source

        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.source = "camera:/dev/video0"
            return self.source

        self.cap = None
        self.source = None
        return None

    ########################################################
    # Read Frame
    ########################################################

    def read(self):

        if self.cap is None:
            return None

        for _ in range(max(0, config.FRAME_SKIP - 1)):
            self.cap.grab()

        ret, frame = self.cap.read()

        if not ret:
            return None

        self.frame_count += 1
        return frame

    ########################################################
    # Write Output
    ########################################################

    def write(self, frame):

        if not config.SAVE_OUTPUT:
            return

        if self.writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                str(config.OUTPUT_VIDEO), fourcc, 10.0, (w, h)
            )

        self.writer.write(frame)

    ########################################################
    # Release
    ########################################################

    def release(self):

        if self.cap is not None:
            self.cap.release()

        if self.writer is not None:
            self.writer.release()
            print("Output video saved: %s" % config.OUTPUT_VIDEO)
