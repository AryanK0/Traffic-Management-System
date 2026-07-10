"""
FPS Counter — rolling average over the last N frames.
"""

import time
from collections import deque


class FPSCounter:

    def __init__(self, window=30):

        self.times = deque(maxlen=window)

    def tick(self):

        self.times.append(time.time())

    def get(self):

        if len(self.times) < 2:
            return 0.0

        span = self.times[-1] - self.times[0]

        if span <= 0:
            return 0.0

        return (len(self.times) - 1) / span
