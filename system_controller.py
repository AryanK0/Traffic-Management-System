# Coordinates YOLO -> DQN -> Traffic Manager (vision pipeline)
"""
=========================================================
System Controller
=========================================================

Orchestrates the full pipeline on the Jetson:

    VideoIO -> VehicleDetector (YOLO) -> LaneCounter
    -> StateBuilder -> DQNController -> TrafficManager
    -> Visualizer + Logger

For the SUMO hardware-in-the-loop mode, see sumo/runner.py.

Author : Traffic AI Team
Platform : NVIDIA Jetson Nano
=========================================================
"""

import time

import config
from core.detector import VehicleDetector
from core.lane_counter import LaneCounter
from core.state_builder import StateBuilder
from core.dqn_controller import DQNController
from core.traffic_manager import TrafficManager
from core.video_io import VideoIO
from core.visualizer import Visualizer
from utils.fps import FPSCounter
from utils.logger import TrafficLogger


class SystemController:

    def __init__(self):

        self.detector = VehicleDetector(
            config.YOLO_MODEL_PATH, device=config.DEVICE
        )
        self.lane_counter = LaneCounter()
        self.state_builder = StateBuilder()
        self.dqn = DQNController()
        self.manager = TrafficManager()
        self.video = VideoIO()
        self.visualizer = Visualizer()
        self.fps = FPSCounter()
        self.logger = TrafficLogger()

        self.display_ok = config.DISPLAY_OUTPUT
        self.last_decision_time = 0.0

    ########################################################
    # Initialize
    ########################################################

    def initialize(self):

        config.print_configuration()

        print("[1/3] Loading YOLO model...")
        self.detector.load_model()
        print("      OK — classes: %s" % self.detector.class_names)

        print("[2/3] Loading DQN model...")
        self.dqn.load_model()
        print("      OK — state=%d actions=%d"
              % (config.STATE_SIZE, config.ACTION_SIZE))

        print("[3/3] Opening video source...")
        source = self.video.open()
        if source is None:
            raise RuntimeError(
                "No video source. Put a file at %s or plug in a camera."
                % config.INPUT_VIDEO
            )
        print("      OK — %s" % source)

    ########################################################
    # Run
    ########################################################

    def run(self):

        print("\nSystem online. Ctrl+C to stop.")
        print("=" * 70)

        try:
            while True:

                self.manager.update()

                frame = self.video.read()
                if frame is None:
                    print("End of video / camera stream.")
                    break

                # 1. Perception
                detections = self.detector.detect(frame)
                counts = self.lane_counter.count(
                    detections, frame.shape
                )

                # 2. State
                state = self.state_builder.build(
                    counts, self.manager.time_in_phase()
                )

                # 3. Decision (rate-limited to MIN_GREEN pace)
                now = time.time()
                action_str = "..."
                if now - self.last_decision_time >= config.MIN_GREEN_TIME:
                    action, q = self.dqn.decide(state)
                    action_str = self.manager.apply(action)
                    self.last_decision_time = now

                    self.logger.log(
                        counts, self.manager.current_phase,
                        action_str, q,
                        self.detector.get_inference_time()
                    )

                    print("N:%3d S:%3d E:%3d W:%3d | Spill:%2d | "
                          "Phase %-8s %5.1fs | %s | YOLO %5.1f ms" % (
                              counts["north"], counts["south"],
                              counts["east"], counts["west"],
                              counts["center"],
                              self.manager.current_phase,
                              self.manager.time_in_phase(),
                              action_str,
                              self.detector.get_inference_time() * 1000))

                # 4. Visualize
                self.fps.tick()
                annotated = self.visualizer.annotate(
                    frame, detections, counts,
                    self.manager.current_phase,
                    action_str, self.fps.get()
                )
                self.video.write(annotated)

                if self.display_ok:
                    try:
                        import cv2
                        cv2.imshow("Traffic AI", annotated)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    except Exception:
                        # Headless (SSH) — no display available
                        self.display_ok = False

        except KeyboardInterrupt:
            print("\nStopped by user (Ctrl+C)")

    ########################################################
    # Shutdown
    ########################################################

    def shutdown(self):

        self.video.release()
        self.logger.close()

        print("\nSession summary")
        print("  Frames processed : %d" % self.detector.get_total_frames())
        print("  Total detections : %d" % self.detector.get_total_detections())
        print("  DQN decisions    : %d" % self.dqn.total_decisions)
        print("  Signal switches  : %d" % self.manager.total_switches)
        print("  Log file         : %s" % config.LOG_FILE)
