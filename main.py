# Main application entry
"""
=========================================================
Traffic AI using YOLOv11 + DQN + SUMO
Main Entry Point

Author : Team Project
Platform : NVIDIA Jetson Nano

=========================================================
"""

from system_controller import SystemController


def main():
    """
    Application Entry Point
    """

    print("\nStarting Traffic AI System...\n")

    controller = SystemController()

    controller.initialize()

    controller.run()

    controller.shutdown()


if __name__ == "__main__":
    main()