"""Thin pass-through over the harvested rosbridge config.

Wraps roslibpy publishers / services, exposing only what the five
experiment tools need:

    - cmd_vel publisher (drive_robot_forward_raw / rotate_robot_raw)
    - panorama capture (vision_helpers.capture_panoramic_frames path)

No business logic. The config-flag `ros.enabled` toggles whether the
agent's tool wrappers route through this module or stay mock. ROS
integration lands in step 5; step 2 ships the stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class RosBridge:
    host: str = "localhost"
    port: int = 9090

    def connect(self) -> None:
        raise NotImplementedError("ROS bridge is implemented in step 5")

    def drive_forward(self, distance_m: float, speed_mps: float = 0.3) -> None:
        raise NotImplementedError

    def rotate_in_place(self, angle_deg: float, angular_speed_dps: float = 30.0) -> None:
        raise NotImplementedError

    def capture_panorama(self) -> "list":
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError
