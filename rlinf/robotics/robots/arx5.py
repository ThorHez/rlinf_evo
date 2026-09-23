# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ARX robot composition and scheduler resource configuration."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ..discovery import RobotConfig
from ..parts.arms import Arm
from ..parts.cameras import Camera
from ..robot import Robot


class Arx5Robot(Robot):
    """One ARX service connection, its shared gripper, and named cameras."""

    ROBOT_TYPE = "ARX5"
    BACKEND = "arx5_zmq"

    @classmethod
    def build(
        cls,
        *,
        endpoint: str,
        node_rank: Optional[int] = None,
        backend: Optional[str] = None,
        timeout_ms: int = 1000,
        max_state_age: float = 0.25,
        cameras: Optional[Mapping[str, Any]] = None,
        camera_node_rank: Optional[int] = None,
        worker_rank: int = 0,
        env_idx: int = 0,
    ) -> "Arx5Robot":
        """Declare the service client and camera resources without opening them."""
        return cls(
            arm=Arm.backend(backend or cls.BACKEND)(
                endpoint,
                timeout_ms=timeout_ms,
                max_state_age=max_state_age,
                node_rank=node_rank,
                worker_name=f"{cls.ROBOT_TYPE}Arm-{worker_rank}-{env_idx}",
            ),
            **Camera.declare(cameras, node_rank=camera_node_rank),
        )


@dataclass
class Arx5Config(RobotConfig):
    """Local ARX service endpoint and optional UVC camera identifiers."""

    endpoint: str = "tcp://127.0.0.1:5557"
    backend: Optional[str] = None
    timeout_ms: int = 1000
    max_state_age: float = 0.25
    camera_serials: Optional[list[str]] = None
    camera_type: str = "uvc"
    controller_node_rank: Optional[int] = None


Arx5Robot.register_type(Arx5Config)
