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

"""Run the ARX SDK owner in Python 3.10, or send an operator mode event.

Examples::

    python arx5_server.py --sdk-path /path/to/arx5-sdk --model X5 --interface can0
    python arx5_server.py --operation state
    python arx5_server.py --operation enter_drag

Service startup never calls reset_to_home or restores active joint gains.
Motion needs --allow-motion plus a calibrated JSON configuration. Mode changes
are explicit. The SDK itself enables its CAN receive/control loop on creation;
this is not a passive CAN sniffer. SDK shutdown may enter damping, so support
the arm and any payload before stopping the owning process.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import sys
import tempfile
import threading
from pathlib import Path


def load_protocol():
    """Load the Python 3.10-compatible protocol without RLinf dependencies."""
    path = (
        Path(__file__).resolve().parents[2]
        / "rlinf/robotics/parts/arms/arx5_protocol.py"
    )
    spec = importlib.util.spec_from_file_location("rlinf_arx5_protocol", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    """Own one controller, or issue one independent operator request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--sdk-path", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--interface")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON with robot/controller/adapter/service/limits settings",
    )
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument(
        "--operation",
        choices=["state", "enter_drag", "resume_hold", "hold", "enable_policy"],
    )
    args = parser.parse_args()
    protocol = load_protocol()
    protocol.local_endpoint(args.endpoint)
    if args.operation:
        client = protocol.Arx5Client(args.endpoint)
        try:
            state = client.request("state")
            result = (
                state
                if args.operation == "state"
                else client.request(
                    args.operation, expected_epoch=state["control_epoch"]
                )
            )
            print(json.dumps(result, indent=2))
        finally:
            client.close()
        return
    if not args.sdk_path or not args.model or not args.interface:
        parser.error("service requires --sdk-path, --model, and --interface")
    config = json.loads(args.config.read_text()) if args.config else {}
    unknown = set(config) - {
        "robot",
        "controller",
        "adapter",
        "service",
        "limits",
        "calibration_confirmed",
    }
    if unknown:
        parser.error(f"unknown configuration keys: {sorted(unknown)}")
    if args.allow_motion and (
        config.get("calibration_confirmed") is not True or not config.get("limits")
    ):
        parser.error(
            "--allow-motion requires calibration_confirmed: true and explicit limits in JSON"
        )
    limits = protocol.MotionLimits(**config["limits"]) if config.get("limits") else None
    # A second RLinf service must fail before opening a second CAN controller.
    digest = hashlib.sha256(args.interface.encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"rlinf-arx5-{os.getuid()}-{digest}.lock"
    with lock_path.open("a") as owner_lock:
        try:
            fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"another RLinf SDK service owns {args.interface}")
        sdk_directory = (args.sdk_path / "python").resolve()
        if not sdk_directory.is_dir():
            parser.error(f"SDK Python directory does not exist: {sdk_directory}")
        sys.path.insert(0, str(sdk_directory))
        import arx5_interface as sdk

        robot = sdk.RobotConfigFactory.get_instance().get_config(args.model)
        controller_config = sdk.ControllerConfigFactory.get_instance().get_config(
            "joint_controller", robot.joint_dof
        )
        for target, overrides in (
            (robot, config.get("robot", {})),
            (controller_config, config.get("controller", {})),
        ):
            for key, value in overrides.items():
                if (
                    key.startswith("_")
                    or not hasattr(target, key)
                    or callable(getattr(target, key))
                ):
                    parser.error(f"unknown SDK setting {key}")
                setattr(target, key, value)
        if not controller_config.background_send_recv:
            parser.error("ARX service requires background_send_recv")
        if (
            args.allow_motion
            and controller_config.default_preview_time
            <= controller_config.controller_dt
        ):
            parser.error(
                "motion requires controller.default_preview_time > controller_dt for SDK interpolation"
            )
        solver = sdk.Arx5Solver(
            robot.urdf_path,
            robot.joint_dof,
            robot.joint_pos_min,
            robot.joint_pos_max,
            robot.base_link_name,
            robot.eef_link_name,
            robot.gravity_vector,
        )
        controller = sdk.Arx5JointController(robot, controller_config, args.interface)
        adapter = protocol.SdkArx5Adapter(
            sdk, controller, solver, **config.get("adapter", {})
        )
        service = protocol.Arx5Service(
            adapter,
            allow_motion=args.allow_motion,
            limits=limits,
            **config.get("service", {}),
        )

        def stop(_signum, _frame):
            # Avoid joining a sampler while a signal interrupts SDK code.
            threading.Thread(target=service.stop, daemon=True).start()

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        print(
            f"ARX service listening on {args.endpoint}; mode=disabled; motion_allowed={args.allow_motion}",
            flush=True,
        )
        try:
            service.serve(args.endpoint)
        finally:
            service.stop()


if __name__ == "__main__":
    main()
