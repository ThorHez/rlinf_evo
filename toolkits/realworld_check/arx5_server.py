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

    python arx5_server.py X5 can0 --allow-motion
    python arx5_server.py X5 can0 --config /path/to/overrides.json --print-config
    python arx5_server.py --operation state
    python arx5_server.py --operation enter_drag

Service startup never calls reset_to_home or restores active joint gains.
Motion needs --allow-motion. The bundled JSON supplies software motion limits
and timing defaults, not measured calibration; --config optionally overrides
them. Robot and tool coordinates use the SDK defaults unless overridden.
Mode changes are explicit. The SDK enables its CAN receive/control loop on
creation; this is not a passive CAN sniffer. SDK shutdown may enter damping,
so support the arm and any payload before stopping the owning process.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
import signal
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("arx5_server_defaults.json")
CONFIG_SECTIONS = ("robot", "controller", "adapter", "service", "limits")


def load_protocol() -> ModuleType:
    """Load the Python 3.10-compatible protocol without RLinf dependencies."""
    path = REPOSITORY_ROOT / "rlinf/robotics/parts/arms/arx5_protocol.py"
    spec = importlib.util.spec_from_file_location("rlinf_arx5_protocol", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ARX protocol from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_config(path: Path | None) -> dict[str, Any]:
    """Merge optional section overrides onto the bundled software defaults."""
    config = json.loads(DEFAULT_CONFIG.read_text())
    if path is not None:
        overrides = json.loads(path.read_text())
        if not isinstance(overrides, dict):
            raise ValueError("configuration must be a JSON object")
        unknown = set(overrides) - set(CONFIG_SECTIONS) - {"calibration_confirmed"}
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        for section in CONFIG_SECTIONS:
            if section in overrides:
                if not isinstance(overrides[section], dict):
                    raise ValueError(f"{section} must be a JSON object")
                config[section].update(overrides[section])
        if "calibration_confirmed" in overrides:
            if type(overrides["calibration_confirmed"]) is not bool:
                raise ValueError("calibration_confirmed must be a boolean")
            config["calibration_confirmed"] = overrides["calibration_confirmed"]
    return config


def load_sdk(sdk_path: Path) -> ModuleType:
    """Load a compiled binding only from the selected SDK Python directory."""
    sdk_directory = (sdk_path / "python").resolve()
    context = (
        f"SDK directory: {sdk_directory}; interpreter: {sys.executable} "
        f"(Python {sys.version_info.major}.{sys.version_info.minor})"
    )
    spec = importlib.machinery.PathFinder.find_spec(
        "arx5_interface", [str(sdk_directory)]
    )
    if (
        spec is None
        or not isinstance(spec.loader, importlib.machinery.ExtensionFileLoader)
        or spec.origin is None
        or Path(spec.origin).resolve().parent != sdk_directory
    ):
        bindings = sorted(
            path.name for path in sdk_directory.glob("arx5_interface*.so")
        )
        raise ImportError(
            f"No compatible compiled arx5_interface binding. {context}. "
            f"Available bindings: {bindings}. Build the SDK with this interpreter "
            "or use the interpreter that matches the compiled binding."
        )
    existing = sys.modules.get("arx5_interface")
    if existing is not None:
        if (
            Path(getattr(existing, "__file__", "")).resolve()
            != Path(spec.origin).resolve()
        ):
            raise ImportError(
                f"A different arx5_interface is already loaded. {context}"
            )
        return existing
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    except (ImportError, OSError) as exc:
        sys.modules.pop(spec.name, None)
        raise ImportError(
            f"Cannot load compiled ARX binding. {context}. {exc}"
        ) from exc
    return module


def _validate_settings(config: dict[str, Any]) -> None:
    positive_fields = {
        "service": {"sample_hz", "history_seconds", "state_timeout", "command_timeout"},
        "adapter": {"gripper_scale", "feedback_timeout", "restore_tolerance"},
        "limits": {"translation", "rotation", "gripper"},
    }
    allowed = {
        "service": positive_fields["service"],
        "adapter": positive_fields["adapter"]
        | {"eef_to_tcp", "gripper_offset", "drag_kd", "drag_gripper"},
        "limits": positive_fields["limits"],
    }
    for section, fields in allowed.items():
        unknown = set(config[section]) - fields
        if unknown:
            raise ValueError(f"unknown {section} settings: {sorted(unknown)}")
        for key in positive_fields[section] & config[section].keys():
            value = config[section][key]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{section}.{key} must be finite and positive")
    adapter = config["adapter"]
    if "gripper_offset" in adapter:
        value = adapter["gripper_offset"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("adapter.gripper_offset must be finite")
    if adapter.get("drag_gripper", "hold") not in ("hold", "manual"):
        raise ValueError("adapter.drag_gripper must be hold or manual")


def _validate_adapter_geometry(config: dict[str, Any], joint_dof: int) -> None:
    import numpy as np

    transform = config.get("eef_to_tcp")
    if transform is not None:
        matrix = np.asarray(transform, dtype=float)
        if (
            matrix.shape != (4, 4)
            or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)
        ):
            raise ValueError("adapter.eef_to_tcp must be a rigid 4x4 transform")
    if config.get("drag_kd") is not None:
        gains = np.asarray(config["drag_kd"], dtype=float)
        if (
            gains.shape != (joint_dof,)
            or not np.isfinite(gains).all()
            or np.any(gains < 0)
        ):
            raise ValueError(
                f"adapter.drag_kd must have {joint_dof} finite nonnegative values"
            )


def main() -> None:
    """Own one controller, or issue one independent operator request."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_arg", nargs="?", metavar="MODEL")
    parser.add_argument("interface_arg", nargs="?", metavar="INTERFACE")
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument(
        "--sdk-path", type=Path, default=REPOSITORY_ROOT / "third_party/arx5-sdk"
    )
    parser.add_argument("--model")
    parser.add_argument("--interface")
    parser.add_argument(
        "--config",
        type=Path,
        help="optional JSON overrides for robot/controller/adapter/service/limits",
    )
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="print effective overrides and SDK path without loading SDK or hardware",
    )
    parser.add_argument(
        "--operation",
        choices=["state", "enter_drag", "resume_hold", "hold", "enable_policy"],
    )
    args = parser.parse_args()
    for name in ("model", "interface"):
        positional = getattr(args, f"{name}_arg")
        option = getattr(args, name)
        if positional is not None and option is not None and positional != option:
            parser.error(f"conflicting {name}: {positional!r} and --{name} {option!r}")
        setattr(args, name, option if option is not None else positional)
    if args.operation and args.print_config:
        parser.error("--operation and --print-config cannot be combined")
    if not args.operation:
        try:
            config = load_config(args.config)
            _validate_settings(config)
        except (OSError, TypeError, ValueError) as exc:
            parser.error(str(exc))
        if args.allow_motion and config.get("calibration_confirmed") is False:
            parser.error(
                "--allow-motion conflicts with explicit calibration_confirmed: false"
            )
        if args.print_config:
            print(
                json.dumps(
                    {
                        "sdk_path": str(args.sdk_path.expanduser().resolve()),
                        "model": args.model,
                        "interface": args.interface,
                        "endpoint": args.endpoint,
                        "allow_motion": args.allow_motion,
                        "defaults_file": str(DEFAULT_CONFIG),
                        "config": config,
                        "note": (
                            "Software defaults and overrides; SDK robot defaults "
                            "apply. These settings do not establish measured calibration."
                        ),
                    },
                    indent=2,
                )
            )
            return
    protocol = load_protocol()
    try:
        protocol.local_endpoint(args.endpoint)
    except ValueError as exc:
        parser.error(str(exc))
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
    if not args.model or not args.interface:
        parser.error("service requires MODEL INTERFACE or --model and --interface")
    limits = protocol.MotionLimits(**config["limits"])
    # A second RLinf service must fail before opening a second CAN controller.
    digest = hashlib.sha256(args.interface.encode()).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"rlinf-arx5-{os.getuid()}-{digest}.lock"
    with lock_path.open("a") as owner_lock:
        try:
            fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f"another RLinf SDK service owns {args.interface}")
        try:
            sdk = load_sdk(args.sdk_path.expanduser())
        except ImportError as exc:
            parser.error(str(exc))

        robot = sdk.RobotConfigFactory.get_instance().get_config(args.model)
        controller_config = sdk.ControllerConfigFactory.get_instance().get_config(
            "cartesian_controller", robot.joint_dof
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
                try:
                    setattr(target, key, value)
                except (TypeError, ValueError) as exc:
                    parser.error(f"invalid SDK setting {key}: {exc}")
        try:
            _validate_adapter_geometry(config["adapter"], robot.joint_dof)
        except (TypeError, ValueError) as exc:
            parser.error(str(exc))
        if not controller_config.background_send_recv:
            parser.error("ARX service requires background_send_recv")
        if (
            not math.isfinite(controller_config.controller_dt)
            or controller_config.controller_dt <= 0
            or not math.isfinite(controller_config.default_preview_time)
        ):
            parser.error(
                "controller_dt must be finite and positive; "
                "default_preview_time must be finite"
            )
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
        controller = sdk.Arx5CartesianController(
            robot, controller_config, args.interface
        )
        adapter = protocol.SdkArx5Adapter(
            sdk, controller, solver, **config.get("adapter", {})
        )
        service = protocol.Arx5Service(
            adapter,
            allow_motion=args.allow_motion,
            limits=limits,
            **config.get("service", {}),
        )

        def stop(_signum: int, _frame: Any) -> None:
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
