# Third-Party Dependencies

This directory contains dependencies based on pinned upstream revisions. Each dependency keeps its upstream license; local source fixes are documented below, and RLinf integration settings live outside the dependency directory.

## ARX5 SDK

The `arx5-sdk` directory is based on [real-stanford/arx5-sdk](https://github.com/real-stanford/arx5-sdk) at commit [`c160db7834530a3d80fec524547c74e1d4fa8035`](https://github.com/real-stanford/arx5-sdk/commit/c160db7834530a3d80fec524547c74e1d4fa8035), dated 2026-10-02, imported on 2026-10-08, with local emergency-handling fixes. It includes the upstream MIT [LICENSE](arx5-sdk/LICENSE), robot models, and upstream runtime libraries under `lib/`. Generated Python bindings and build directories remain local build artifacts.

The local fixes release state and command locks before entering emergency handling and keep emergency CAN sends outside those locks. Startup validation failures raise runtime errors, failed construction releases its logger, and background emergency damping can stop during destruction. After a fault, both Joint and Cartesian controllers reject new motion commands; resolve the underlying fault and recreate the controller or restart the service. Normal Cartesian IK behavior and SDK default parameters remain unchanged.

RLinf's service defaults to this patched SDK. An external SDK checkout can still be selected with `--sdk-path`, but it does not receive these source fixes. The separate `/home/lenovo/kian/arx5-sdk` checkout is unchanged.

### Build the SDK

Build with the Python environment that will run the SDK service. The upstream conda definitions under `arx5-sdk/conda_environments/` list the required C++ libraries and Python packages. On the current Lenovo host, the existing `arx-py310` environment already provides these dependencies.

For a fresh build, start from the RLinf repository root and follow the upstream README's CMake and Make procedure:

```bash
conda activate arx-py310
cd third_party/arx5-sdk
mkdir build && cd build
cmake ..
make -j
```

This builds all default targets, including the Python extension and C++ executables. It writes `arx5_interface.cpython-<version>-<architecture>.so` into `third_party/arx5-sdk/python/`. For an existing build directory, enter it and run `cmake ..` followed by `make -j`. Run the service with the same Python version. Keep the build directory directly under the SDK source root, as the upstream fallback for locating robot models uses that layout. Rebuild after relocating the SDK; compiled library search paths may reference the original directory.

### Run the RLinf Service

From the RLinf repository root, inspect the effective settings before opening the controller:

```bash
python toolkits/realworld_check/arx5_server.py X5 can0 --print-config
```

`--print-config` prints merged settings without importing the SDK or connecting to CAN. The normal startup command accepts the same positional arguments, plus explicit motion permission:

```bash
python toolkits/realworld_check/arx5_server.py X5 can0 --allow-motion
```

The service starts with policy commands disabled until an operator enables them. It loads the SDK's `cartesian_controller` configuration and constructs `Arx5CartesianController`. Its controller constructor starts the SDK control loop. RLinf checks end-effector step limits, deadlines, control epochs, and feedback before sending end-effector targets through `set_eef_cmd`. IK retries and normal joint handling remain in the SDK. The original upstream `python/communication/zmq_server.py` uses a different protocol and cannot replace this service for RLinf collection.

### Stop the Service

Support the arm and payload before stopping the service. The SDK runs directly in the service process. During service operation, `Ctrl+C` or `SIGTERM` requests `service.stop()`; the SDK handles normal cleanup. Startup validation errors are reported by the patched SDK as runtime errors. Resolve the underlying error before restarting, and use the SDK built under `third_party/arx5-sdk` to receive the emergency-handling fixes.

### Override Device Settings

The SDK's `cartesian_controller` configuration supplies the controller gains and 0.1-second interpolation preview. The empty `controller` section in `toolkits/realworld_check/arx5_server_defaults.json` preserves these SDK defaults. That file sets a 10-second command timeout and per-command limits of 0.05 m translation, 0.2 rad rotation, and 0.02 m gripper change. These are software defaults, not measured calibration values. The default adapter uses the SDK EEF as the TCP and preserves metre-based gripper widths.

Use optional `--config /path/to/device.json` to override individual fields in `robot`, `controller`, `adapter`, `service`, or `limits`. An explicitly supplied `calibration_confirmed: false` still prevents motion; a missing confirmation flag does not require a separate calibration declaration.

For the current Lenovo X5, `local_setup/arx5_lenovo.json` contains empty `robot` and `controller` objects. It uses the official X5 and `cartesian_controller` defaults: `gripper_open_readout=5.03`, `gripper_torque_max=1.5`, and `default_gripper_kp=5.0`. The startup command keeps the configuration path explicit:

```bash
python toolkits/realworld_check/arx5_server.py X5 can0 \
  --config local_setup/arx5_lenovo.json --allow-motion
```

Because this file supplies no overrides, omitting `--config` gives the same settings. The SDK supplies the robot and controller defaults; RLinf still supplies its service protocol, command timeout, and end-effector step limits described above. Task-specific TCP calibration, model timing, and camera latency remain separate deployment settings. The full local collection workflow is in `local_setup/ARX5_ROUND1_STARTUP.md`.
