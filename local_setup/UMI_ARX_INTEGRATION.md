# UMI + ARX 控制环境与 Stage 2 接入：历史核查记录

> 本文保留实现前的环境与接口核查，文中的接入缺口和复用建议不代表当前方案。当前部署使用 RLinf 原生 ARX/UVC 适配，SDK 服务解释器为 `/media/lenovo/data/miniconda3/envs/arx-py310/bin/python`；`umi-arx-deploy` 仅作为历史数据处理参考。最新启动与操作说明见 [ARX Stage 2 指南](../docs/source-zh/rst_source/examples/embodied/arx5_rlt.rst)。

本文记录本机 UMI + ARX 控制环境的复用方式，以及接入 RLinf RLT Stage 2 仍需完成的接口工作。核查日期为 2026-09-23；核查对象是当前 `codex/umi-rlt-sft-resume` 分支和本地 `umi-arx-deploy` 源码。此次只验证软件依赖与模块导入，没有连接机器人、打开相机或启动训练。

## 复用本机已有控制环境

保留现有 Python 3.10 控制环境，让新的 RLinf 训练环境独立管理模型依赖。这样可以继续使用已安装的 ARX 控制、相机与共享内存组件，无需在训练环境安装 Franka SDK、ROS 或另一套 ARX SDK。

| 项目 | 已核实的位置或版本 |
| --- | --- |
| UMI 控制项目 | `/home/lenovo/mason/code/umi-arx-deploy` |
| 控制环境 Python | `/media/lenovo/data/miniconda3/envs/umi-arx/bin/python`，3.10.21 |
| Conda 环境 | `/media/lenovo/data/miniconda3/envs/umi-arx` |
| OpenPI websocket 客户端源码 | `/home/lenovo/mason/code/umi-arx-deploy/packages/openpi-client/src` |
| 现有 OpenPI 模型项目 | `/home/lenovo/mason/code/openpi-umi`；本次未改动其 `.venv` |

控制环境中查到的包版本如下。这是本机现状记录，不是要求升级其他环境的依赖清单。

| 包 | 版本 | 包 | 版本 |
| --- | --- | --- | --- |
| numpy | 1.24.4 | scipy | 1.15.2 |
| pyzmq | 27.2.0 | websockets | 16.1.1 |
| msgpack | 1.2.2 | click | 8.4.2 |
| pynput | 1.8.2 | opencv-python | 5.0.0 |
| av | 16.0.1 | threadpoolctl | 3.6.0 |
| zarr | 2.18.3 | numcodecs | 0.13.1 |
| atomics | 1.0.3 | spnav | 0.9 |

本机控制环境未安装 `openpi-client` distribution，但项目脚本会把上述客户端源码目录加入 `sys.path`，实际导入已通过。不要为此直接在现有控制环境安装整个 `openpi-umi`：其 `pyproject.toml` 要求 Python >= 3.11，并含独立的 Torch/JAX 依赖。`modules/replay_buffer.py` 使用 Zarr 2 API，复用时应保持现有 Zarr 2 环境。

项目依赖来源是 [umi_arx_environment.yaml](/home/lenovo/mason/code/umi-arx-deploy/umi_arx_environment.yaml) 和 [OpenPI 客户端 pyproject.toml](/home/lenovo/mason/code/umi-arx-deploy/packages/openpi-client/pyproject.toml)。前者规定 Python 3.10、NumPy 1.24.4，并列出 UVC 视频、SpaceMouse、ZMQ 和共享内存依赖。

## 当前可复用的接口

| 已有组件 | 可复用的职责 | 源码位置 |
| --- | --- | --- |
| `Arx5Env` | 相机和机器人观测同步、`get_obs()`、带时间戳的 `exec_actions()`、episode 记录 | [arx5_env.py](/home/lenovo/mason/code/umi-arx-deploy/modules/arx5_env.py:20) |
| `Arx5Controller` | 控制子进程、共享内存状态、轨迹调度 | [arx5_controller.py](/home/lenovo/mason/code/umi-arx-deploy/modules/arx5_controller.py:32) |
| `Arx5Client` | ZMQ `GET_STATE`、末端动作与控制服务通信 | [arx5_zmq_client.py](/home/lenovo/mason/code/umi-arx-deploy/modules/arx5_zmq_client.py:143) |
| UMI 输入/输出转换 | 相对位姿、6D rotation 表示、10 维模型动作到控制器位姿的转换 | [real_inference_util.py](/home/lenovo/mason/code/umi-arx-deploy/utils/real_inference_util.py:194) |
| RLT websocket 包装 | decision ID、episode 上下文、逐步 reward、终止反馈和已提交动作记录 | [rlt_runtime_evan.py](/home/lenovo/mason/code/umi-arx-deploy/utils/rlt_runtime_evan.py:88) |

`rlt_runtime_evan.py` 定义了 `rlt_context`、`rlt_feedback`、`rlt_decision_id` 和 `executed_physical_action` 等字段，并提供 S/F 成功失败标记。此次源码搜索没有发现 `scripts/` 内的直接导入，也没有在 RLinf 中找到与这些字段对应的 websocket endpoint，因此文件存在不代表已经接通在线训练。后续可复用其协议设计，但仍需把反馈转换为 RLinf replay buffer 使用的 transition，并接入 rollout/actor 权重同步。

## 当前 RLinf 分支的接入缺口

当前分支已提供 `pi05_umi_v4` 数据与模型转换，以及 UMI RLT Stage 1 配置；机器人注册列表和真机任务中没有 ARX 驱动或 ARX environment。核查位置是 [robots/__init__.py](/home/lenovo/mason/code/rlinf_evo/rlinf/robotics/robots/__init__.py:20)、[真机 environment 注册入口](/home/lenovo/mason/code/rlinf_evo/rlinf/envs/real/__init__.py:34) 和 [UMI Stage 1 配置](/home/lenovo/mason/code/rlinf_evo/examples/sft/config/umi_rlt_stage1_sft_openpi_pi05.yaml:41)。

[现有 Stage 2 示例](/home/lenovo/mason/code/rlinf_evo/examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml:20) 使用双节点 Franka 部署、7 维动作、19 维 proprio 和 20 步参考动作。不能只修改机器人地址就将它用于 UMI + ARX。训练环境安装完成后，仍须补齐以下内容：

1. **完整的 UMI Stage 1 checkpoint。** 提供训练后的 Pi0.5 和 RLT 模块权重、与训练一致的模型配置及 `assets/norm_stats.json`。不能仅用未训练 RLT 的基座模型替代 Stage 1 结果。模型加载器支持 `model.safetensors` 或 RLinf `full_weights.pt`，实际目录解析见 [openpi_rlinf/__init__.py](/home/lenovo/mason/code/rlinf_evo/rlinf/models/embodiment/openpi_rlinf/__init__.py:34)。Stage 1 路径属于 `rollout.rlt_feature_model.model_path`；`runner.resume_dir` 用于完整 Stage 2 训练恢复。
2. **UMI 输入 shape 与 Stage 2 proprio 定义。** `robot0_eef_pos` 为 `(2, 3)`；`robot0_eef_rot_axis_angle` 和 `robot0_eef_rot_axis_angle_wrt_start` 均为 `(2, 6)`；夹爪为 `(2, 1)`。这些字段构成两个 16 维状态，模型输入展平为 32 维；腕部图像为 `224×224` RGB。Stage 2 采用哪部分状态作为 proprio，必须由接入代码明确，不能沿用 Franka 的 19。精确校验见 [umi_policy.py](/home/lenovo/mason/code/rlinf_evo/rlinf/models/embodiment/openpi/policies/umi_policy.py:27)。
3. **动作语义与执行窗口。** UMI Stage 1 的物理动作是 `(16, 10)`，模型内部补齐到 32 维；10 维表示位置、6D rotation 和夹爪。需保持训练时的 XYZ 归一化、相对/绝对坐标定义、TCP/末端偏移、夹爪单位与方向。控制接口接收的位姿表示需经过转换，不能把 10 维数组直接当作 Franka 7 维动作。参考窗口为 16 步，在线实际执行几步、何时重新规划及哪些动作进入 transition 也需一致。
4. **RLinf 与控制进程之间的适配。** 复用上述 ZMQ/UVC 控制接口，新增 RLinf environment wrapper 或兼容现有 websocket 协议的服务，将原始观测交给冻结 Stage 1 特征模型，再生成 `z_rl`、`proprio`、`ref_chunk`。还需核对动作回传、已执行动作记录、episode 边界和连接关闭行为。
5. **任务反馈和人工控制。** 明确成功/失败 reward、超时处理、reset 流程、人工接管与恢复方式，以及接管动作如何写入 replay buffer。已有 S/F 包装和 RLinf `rlt_policy_switch` 只是不同接口的参考实现，不能假定语义一致。节点数、GPU placement、控制服务地址和相机映射也需要按本机填写。

这些是后续集成工作，此次环境安装没有实现或验证完整在线闭环，因此此文不提供可直接启动 ARX Stage 2 训练的命令。

## 已完成的非运动验证

使用现有控制 Python，仅导入模块而不创建机器人或相机对象：NumPy、SciPy、ZMQ、websockets、msgpack、OpenCV、PyAV、Zarr、numcodecs、atomics、spnav、OpenPI websocket 客户端，以及 `arx5_zmq_client`、`arx5_controller`、`arx5_env` 均通过。PyAV 的 codec 列表包含 `hevc_nvenc`，但没有测试实际编码。系统已有 `lsusb`、`candump`、`libspnav`、`libusb`、`libGL` 和 `libEGL`。

当前工具进程没有 `DISPLAY`，默认导入键盘模块会报 pynput 无法连接 X server。使用 `PYNPUT_BACKEND=dummy` 后，`peripherals.keystroke_counter` 和 `utils.rlt_runtime_evan` 导入通过；这只说明依赖可加载，真实键盘输入仍需桌面会话。可重复执行以下检查，它不会创建控制对象：

```bash
PYNPUT_BACKEND=dummy PYTHONDONTWRITEBYTECODE=1 \
  /media/lenovo/data/miniconda3/envs/umi-arx/bin/python - <<'PY'
import importlib
import sys

sys.path.insert(0, "/home/lenovo/mason/code/umi-arx-deploy")
sys.path.insert(0, "/home/lenovo/mason/code/umi-arx-deploy/packages/openpi-client/src")
for name in (
    "numpy", "scipy", "zmq", "websockets", "msgpack", "cv2", "av",
    "zarr", "numcodecs", "atomics", "spnav",
    "openpi_client.websocket_client_policy",
    "modules.arx5_zmq_client", "modules.arx5_controller", "modules.arx5_env",
    "peripherals.keystroke_counter", "utils.rlt_runtime_evan",
):
    importlib.import_module(name)
    print("PASS", name)
PY
```

不要用创建对象代替上述导入检查：`Arx5Env.__init__()` 会重置 Elgato USB；`Arx5Client.__init__()` 会连接控制服务并请求状态；`SingleArmClient.__init__()` 会初始化 ARX SDK。它们均未在此次检查中调用。
