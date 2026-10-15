# ARX5 首轮数据采集启动说明

这份说明用于在当前 Lenovo 主机上启动 ARX5 首轮数据采集：加载已训练的 Stage 1 模型，执行参考动作，保存供 Stage 2 使用的 replay 数据。按顺序填写现场参数、启动 SDK 服务，再启动采集程序；文末说明键盘操作、数据位置和结束方法。

适用目录为 `/home/lenovo/mason/code/rlinf_evo`，配置为 `realworld_arx5_rlt_stage2_ac_mlp`。本轮设置 `algorithm.update_epoch=0`，Stage 2 actor 和 Q 网络会初始化，但不执行优化器更新。动作选择保持在 Stage 1 参考分支，因此可以在 Q 网络尚未训练时采集数据。

模型权重、归一化统计和合成观测下的 GPU 推理已于 2026-10-08 检查通过；这些结果覆盖模型加载与计算，真机坐标、时序和运动参数仍需使用现场验证值。

## 1. 准备文件和现场参数

模型与解释器路径已经确定。下面的启动命令使用这些本机文件；SDK 服务使用 Python 3.10，RLinf 使用项目的 Python 3.11 环境。

| 用途 | 路径 |
|---|---|
| Stage 1 模型目录 | `/home/lenovo/mason/code/rlinf_evo/models/pi05_umi_stage1_20000` |
| 完整模型权重 | 模型目录中的 `full_weights.pt` |
| 归一化统计 | 模型目录中的 `norm_stats.json` |
| RLinf 环境激活脚本 | `/home/lenovo/mason/code/rlinf_evo/local_setup/activate.sh` |
| SDK 解释器 | `/media/lenovo/data/miniconda3/envs/arx-py310/bin/python` |
| ARX SDK | `/home/lenovo/mason/code/rlinf_evo/third_party/arx5-sdk` |
| 当前机械臂配置 | `/home/lenovo/mason/code/rlinf_evo/local_setup/arx5_lenovo.json` |

SDK 基于官方提交 `c160db7834530a3d80fec524547c74e1d4fa8035`，工程内加入了 emergency 死锁和退出处理修复。更换机器、Python 版本、移动目录或更新 SDK 源码后，按 [SDK 构建说明](/home/lenovo/mason/code/rlinf_evo/third_party/README.md) 编译 Python 3.10 绑定。启动服务时使用上表中的 SDK 解释器。服务默认使用工程内版本；通过 `--sdk-path` 指向外部目录不会获得这些修复，原来的 `/home/lenovo/kian/arx5-sdk` 保持原样。

### 机械臂服务配置

本次采集通过仓库中的 `arx5_server.py` 接收 RLinf 动作。服务默认加载仓库内的 SDK 和软件配置，`--config` 用来覆盖设备差异。启动形式支持原来熟悉的 `X5 can0` 位置参数，也兼容 `--model X5 --interface can0`。

控制路径使用 SDK 的 `Arx5CartesianController`：服务从配置工厂读取 `cartesian_controller` 参数，通过 `set_eef_cmd` 下发末端目标，IK 重试、关节目标生成和插值由 SDK 处理。本地 emergency 修复保留正常 Cartesian IK 行为、默认关节增益和 0.1 秒插值预览时间。

| 启动方式 | JSON 要求 |
|---|---|
| 直接使用 ARX SDK 的示例或已有控制脚本 | SDK 可按型号加载内置配置；其他要求由具体脚本决定。 |
| `arx5_server.py`，省略 `--allow-motion` | `--config` 可省略，服务使用默认配置并保持禁止 policy 命令；可用于查询状态。 |
| `arx5_server.py --allow-motion` | 可以使用内置软件默认值；JSON 为可选覆盖。当前机械臂使用下面已经准备好的本机配置。 |

`ARX_SERVICE_CONFIG` 只是下面命令自定义的 Bash 路径变量，程序实际读取的是 `--config` 后面的文件路径。也可以直接写 `--config /绝对路径/配置.json`。

当前 [arx5_lenovo.json](/home/lenovo/mason/code/rlinf_evo/local_setup/arx5_lenovo.json) 的 `robot` 和 `controller` 均为空，直接使用官方 X5 和 `cartesian_controller` 默认配置。其中 `robot.gripper_open_readout=5.03`、`robot.gripper_torque_max=1.5`、`controller.default_gripper_kp=5.0`，已移除先前从本机旧脚本迁移的三项覆盖。

下面的命令保留本机配置路径；该文件当前没有覆盖任何参数，因此也可以省略 `--config`。SDK 提供机械臂和控制器默认值，RLinf 服务另有自己的通信协议、超时和末端变化限制。后续需要调整设备或任务参数时，JSON 按 section 内的字段覆盖默认值：

| JSON 字段 | 需要确认的内容 |
|---|---|
| `robot` / `controller` | 需要覆盖的 SDK 机械臂和控制器参数，例如关节范围、控制增益和负载相关配置。省略的字段沿用 SDK 默认值。 |
| `controller.default_preview_time` | 目标插值预览时间，沿用 SDK `cartesian_controller` 的 0.1 秒默认值，必须大于 `controller_dt`。它与模型动作间隔分别配置。 |
| `adapter.eef_to_tcp` | SDK 末端坐标系到模型工具中心点的 4×4 刚体变换。默认单位矩阵，表示使用 SDK EEF 作为 TCP。 |
| `adapter.gripper_scale` / `adapter.gripper_offset` | 夹爪宽度转换，关系为 `SDK 位置 = 模型宽度 × scale + offset`；默认分别为 1.0 和 0.0。 |
| `adapter.drag_kd` / `adapter.drag_gripper` | 经验证的拖动阻尼；夹爪在拖动时使用 `hold` 或 `manual`。 |
| `limits` | 三个逐命令变化限制：`translation=0.05` 米、`rotation=0.2` 弧度、`gripper=0.02` 米；超限命令被拒绝。 |
| `service.command_timeout` | policy 模式下等待有效命令的最长时间，默认 10 秒；超时切换到保持模式。 |
| `calibration_confirmed` | 可选旧字段；显式为 `false` 时仍阻止 `--allow-motion`，省略时无需补写确认声明。 |

软件默认值保存在 [arx5_server_defaults.json](/home/lenovo/mason/code/rlinf_evo/toolkits/realworld_check/arx5_server_defaults.json)，其中 `controller` 为空，表示沿用 SDK 的 Cartesian controller 参数。服务检查末端平移、旋转和夹爪的单步变化、夹爪行程、命令时限、控制 epoch 和实时反馈，并保留命令 watchdog；IK 与关节处理交给 SDK。默认限幅属于软件运行参数，单位矩阵也只表示 SDK 原坐标约定；它们不能证明已与 UMI 训练数据的工具坐标完成对齐。

> **超时设置：** 这个服务入口使用 10 秒等待窗口，本机此前合成观测测试的首次模型推理约为 5.384 秒。进入 policy 模式后，首次推理等待也计入超时。完整运行还需考虑同步保存等等待，按实际最大正常间隔调整；超时后旧动作失效，需要查明原因再显式恢复。

### 采集时序与任务参数

本机在线动作频率设为 20 Hz，对应 `ARX_ACTION_DT=0.05` 秒。终端二还需要填写其余 5 项：历史状态间隔按训练数据的采样方式确定，相机延迟使用实测值，夹爪范围与 SDK 转换一致，任务指令与 Stage 1 训练任务匹配。

| 环境变量 | 含义与约束 |
|---|---|
| `ARX_ACTION_DT` | 本机设为 `0.05` 秒，对应 20 Hz 的目标动作频率。 |
| `ARX_HISTORY_DT` | 两帧历史状态之间的间隔，单位为秒，有限且大于 0。 |
| `ARX_CAMERA_LATENCY` | 实测相机延迟，单位为秒，有限且大于等于 0。 |
| `ARX_GRIPPER_MIN` | 模型坐标约定下的夹爪最小宽度，单位为米，大于等于 0。 |
| `ARX_GRIPPER_MAX` | 夹爪最大宽度，单位为米，严格大于最小宽度。 |
| `ARX_TASK_DESCRIPTION` | 与 Stage 1 训练任务匹配的指令文本。 |

环境默认缓存 10 秒历史，需满足 `ARX_HISTORY_DT + ARX_CAMERA_LATENCY < 10`。数值变量只填写数字，不要在值中附加 `s` 或 `m`。

`ARX_ACTION_DT` 同时用于训练和评估环境。每个 16 步 chunk 的名义动作时长为 0.8 秒；模型推理、通信和观测等待可能延长实际耗时。相机采集帧率由 `camera_fps` 单独配置，历史状态间隔由 `ARX_HISTORY_DT` 单独配置。

### 每个 episode 结束后的回程目标

成功、失败或达到 episode 步数上限后，机械臂会缓慢回到固定起始位姿，再保持位置。首次启动仍由操作者摆好机械臂和物体并按 `s`；自动回程只负责机械臂位姿和夹爪宽度，物体需要人工恢复。

本机回程目标沿用 [eval_arx5_pi_hzl.py](/home/lenovo/mason/code/umi-arx-deploy/scripts/eval_arx5_pi_hzl.py) 中的 `DEFAULT_INIT_POSE=[-0.006958,-0.002573,0.120019,2.464006,-2.497232,1.524022]`，使用已确认的 `direct-zmq-origin=[0.094905,-0.002323,0.158023]` 和 `eef_offset=0.145` 米。原脚本还通过 `tcp2ee` 转换末端朝向；下面的 `ARX_RESET_POSE` 已包含这一步，格式为服务 TCP 的 `[x,y,z,rx,ry,rz]`，前三项单位为米，后三项为弧度制旋转向量。

这个数值对应当前 `adapter.eef_to_tcp` 为单位矩阵的服务配置，不能直接用原脚本的 `DEFAULT_INIT_POSE` 替换。修改工具坐标标定、机械臂安装或原点偏移后，需要重新换算目标。`ARX_RESET_GRIPPER=0.085` 沿用原脚本的初始夹爪宽度，必须处于本次设置的 `ARX_GRIPPER_MIN` 与 `ARX_GRIPPER_MAX` 之间。

环境配置默认启用 `return_on_episode_end=true`，根据 `reset_frequency=20` Hz 和 `reset_duration=4.0` 秒生成平滑回程目标。逐次发送会计入通信等待，因此回程至少持续 4 秒。末段最多再等待 `reset_settle_timeout=2.0` 秒；实测位置误差不超过 5 毫米、姿态误差不超过 0.05 弧度、夹爪宽度误差不超过 5 毫米后，才切换为保持。超时未到达会停止并报错。回程使用相同的服务限幅和故障检查，按键中断方法见第 5 节。

**终端二命令中的 `<…>` 都是待替换占位符，需要填写实际时序、夹爪范围和任务指令。终端一的本机服务配置文件已经准备好。**

## 2. 终端一：启动 CAN 和 SDK 服务

在机械臂主机的 Bash 终端中执行下面的命令。它将 USB-CAN 串口桥接为 `can0`，然后使用官方 X5 和 Cartesian controller 默认参数启动 SDK 服务。执行前确认原部署程序已经释放同一机械臂的控制连接。原来的 ZMQ 服务使用不同的通信协议，RLinf 采集需要连接这里的服务入口。

```bash
(
set -e

ARX_SERVICE_CONFIG='/home/lenovo/mason/code/rlinf_evo/local_setup/arx5_lenovo.json'
test -f "$ARX_SERVICE_CONFIG"

if ! ip link show can0 >/dev/null 2>&1; then
  sudo slcand -o -f -s8 /dev/arxcan0 can0
fi
sudo ip link set can0 up

env -u PYTHONPATH \
  /media/lenovo/data/miniconda3/envs/arx-py310/bin/python \
  /home/lenovo/mason/code/rlinf_evo/toolkits/realworld_check/arx5_server.py \
  X5 can0 \
  --endpoint tcp://127.0.0.1:5557 \
  --config "$ARX_SERVICE_CONFIG" \
  --allow-motion
)
```

这里的 `slcand -s8` 对应当前 USB-CAN 使用的 1 Mbit/s 总线设置。`env -u PYTHONPATH` 清除其他 Python 环境带来的模块路径。服务启动后保持这个终端运行；正常启动会输出包含 `ARX service listening`、`mode=disabled` 和 `motion_allowed=True` 的信息，等待后续操作启用 policy。

如果只想查看将要使用的配置，在上述服务命令末尾追加 `--print-config`。该选项会输出合并后的参数和 SDK 路径后退出，不导入 SDK 或打开 CAN；CAN 桥接命令本身仍会按代码块执行。

SDK 构造会启动 CAN 收发和控制循环，可能施加重力补偿或阻尼。服务处于 `disabled` 表示尚未允许执行 policy 命令，并不表示机械臂处于断电状态。

## 3. 终端二：设置采集参数并解析配置

在第二个 Bash 终端中激活 RLinf 环境，填写现场参数，并建立本轮日志目录。这个终端后续会复用 `ARX_RUN_ARGS` 和 `ARX_LOG_DIR`，请在同一个终端内依次执行。

```bash
cd /home/lenovo/mason/code/rlinf_evo
source /home/lenovo/mason/code/rlinf_evo/local_setup/activate.sh

export RLT_STAGE1_CHECKPOINT='/home/lenovo/mason/code/rlinf_evo/models/pi05_umi_stage1_20000'
export RLT_NORM_STATS="$RLT_STAGE1_CHECKPOINT/norm_stats.json"

export ARX_ENDPOINT='tcp://127.0.0.1:5557'
export ARX_CAMERA_DEVICE='/dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB51023HTXK-video-index0'
unset RLINF_KEYBOARD_DEVICE

# 在线动作频率为 20 Hz，动作间隔单位为秒。
export ARX_ACTION_DT=0.05

# 其余参数替换为已确认的实际值；时间单位为秒，宽度单位为米。
export ARX_HISTORY_DT='<历史状态间隔>'
export ARX_CAMERA_LATENCY='<实测相机延迟>'
export ARX_GRIPPER_MIN='<夹爪最小宽度>'
export ARX_GRIPPER_MAX='<夹爪最大宽度>'
export ARX_TASK_DESCRIPTION='<与训练任务匹配的指令>'

# 已换算到当前默认服务 TCP 坐标系的固定回程目标。
export ARX_RESET_POSE='[0.21379852650797487,-0.012889266729693302,0.20647035084615928,-0.03253125667781396,0.5174541738443816,-0.04908682931452774]'
export ARX_RESET_GRIPPER=0.085

ARX_LOG_DIR="/home/lenovo/mason/code/rlinf_evo/logs/arx5_round1_$(date +%Y%m%d_%H%M%S)"

ARX_RUN_ARGS=(
  --config-name realworld_arx5_rlt_stage2_ac_mlp
  env.train.override_cfg.camera_pixel_format=YUYV
  env.eval.override_cfg.camera_pixel_format=YUYV
  env.train.override_cfg.enable_camera_preview=true
  env.eval.override_cfg.enable_camera_preview=false
  env.train.override_cfg.start_paused=true
  env.train.override_cfg.start_with_actor=false
  algorithm.update_epoch=0
  algorithm.train_actor=false
  algorithm.replay_buffer.auto_save=true
  runner.max_epochs=1
  runner.save_interval=1
  runner.resume_dir=null
  "runner.logger.log_path=$ARX_LOG_DIR"
)
```

`ARX_RUN_ARGS` 是 Bash 数组，保存完整的启动参数，供配置解析和正式启动复用。相机使用本机 HD60 X 的稳定设备路径；该设备支持 `YUYV`，因此同时覆盖训练与评估配置中的像素格式。日志目录包含时间戳，每次重新执行准备命令都会选择一个新目录。

正式启动前，先执行下面的配置解析命令。它检查模型文件可读性及占位符是否已替换，然后将解析结果写入本轮日志目录。

```bash
(
set -e

test -r "$RLT_STAGE1_CHECKPOINT/full_weights.pt"
test -r "$RLT_NORM_STATS"

for arx_required_var in \
  ARX_ACTION_DT ARX_HISTORY_DT ARX_CAMERA_LATENCY \
  ARX_GRIPPER_MIN ARX_GRIPPER_MAX ARX_TASK_DESCRIPTION \
  ARX_RESET_POSE ARX_RESET_GRIPPER; do
  if [[ -z "${!arx_required_var}" || "${!arx_required_var}" == *'<'* ]]; then
    printf '请先填写 %s 的实际值。\n' "$arx_required_var" >&2
    exit 1
  fi
done

mkdir -p "$ARX_LOG_DIR"
python /home/lenovo/mason/code/rlinf_evo/examples/embodiment/train_embodied_agent.py \
  "${ARX_RUN_ARGS[@]}" \
  --cfg job --resolve > "$ARX_LOG_DIR/resolved.yaml"
)
```

打开 `$ARX_LOG_DIR/resolved.yaml` 核对路径和参数。`--cfg job --resolve` 只解析 Hydra 配置，完整的环境数值校验与设备初始化发生在正式启动阶段。解析成功也不能证明坐标标定或运动参数正确。

## 4. 终端二：启动首轮采集

配置解析成功且现场参数确认后，在同一个终端中运行下面的命令。它通过 `by-id` 和 `by-path` 找到已连接的键盘事件设备，去重后给当前用户授予读取权限，然后启动采集；单机 Ray 由 RLinf 自动启动或连接。

```bash
(
set -e

test -s "$ARX_LOG_DIR/resolved.yaml"
mapfile -t arx_keyboard_devices < <(
  for arx_keyboard_link in \
    /dev/input/by-id/*-event-kbd /dev/input/by-path/*-event-kbd; do
    if [[ -e "$arx_keyboard_link" ]]; then
      readlink -f "$arx_keyboard_link"
    fi
  done | sort -u
)
if (( ${#arx_keyboard_devices[@]} == 0 )); then
  printf '未找到键盘事件设备，请检查实体键盘连接。\n' >&2
  exit 1
fi
sudo setfacl -m "u:$(id -un):r" "${arx_keyboard_devices[@]}"

python /home/lenovo/mason/code/rlinf_evo/examples/embodiment/train_embodied_agent.py \
  "${ARX_RUN_ARGS[@]}"
)
```

模型与环境初始化完成后，程序等待实体键盘的 `s` 键。本轮关键参数含义如下。

| 参数 | 本轮行为 |
|---|---|
| `start_paused=true` | 等待操作者开始。 |
| `start_with_actor=false` | 初始选择 Stage 1 参考动作。 |
| `env.train.override_cfg.enable_camera_preview=true` | 在本机浏览器显示实时画面、EP 编号和状态。 |
| `return_on_episode_end=true` | 完成 episode 后，用至少 4 秒平滑回到 `ARX_RESET_POSE`，随后保持。 |
| `algorithm.update_epoch=0` | 保留采集与保存流程，actor/Q 优化器更新次数为 0。 |
| `algorithm.train_actor=false` | 禁用 actor 和熵温度 alpha 的更新；增加 `update_epoch` 后仍只训练 Q。 |
| `algorithm.replay_buffer.auto_save=true` | 自动保存接收到的有效 trajectory。 |
| `runner.max_epochs=1` | 运行一个采集与保存迭代。 |
| `runner.save_interval=1` | 完成该迭代后保存 checkpoint。 |
| `runner.resume_dir=null` | 本次从新建的 Stage 2 actor/Q 和空 replay 开始。 |

当前配置每个 rollout 计划执行 64 个动作步，即 4 个 16 步 chunk；episode 上限为 320 步。`max_epochs=1` 表示一个迭代，并不保证采集一个完整 episode，也不保证得到 64 个有效 replay transition。只有成功、失败或 episode 步数截断才触发自动回程；单独完成 64 步 rollout 不会触发。中断、故障或过期数据会被过滤。

后续通过下面两个参数选择训练阶段。`algorithm.train_actor` 在 ARX AC 和 TD3 配置中默认为 `true`；本轮显式关闭。各阶段需要重新设置启动参数，不会自动切换。

| `algorithm.update_epoch` | `algorithm.train_actor` | 行为 |
|---|---|---|
| `0` | `false` | 采集和保存，actor 与 Q 均不更新。 |
| 正数 | `false` | replay 达到训练门槛后只更新 Q。 |
| 正数 | `true` | replay 达到训练门槛后，按原有更新比例训练 Q 和 actor。 |

### 查看实时画面与 EP 编号

环境启动后，在 Lenovo 本机浏览器打开 [http://127.0.0.1:8765/](http://127.0.0.1:8765/)。日志也会打印预览地址，程序不会自动弹出窗口。页面显示按原比例缩小的相机画面、EP 编号和当前状态；等待 `s`、模型推理和回程期间仍持续刷新。画面过期或连接中断时，页面会显示提示。

EP 从 1 开始，每次成功开始一个新 episode 才递增；等待开始时显示即将开始的编号。重复按 `s`、使用 `g`、`r`、`b`、`v` 或完成一个 64 步 rollout 都不会增加编号。进程重启或从 checkpoint 恢复后，编号重新从 1 开始，不能据此读取 runner 的 `global_step` 或 replay trajectory 数量。

关闭浏览器页面不会停止采集。需要关闭预览时，将 `env.train.override_cfg.enable_camera_preview` 设为 `false`；通过 `env.train.override_cfg.camera_preview_port` 修改默认端口 `8765`。评估预览在上面的参数中单独关闭，避免训练与评估使用同一端口。端口冲突会产生警告，预览不可用时采集仍继续。

预览复用已采集的图像，缩放和 EP 显示不会改变模型输入或 replay 中的图像。视频录制由 `video_cfg.save_video` 单独控制，当前仍为 `false`。

## 5. 用实体键盘控制采集

程序默认通过 Linux evdev 同时监听所有可读取的键盘，连接在 Lenovo 主机上的任意实体键盘都可以执行下表操作。在启动采集及 Ray 前执行 `unset RLINF_KEYBOARD_DEVICE`，即可清除旧终端中对单个键盘的限制；SSH 终端里输入相同字符不会触发这些操作。Ray 会保留启动时的环境变量，如果现有 Ray 已继承旧的设备选择，需要重启 Ray 才能让修改生效。

监听器会定期重新扫描设备，新接入且有读取权限的键盘会自动加入；拔掉一个键盘后，其余键盘仍可使用。启动时至少需要一个可读取的键盘。上面的 ACL 只作用于当时存在的设备，重新插拔或接入新键盘后，如无读取权限，需要重新执行授权部分。需要只监听一个设备时，仍可设置 `RLINF_KEYBOARD_DEVICE` 为它的事件设备路径。

| 按键 | 操作 |
|---|---|
| `s` | 从当前实测位姿保持并启用 policy，开始当前 episode。 |
| `g` | 进入同臂拖动接管，作废待执行的模型动作。 |
| `r` | 在当前实测位姿保持后重新启用当前 policy；等待开始时仅保持位置。 |
| `c` | 标记成功，奖励 +1；先保持到当前 chunk 结束，再平滑回程。 |
| `a` | 标记失败，奖励 −1；先保持到当前 chunk 结束，再平滑回程。 |
| `q` | 标记中止；等待开始时取消等待。它不是通用的正常退出并保存按键。 |
| `v` | 后续预测切回 Stage 1 参考动作。 |
| `b` | 后续预测选择 Stage 2 actor；本轮 actor 未训练，应保持参考分支。 |

先建立与训练任务一致的起始状态，再按 `s`。成功或失败标注会在当前 chunk 剩余时间内采样保持位姿，并在 chunk 边界记录终态和奖励，然后执行回程。达到 320 步 episode 上限时，也会在 chunk 边界截断并回程。终态观测取自回程前，回程动作不进入 replay；JSONL 会单独记录回程事件。若一直未标注结果，64 个计划动作步可能只留下零奖励和未终止的片段。

回程期间，按 `g` 可取消回程并进入拖动，按 `q` 或 `r` 可取消回程并保持当前实测位姿；`s`、`b`、`v`、`c` 和 `a` 被忽略。取消后不会自动重试回程。`g` 接管状态会保留到下一次等待开始阶段，直到操作者按 `s` 或 `r` 交还控制。正常采集时按 `q` 中止、出现故障或退出程序，都不会主动回到起点。

若服务已经因超时进入 `hold`，或处于 `fault`，结果标注可能被拒绝。先检查服务状态和原因，再恢复操作；`c`、`a` 和 `q` 不能替代服务保持命令。

**当前配置启用了 `auto_reset=true`。** 完成 episode 和回程后，环境会进入下一次 reset 并等待再次按 `s`；恢复物体的起始状态后再开始。即使在最后一个 chunk 标记成功或失败，也可能先等待这次 `s`，才返回数据并完成本轮保存。因此，出现等待时应先判断是否处于下一轮开始状态，不要把尚未保存的等待误认为进程卡死。回程发生在终止 step 返回前，即使关闭 `auto_reset`，已完成的 episode 仍会执行回程。

## 6. 找到采集数据和 checkpoint

正常完成迭代后，在终端二设置的 `$ARX_LOG_DIR` 中查看本轮输出。这里的 `global_step_1` 表示 runner 完成了一个迭代，不能据此认定 Q 网络已经训练。

```text
arx5_round1_<时间戳>/
├── resolved.yaml
├── arx5_raw/
├── replay_buffer/
│   └── rank_0/
└── realworld_arx5_rlt_stage2_ac_mlp/
    └── checkpoints/
        └── global_step_1/
            └── actor/
                ├── model_state_dict/
                └── sac_components/
                    └── replay_buffer/
                        └── rank_0/
```

| 位置 | 内容与用途 |
|---|---|
| `replay_buffer/rank_0/` | 本轮自动保存的 trajectory 主目录，包含 `trajectory_*.pt`、`metadata.json` 和 `trajectory_index.json`。 |
| `arx5_raw/` | 操作事件、实测 step 和独立回程事件的 JSONL 记录；不能作为完整图像数据集使用。 |
| `global_step_1/actor/model_state_dict/` | 本轮初始化后的 Stage 2 actor/Q 权重，尚未经过优化器更新。 |
| `global_step_1/actor/sac_components/` | Stage 2 相关训练状态和 replay 保存信息。 |

本轮启用 `auto_save=true`。保存 replay checkpoint 时，会先等待已提交的 trajectory 写入完成，再将索引中的全部已持久化 trajectory 连同相互匹配的元数据和索引保存到 `global_step_1/actor/sac_components/replay_buffer/rank_0/`。即使 `algorithm.update_epoch=0`、首轮尚未进行采样，成功保存的 checkpoint 也包含这些 trajectory 数据。实际有效 trajectory 数量仍取决于过滤结果。

后台写入失败或所需 trajectory 文件缺失时，checkpoint 保存会报错。让采集命令正常完成保存，再核对 checkpoint 中的 trajectory 文件与索引是否对应；仅看到目录不能证明保存已完成。这个过程不会补齐旧 checkpoint 中已缺失的文件。

**保留整个 `$ARX_LOG_DIR`，以便同时归档原始 JSONL 和 `resolved.yaml`。** checkpoint 只涵盖保存时已经进入 replay 的数据；环境尚未发送到 replay 的末尾 transition 不在其中。强制中断仍可能留下未完成的写入，不能依靠退出进程触发完整保存。

### 从采集 checkpoint 开始训练 Q

保留前面设置的现场参数和 `ARX_RUN_ARGS`，将下面的 `runner.resume_dir` 替换为实际采集 checkpoint 的完整路径。示例从 `global_step_1` 恢复，再执行一个迭代，并将结果写入新的日志目录。只有 replay 达到 `algorithm.replay_buffer.min_buffer_size=64` 的门槛后才会更新 Q；这里的 64 指 trajectory 条目数。只采集了首轮时，下一轮仍可能不执行任何 Q 更新。此命令用于演示恢复参数，实际 Q 训练需要设置相应的累计迭代上限：

```bash
ARX_Q_LOG_DIR="/home/lenovo/mason/code/rlinf_evo/logs/arx5_qonly_$(date +%Y%m%d_%H%M%S)"
python /home/lenovo/mason/code/rlinf_evo/examples/embodiment/train_embodied_agent.py \
  "${ARX_RUN_ARGS[@]}" \
  algorithm.update_epoch=8 \
  algorithm.train_actor=false \
  runner.resume_dir=/path/to/checkpoints/global_step_1 \
  runner.max_epochs=2 \
  "runner.logger.log_path=$ARX_Q_LOG_DIR"
```

后面的参数覆盖数组中的采集设置。`runner.max_epochs` 是累计迭代上限，必须大于加载的 `global_step`；从其他步数恢复时，应同步增大该值。这个入口仍会初始化真机环境，每轮先执行 rollout，再在 replay 满足门槛后进行学习更新。下一阶段要训练 actor 时，需要显式设置 `algorithm.train_actor=true`。

仅训练 Q 时，actor 和熵温度 alpha 的优化器及学习率调度停止，Q 优化器、Q 学习率调度和 target model 的 EMA 仍按原配置执行。当前 RLT MLP 的 actor 与 `q_head` 参数独立，因此在线 actor 保持初始化或恢复时的权重，Stage 1 继续冻结。本页 AC 配置的 TD 目标动作仍由这个固定的在线 actor 生成，并保留 `fixed_std=0.002` 的采样噪声。`start_with_actor=false` 选择机械臂执行的参考动作分支，不会将 TD 目标动作改为精确的参考动作。

发生学习更新时，`train/actor/updated` 是各次更新是否实际执行 actor 优化的 0/1 均值，仅训练 Q 时应为 0。启用 RLT 调度后，`train/rlt/actor_updates_run` 记录本轮实际 actor 更新次数。runner 的 `global_step` 表示采集与训练迭代进度，不是优化器更新次数。

## 7. 结束采集并保持机械臂

采集进程结束后，终端一的 SDK 服务会继续运行。可以在另一个终端发送保持当前位置的请求；该请求不代替采集程序完成 replay 和 checkpoint 保存。

```bash
env -u PYTHONPATH \
  /media/lenovo/data/miniconda3/envs/arx-py310/bin/python \
  /home/lenovo/mason/code/rlinf_evo/toolkits/realworld_check/arx5_server.py \
  --endpoint tcp://127.0.0.1:5557 \
  --operation hold
```

命令返回当前服务状态；正常请求后 `mode` 应为 `hold`。如果服务已经进入 `fault`，保持请求也可能被拒绝，需要先排查故障。确认机械臂和负载已获得支撑后，再在终端一按 `Ctrl+C` 结束服务。正常退出可能切换到被动或阻尼状态，强制终止则跳过 SDK 清理。

工程内 SDK 的 emergency 处理会先释放状态与命令锁，初始化校验失败时抛出 `RuntimeError`，后台阻尼循环可在析构时退出。进入 fault 后，Joint 和 Cartesian controller 都拒绝后续运动命令；处理原始故障后需要重建 controller 或重启服务。

当前入口在同一个服务进程内直接运行 SDK。服务运行期间按 `Ctrl+C` 或发送 `SIGTERM` 会请求 `service.stop()`，正常退出由 SDK 完成清理。使用工程内重新编译的 `third_party/arx5-sdk` 才能获得 emergency 修复；引发 emergency 的夹爪标定错误仍需处理后再启动。

## 8. 常见启动问题

遇到以下现象时，先检查对应环节，再恢复运行。软件配置解析通过后，仍可能在设备连接或动作执行阶段发现问题。

| 现象 | 检查内容 |
|---|---|
| 仍提示必须提供 `calibration_confirmed: true` | 检查执行的是否为当前工程中的新版 `arx5_server.py`。当前入口已提供默认限幅，JSON 为可选覆盖。 |
| 提示与 `calibration_confirmed: false` 冲突 | 所提供的覆盖文件显式禁止运动，应核对该设备配置的真实状态。 |
| 提示 `No compatible compiled arx5_interface binding` | 使用构建 SDK 时的 Python 版本；本机已编译的绑定对应 `arx-py310`。 |
| 提示 `default_preview_time > controller_dt` | JSON 中的插值预览时间是否使用了验证值。 |
| 按 `s` 后很快进入保持 | 检查 `command_timeout` 与首次推理等正常等待时间。 |
| `can0` 不存在或 SDK 无法连接 | 检查 `/dev/arxcan0`、USB-CAN、总线参数及其他控制进程。 |
| 相机格式错误 | 训练与评估是否都覆盖为 `YUYV`，设备路径是否仍指向 HD60 X。 |
| 实体键盘无响应 | 确认采集及 Ray 启动前已执行 `unset RLINF_KEYBOARD_DEVICE`，并检查键盘事件设备的读取权限；新接入或重新插拔后可能需要重新设置 ACL。 |
| 成功或失败后仍在等待 | 确认平滑回程已完成或被取消；自动 reset 会等待恢复物体等起始状态并再次按 `s`。 |
| 回程目标未配置或格式错误 | `ARX_RESET_POSE` 必须为服务 TCP 的六个有限数值：XYZ 米制位置与弧度制旋转向量。 |
| 回程未在等待时间内到达 | 检查目标坐标、实测误差、跟踪能力和服务状态；程序停止并报错，不自动重试。 |
| `step_limit` | 目标超过末端平移、旋转或夹爪变化限制；核对坐标转换、动作解码和现场限幅。 |
| SDK 初始化报错或 fault 后拒绝命令 | 确认加载的是工程内已重新编译的 SDK，处理夹爪标定等原始错误后重启服务；fault 后需要重建 controller，不能通过继续发送命令恢复。 |

## 相关代码与说明

需要调整流程时，从以下入口核对实际行为；本机设备路径或训练配置变化后，应同步更新这份启动说明。

- [SDK 服务入口](/home/lenovo/mason/code/rlinf_evo/toolkits/realworld_check/arx5_server.py)
- [仓库内 SDK 与构建说明](/home/lenovo/mason/code/rlinf_evo/third_party/README.md)
- [当前机械臂配置](/home/lenovo/mason/code/rlinf_evo/local_setup/arx5_lenovo.json)
- [Stage 2 AC 配置](/home/lenovo/mason/code/rlinf_evo/examples/embodiment/config/realworld_arx5_rlt_stage2_ac_mlp.yaml)
- [ARX5 环境配置](/home/lenovo/mason/code/rlinf_evo/examples/embodiment/config/env/realworld_arx5.yaml)
- [ARX5 通用说明](/home/lenovo/mason/code/rlinf_evo/docs/source-zh/rst_source/examples/embodied/arx5_rlt.rst)
- [本机模型验证结果](/home/lenovo/mason/code/rlinf_evo/logs/model_validation/pi05_umi_stage1_20000/results.json)
