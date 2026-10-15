# ARX RLT Stage 2 测试记录

本文记录本次新增功能的测试证据与尚未完成的硬件验证。纯函数和假设备验证不等同于机械臂运动验收。

最新提交前定向验证为 **186 项通过**，见下节。后文保留 2026-09-23 的 AC／TD3 共用适配回归（622 通过、7 跳过、27 排除）和初版 ARX 记录；不同轮次的结果不累加。

## 真机采集流程提交前验证（2026-10-15）

| 范围 | 通过 |
| --- | ---: |
| ARX 控制服务、协议与 CLI | 34 |
| ARX 环境、回位、预览及键盘 | 44 |
| ARX 训练数据、replay 和 AC／TD3 更新 | 57 |
| AC／TD3 配置与必填参数 | 51 |

本轮补充自动回位、中断回位后控制权保留、回位轨迹不写入 episode 动作、HTTP 只读预览、多键盘热插拔、仅训练 critic、replay 保存恢复及后台写盘错误传播测试。CLI 测试通过 Python 3.10 运行 `--print-config`，使用不存在的 SDK 路径证明配置预览不会加载 SDK 或连接设备；同时验证两种参数写法、默认配置合并及显式禁止运动的配置。

配置回归发现 `oc.decode` 返回 Python list 后，当前 OmegaConf 的原地 resolve 会报 `UnsupportedValueType`，阻止设置 `ARX_RESET_POSE` 的正常启动。已改为保留环境变量字符串，在环境配置构造时解析 JSON，并验证六维、有限数值和缺失参数。直接提供列表仍受支持。

上述测试均使用 CPU 和假设备。IPC／TCP 与 HTTP 测试最初受沙箱 socket 限制；经自动审批允许本机临时端口后通过。没有连接真实 CAN、相机，也没有进行真实运动或完整 GPU 训练。未运行的其余测试不计作通过。

执行时使用 `env -u PYTHONPATH`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、`CUDA_VISIBLE_DEVICES=''`、`JAX_PLATFORMS=cpu`，并将 OMP／OpenBLAS／MKL 线程数设为 1。选取范围为 `test_robotics.py` 中的 ARX 用例、`test_real_env.py` 中的 ARX／键盘用例、`test_data.py` 中的 ARX／replay 用例，以及 `test_utils.py` 中的 `arx_stage2` 用例。筛选结果按互不重叠的用例计数。

RLinf 自有 Python 文件 Ruff lint／format、自有文件的 `git diff --check` 通过；四份 ARX／Franka 双语文档的 markup 和符号检查通过，双语完整 Sphinx 构建均零警告。SDK 保留来源、MIT 许可证和上游格式；整体 staged diff 检查会报告其中已有的行尾空白及 CRLF，未为消除这些提示重写第三方源码。上游运行库纳入版本，构建目录及生成的 Python 扩展保持忽略。约 16.4 GB 的本机模型目录不纳入代码提交。

## AC／TD3 共用适配回归（2026-09-23）

| 测试范围 | 通过 | 跳过 | 排除 |
| --- | ---: | ---: | ---: |
| Models + Utils + Data | 371 | 6 | 9 |
| Real environment | 251 | 1 | 18 |
| 本轮合计 | 622 | 7 | 27 |

两种算法共用 EEF 残差缩放、行式 rot6d 投影和米制夹爪边界；人工 BC 也使用同一目标选择实现。默认旧 AC 的 tanh 和旧 TD3 的 clamp 行为保持不变。新增 TD3 配置沿用仓库现有 RLT TD3 变体，使用自己的 head 和 checkpoint，共用 ARX／GoPro、Stage 1 特征、schema、replay 与同步执行流程。

本轮覆盖以下实际代码路径：

- 两种 policy 的确定性初始输出等于 reference；归一化 XYZ 超过 1 时保留、旋转行正交、夹爪按物理边界限幅；reference dropout 不改变动作的参考基准。
- 残差噪声按各维尺度缩放、反向传播梯度有限、实际 optimizer 更新；默认 AC 和 TD3 噪声强度仍由各自配置决定，并未强制相同。
- 模型工厂与 state_dict 往返、生产 rollout 初始化加载保存的 H=16 actor 文件，恢复后确定性输出一致。
- 两种真实模型输出经过实际动作／人工覆盖、TrajectoryStep 和真实 replay，再计算生产 actor／critic loss 并执行 CPU Adam 更新；无效 chunk 被过滤，原 reference 保留，人工动作仅作为对应步骤的 BC 目标。
- BC 的无人、全人工、混合和缺省标记；逐步、展开及三维 mask；错误 batch 或元素数明确拒绝。
- chunk 奖励折扣、终止与截断的 bootstrap 区别、standard／always 模式，AC 使用在线 actor 而 TD3 使用 target actor 形成下一动作；target 不产生梯度。
- 直接执行生产 update_one_epoch，检查 critic 持续更新、actor 延迟更新、target 更新频率与 EMA，以及 actor／critic optimizer 参数隔离。CPU 适配器仅提供 FSDP 的梯度裁剪接口、空计时器和单 rank 指标归约；未运行多 GPU FSDP。
- 两种 policy 的实际推理结果经过同一 ARX environment、SDK adapter 和 service 执行，位置／旋转／夹爪反馈与目标一致；仅 CAN、IK 和相机为外部设备模拟。两种模型均覆盖自动复位开／关下 EnvWorker 的最终动作、奖励、done 和路由标记。
- 44 项 Hydra 测试覆盖 AC／TD3 的训练／评估配置、checkpoint 路径，以及全部必填模型和标定输入。TD3 训练与评估 Python 入口另通过 --cfg job --resolve 检查，不启动设备。

模型文件另独立执行得到 **113 passed、6 skipped**，配置专项 **44 passed**，数据专项 **52 passed**；这些都包含在合并计数内，不重复累加。Ruff lint／format 和 git diff --check 通过；EN／ZH 完整 Sphinx 构建均为零警告，首次本机 sitemap socket 受沙箱限制后，经自动审批重跑通过。

合并回归命令：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest \
  tests/unit_tests/test_models.py tests/unit_tests/test_utils.py tests/unit_tests/test_data.py \
  -k 'not load_jax_params and not redirected_entrypoint and not test_vlm_trend_batch_video_metadata_stays_nested_per_sample' \
  -q -o faulthandler_timeout=60

env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/unit_tests/test_real_env.py \
  -m 'not placement' -q -o faulthandler_timeout=60
```

日志：`test_logs/arx_ac_td3_model_data.log`、`test_logs/arx_ac_td3_real_env.log`，以及 `test_logs/realworld_arx5_rlt_stage2_td3_{mlp,eval}-resolved.log`。

9 项排除沿用下文已记录的 6 个 Orbax 保存阻塞、2 个入口子进程阻塞和 1 个 VLM／Transformers 接口失败；另排除 18 项 placement。7 项跳过来自既有可选依赖条件。这些项目不计作通过。本轮没有连接真机、加载完整 GPU checkpoint 或进行真实训练保存／恢复验收。

## 初版 ARX 适配的历史回归

此前选定 CPU 回归为 **883 通过、7 跳过、35 排除**。当时新增 ARX 功能相关测试全部通过；扩大回归发现的 1 个现有 VLM 依赖失败及 8 个阻塞项保留在下文，不计入通过数。

| 模块测试文件 | 通过 | 跳过 | 排除 |
| --- | ---: | ---: | ---: |
| Robotics（含驱动／相机／安装） | 348 | 0 | 8 |
| Real environment（含操作／worker 集成） | 247 | 1 | 18 |
| Models + Utils + Data | 288 | 6 | 9 |


## 模型数据接口与动作 codec

已在 CPU 上通过 11 项专项测试（`test_utils.py` 和 `test_models.py`，筛选 `eef_ or arx_repack or arx_rlt_feature`）：

- 非零位姿、非对称旋转下，固定 chunk anchor 的绝对/相对动作互逆；明确检验旋转取矩阵前两行，不取列。
- 原 checkpoint min/max 优先、q01/q99 回退、常量维处理、XYZ 独立归一化、旋转与米制夹爪不变，且不把越界归一化值裁剪到 [-1, 1]。
- 两帧状态的相对当前姿态、相对 episode 起点姿态、夹爪顺序和 2×16→32 展平。
- 显式模型字段 repack 和现有 OpenPI 输入 transform 组合；缺失历史字段报错，不合成历史。
- Stage 1 RLT 特征提取交付归一化 proprio32 和归一化 reference10，避免 reference、actor 和 replay 尺度不一致。
- 可微行式 rot6d 投影的双精度梯度检查、批量输入、退化行输入的有限梯度。
- 新动作参数化下 eval 确定性、train 探索、人工 BC 与 Q 的实际反向传播和参数更新，以及默认 tanh 模式兼容性。

另直接读取用户指定的 `/home/lenovo/mason/code/openpi-umi/src/openpi/transforms.py`，仅抽取 `Normalize` 类进行独立原源码对照：min/max、q01/q99、常量维与 XYZ mask 的归一化结果完全一致。直接读取原 `src/openpi/utils/pose_utils.py` 的 `mat_to_pose10d`，对 20 个随机位姿比较 `inverse(T_anchor) @ T_pose`，动作几何最大绝对误差为 **8.881784197001252e-16**。

原源码只用于此次核对；新实现及单元测试不依赖外部 `openpi-umi` 或 `umi-arx-deploy` 工程。核对命令使用 `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=''`，没有加载完整 checkpoint 或占用 GPU。

新 actor 参数化为显式配置 `action_parameterization: eef_rot6d`：对 reference 预测增量，零初始化输出 head 使初始确定性 actor 跟随 reference；XYZ 保持 checkpoint 归一化域，旋转投影到 SO(3)，夹爪保持米制并使用校准边界。该行为由新配置启用，其他模型默认 tanh 行为不变。

随后补充了工厂配置接线与 Stage 2 `state_dict` 保存/恢复后确定性输出一致性测试，模型/codec 专项总计 12 项。

## 通信与人工数据回放

新增 9 项 ARX 数据专项测试已通过：

- `EnvTransition` 的实际动作、有效标记在拆分/合并和字典转换后保持一致。
- 实际执行动作覆盖模型提议，人工动作按步覆盖，人工标记按 10 维展开；终止状态和下一观测不发生错位。
- 新 EEF actor 在 replay 中保留原 Stage 1 reference，人工动作只作为 BC 目标。真实 actor 推理结果经过 schema 转换，再计算人工 BC，验证损失及 actor 梯度非零，防止 reference 直接泄漏纠正目标。
- 五个原模型字段通过 `EnvOutput` 和 `PolicyInput` 的拆分/合并，避免通信层丢弃模型观测。
- 使用真实 `TrajectoryReplayBuffer` 验证过滤无效 chunk 后，动作、curr/next 特征、reward、done 和人工 demo 仍属于同一条 transition；包含每个 rollout epoch 末尾额外 bootstrap value 的情况。
- 过滤规则未启用时保留原行为；启用但缺少有效性标记时报错；全部无效时不写入 replay。

模型与工具扩大回归（排除下述环境阻塞项）结果为 **139 passed, 6 skipped, 8 deselected**。六个 `load_jax_params` 用例在既有 Orbax 保存步骤长时间无进展；两个 `redirected_entrypoint` 用例卡在启动的子进程，其中首项触发其原有 300 秒超时。已中断这些阻塞执行，不能将其计作通过。六个 skipped 项来自既有模型测试的可选组件条件。

`test_data.py` 全量回归完成：**134 passed，1 failed**。唯一失败为未修改的 `test_vlm_trend_batch_video_metadata_stays_nested_per_sample`：当前安装的 `transformers.video_utils.VideoMetadata` 不接受已有 VLM 数据集代码传入的 `frames_indices` 参数。该失败不在 ARX 模型、环境或 replay 路径，未修改 VLM 功能来掩盖问题。

## ARX SDK 服务、驱动与机器人注册

专项回归最终为 **44 passed，312 deselected**。命令：

```bash
ARX_SDK_PYTHON=/media/lenovo/data/miniconda3/envs/arx-py310/bin/python \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. \
.venv/bin/python -m pytest tests/unit_tests/test_robotics.py \
  -k 'arx or lazy_package or registry' --disable-warnings -q
```

测试使用真正的本机 ZMQ IPC 和 TCP 传输、生产 SDK adapter/service/arm 对象；仅 CAN 控制器和运动学求解器替换为外部设备模拟。操作客户端另起 Python 3.10 进程，运行正式 `arx5_server.py --operation ...` 入口。覆盖：

- 构造无设备动作、连接仅查询、夹爪共享连接、重复断开、机器人资源注册与延迟导出。
- 进入拖动清除目标、关节 kp=0、夹爪保持／手动模式；退出按实测位置重建目标，迟到的旧 epoch 命令不能重新取得控制。
- IK 失败、关节越界、平移／旋转／夹爪独立步进限制、计算 IK 时请求过期：均验证未向 SDK 提交目标。
- 无响应超时、错误 JSON 后恢复通信、服务重启身份校验、SDK 命令失败及反馈过期锁存故障；普通状态读取不能解除故障。
- 没有模型请求时仍持续采样，watchdog 保持及 epoch 更新，服务和采样线程有界停止。
- 非对称旋转、半周旋转、TCP 点／轴标定及夹爪标定往返。
- SDK 反馈按固定 tick 更新：重复读取不产生重复历史结点；固定 SDK→主机 monotonic 偏移消除重复 tick 的时间抖动。控制模式切换后等待首个新反馈 tick，避免将切换前测量回填为新 mode/epoch；可直接通过严格历史插值。

沙箱禁止本机 socket bind，初次 transport 测试得到 `Operation not permitted`；随后经执行权限审核在沙箱外运行上述纯假硬件测试通过。该权限用于临时本机 IPC/TCP 端口，不访问 CAN 或视频设备。系统 ROS 的自动 pytest 插件与当前环境不兼容，因此命令禁用了第三方插件自动加载。

额外使用本机 `/media/lenovo/data/miniconda3/envs/arx-py310/bin/python -I -B` 验证生产 protocol 模块可在 Python 3.10 中独立加载，并实际导入已有 `arx5_interface`：验证 `Gain`、`JointState` 数组赋值，以及实际 X5 URDF 的 FK→IK 往返。此项只创建运动学 solver，没有创建 CAN controller。结果通过；SDK 输出既有 KDL 根节点惯量提示。

核查发现现有 SDK CartesianController 会在 IK 失败后仍提交目标，新增服务改用 JointController＋先验检查的独立 Solver，未修改外部 SDK。SDK JointController 默认插值预览时间为 0；运动模式必须显式提供大于 `controller_dt` 的 `default_preview_time`，服务在打开硬件前校验，普通动作按未来目标时间提交。

## 键盘操作生命周期

`test_real_env.py -k 'keyboard or session or presses'` 结果为 **6 passed，248 deselected**。覆盖短按保留、重复事件不重复计数、空闲关闭、关闭打断重连且不再打开设备、不可恢复读取错误可被上层发现，以及 session 释放输入设备。真实 `KeyboardListener` 接收模拟 evdev 边界事件，未打开本机键盘设备。

本节涉及的生产模块与共享设备模拟均通过 Ruff 检查。仍未进行真实机械臂运动、重力补偿载荷、夹爪持物、CAN 故障恢复或实际相机时间对齐验收；模拟通过不代表这些硬件项目已经完成。

## 启动与固定权重评估复核

新增 3 项 CPU 验证通过：

- Stage 1 `model.safetensors` 加载：SFT 仍可从 base checkpoint 初始化新 RLT head；Stage 2 使用的 `eval + use_rlt` 必须含完整 RLT 权重，缺失或只有部分 head 权重均报错。修复了此前 safetensors 分支会冻结随机 RLT head 继续运行的缺口。
- Stage 2 使用训练保存代码相同的 `get_model_state_dict(full_state_dict=True)` 导出小型 actor，保存至 `actor/model_state_dict/full_weights.pt`，通过真实 rollout `init_worker` 加载，H=16 的确定性动作逐元素一致；Stage 1 feature 模型处于 eval 且参数冻结。
- 真实 `build_openpi_transforms` 读取显式 `openpi_data.norm_stats_path`；即使 checkpoint 目录不同，仍与环境 codec 使用同一统计文件。state32、有效 action10、22 维零 padding 和动作逆归一化均正确。

训练与评估模板共用 actor 结构、H=16、原统计文件及 EEF 动作域。完整模型、FSDP/GPU 显存和真实保存/恢复训练仍需要现场资源完成验收。

## Robotics 全量 CPU 回归

最终结果：**348 passed，8 deselected，42.72 秒**。八个 deselected 均为明确排除的 placement 测试。此次覆盖整份 `tests/unit_tests/test_robotics.py`，包括新旧机械臂、共享夹爪、相机基类、UVC、注册导出和 ARX IPC/TCP；未访问实际硬件。

```bash
ulimit -c 0
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
ARX_SDK_PYTHON=/media/lenovo/data/miniconda3/envs/arx-py310/bin/python \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. timeout 300s \
.venv/bin/python -m pytest tests/unit_tests/test_robotics.py \
  -m 'not placement' -o faulthandler_timeout=60 --disable-warnings -q -s
```

首次未限制 CPU 数学库线程数的全量执行，在前序测试后创建 ARX ZMQ socket 时发生原生 `SIGABRT`（退出码 134），没有 Python 断言失败；该中断不计作通过。随后设置上述线程上限与禁用 core dump，同一完整测试选择全部通过，期间没有修改生产逻辑或删减测试。原生中断的确切原因尚未独立复现定位，因此保留首次结果作为环境稳定性记录。完整成功日志在 `local_setup/test_logs/arx_robotics_full_regression.log`。

## 环境、同步 chunk 与训练 worker 集成

最终 `test_real_env.py -m 'not placement'`：**247 passed，1 skipped，18 deselected**，11.46 秒；其中新增 ARX 环境集成筛选 **29 项通过**。skipped 为既有可选环境条件，18 项 placement 不纳入此次本机 CPU 回归。

测试运行真实 ARX environment、SDK adapter、控制 service、向量环境、`RealWorldEnv` 和 `EnvWorker` 数据打包；仅 CAN、IK 和相机设备读数由共享设备模拟提供。覆盖：

- 固定原始观测 anchor 连续解码多个动作，防止每步重新基准化造成位置累加；首帧按原流程重复历史，观测 32D／动作 10D／图像 RGB 顺序正确。
- 插值无外推、最短四元数路径、反馈间隙、服务重启跨两帧历史、过期图像和不合法裁切。
- 推理等待时进入拖动立即更新控制 epoch，迟到 chunk 和中途恢复后旧 chunk 均不能运动；下一份新观测才允许恢复。
- 真实拖动位姿编码、固定时间步采样与下一 chunk 最新图像，避免推理耗时不断积累为观测滞后。
- 成功／失败立即保持，剩余 chunk 真实等待并确认保持反馈，只在最后一步发布一次奖励与 done；拖动后的保持保留人工来源。`policy → drag → success` 不会错误通过原 policy 动作的有效性检查。
- abort 使 transition 无效；暂停等待开始可以有界取消；关闭与操作线程并发时最终保持；拖动到达 episode 上限后保持。
- 相机断开在推理等待期间也会保持机械臂；无效旋转不提交目标；夹爪显式限幅后写入实际执行值。
- 两个方向的 Stage 1／Stage 2 切换、跨训练 epoch 的路由保留，以及固定权重评估第一次请求直接使用 actor。
- 同时测试 auto-reset 开／关：最终动作、有效标记、奖励、终止观测经过 `RealWorldEnv → EnvWorker → EnvTransition` 保持对应；兼容本机 Gymnasium 0.29 的 object-array infos。
- 关闭最外层 `RealWorldEnv` 会释放内部环境，并完成 ARX 保持。

此外，直接抽取原 `/home/lenovo/mason/code/umi-arx-deploy/utils/cv2_util.py` 的 `get_image_transform`，对 1920×1080、407×301、45×73、223×345 的随机彩色图像逐像素对照：**四种分辨率均完全一致，最大像素误差 0**。实际流程为保持比例缩放（缩小时 AREA、放大时 LINEAR）、中心裁切、BGR→RGB；不依赖原部署工程运行。

复现命令：

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest tests/unit_tests/test_real_env.py \
  -m 'not placement' -q -o faulthandler_timeout=60
```

完整成功日志：`local_setup/test_logs/arx_real_env_final.log`。

## 相机、安装、配置与文档

UVC／相机基类／安装路由专项 **38 项通过**，另补充 native open/read/release 挂起与清理测试。这些测试包含在上述 robotics 全量 348 项中，不能重复加总。使用真实 spawn 子进程与外部 Capture 模拟，核验打开失败、读失败、帧副本、过期帧、重连后的时间／序号、格式／FPS 校验和有界终止。

Hydra 配置 **12 项通过**，已包含在下方合并回归中。训练和评估的实际 Python 入口 `--cfg job --resolve` 均退出 0，未连接设备或加载完整模型。EN/ZH Sphinx 完整构建均零警告（仓库 docs-only harness 模式），RST markup 与代码符号检查通过。所有变更 Python 文件 Ruff 检查／格式、安装脚本 `bash -n`、`git diff --check` 通过。

本机依赖核验：Python 3.11 RLinf 环境中已有 torch 2.11.0+cu128、Gymnasium 0.29.1、OpenCV 4.11.0.86、pyzmq 27.2.0、evdev 2.0.0、SciPy 1.17.1、rlinf-openpi 0.1.1；SDK 继续使用独立 Python 3.10 环境。新增 ARX 安装依赖包含 evdev。

## 最终合并回归与未通过项

模型、工具和数据三份测试合并执行，最终 **288 passed，6 skipped，9 deselected**，26.91 秒。其中排除了已在上文记录的 6 个 Orbax 阻塞、2 个入口子进程阻塞，以及 1 个 VLM／Transformers 参数兼容失败；这些均不计作通过。第一次合并发现既有测试 shim 在已安装 Gymnasium 时仍向 `sys.modules` 注入假模块，导致新增配置测试失败；已修正为仅在库确实未安装时使用 shim，合并重跑通过。

```bash
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m pytest \
  tests/unit_tests/test_models.py tests/unit_tests/test_utils.py tests/unit_tests/test_data.py \
  -k 'not load_jax_params and not redirected_entrypoint and not test_vlm_trend_batch_video_metadata_stays_nested_per_sample' \
  -q -o faulthandler_timeout=60
```

完整成功日志：`local_setup/test_logs/arx_model_data_final.log`。按互不重叠的五份文件计数，最终选定 CPU 回归共 **883 passed、7 skipped、35 deselected**；该数字不是整个仓库全部测试通过，也不包含实际硬件验收。

## 实机验收仍待完成

当前未创建真实 CAN controller、未打开实际 GoPro、未加载完整 GPU checkpoint、未启动机械臂运动。仍需提供／确认完整 Stage 1 RLT checkpoint 与原始 norm stats、ARX 型号和 CAN、实际相机设备与规格、EEF/TCP／夹爪标定、动作／历史步长及实测相机延迟。

实机验收应依次覆盖：只查询状态与图像；核对坐标方向和米制夹爪；验证小幅目标及 SDK 插值；带实际负载进入拖动、保持夹爪并从测量位姿恢复；在推理等待和 chunk 中途接管／结束；核对 JSONL、人工动作与 replay；最后执行短程训练、保存、恢复和固定 checkpoint 评估。每项使用现场已确认限值记录结果，未完成前不能以本报告的假硬件测试替代验收。
