在 ARX 上运行 RL Token Stage 2
==============================

本页说明如何在单台 ARX 机械臂上接入 GoPro 图像、通过拖动同一机械臂进行人工介入，并训练 RL Token Stage 2。SDK 服务负责机械臂控制，RLinf 保留原 checkpoint 的 EEF 表示；训练和固定权重评估共用执行流程。先核对模型、时序和标定，再进行真机验证与训练。

概览
----

配置保留原来的 16 步预测与执行长度。预测首步对应观测时的当前位姿，只执行首步会丢失后续运动。模型推理与动作执行采用同步顺序；相机采集、状态采样和操作事件处理独立运行。等待下一次推理时，策略动作仍可能出现停顿。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型

      冻结的 π₀.₅ 与 RLT 特征，Stage 2 MLP actor 和 critic。

   .. grid-item-card:: 算法

      RLT AC 或已有 RLT TD3 变体，结合参考动作与人工动作的 BC 目标。

   .. grid-item-card:: 任务

      人工标注结果的操作任务，以及同臂拖动纠正。

   .. grid-item-card:: 硬件

      一台 ARX、腕部 UVC 图像采集和本机 NVIDIA GPU。

任务
~~~~

选择与 Stage 1 checkpoint 一致的任务，并填写对应指令。

.. list-table::
   :header-rows: 1

   * - 环境
     - 任务
     - 配置
   * - ``Arx5RLTEnv-v1``
     - 手动复位，人工标注成功或失败
     - ``realworld_arx5_rlt_stage2_ac_mlp``
   * - ``Arx5RLTEnv-v1``
     - 同样的任务和操作，使用 TD3 Stage 2 head
     - ``realworld_arx5_rlt_stage2_td3_mlp``

观测与动作
~~~~~~~~~~

训练、推理和人工数据都使用同一套模型坐标与数值约定。

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - 字段
     - 约定
   * - 观测
     - 一张 224 × 224 RGB 腕部图像，两帧各 16 维状态，展平为 32 维。状态包含相对 XYZ、行式 rot6d、相对 episode 起点的旋转和夹爪宽度。
   * - 动作
     - 十维：归一化 XYZ、旋转矩阵前两行、米制夹爪宽度。整个 chunk 使用输入观测对应的 EEF 位姿作为固定参考。Stage 1 与 Stage 2 都采用 16 步 chunk。
   * - 奖励
     - 成功 +1、失败 −1、人工中止 0；超时作为截断。
   * - 指令
     - ``ARX_TASK_DESCRIPTION``，与 checkpoint 的任务一致。

``pi05_umi_v4`` 是仓库已有的模型配置标识。使用原 checkpoint 的 ``norm_stats.json``：仅归一化 XYZ，有 min/max 时优先使用，否则使用 q01/q99，不裁剪数值；旋转和夹爪保留原单位。历史部署中的夹爪二值化、世界系 Z 偏移及 EEF 到工具坐标的标定，都应显式核对，不能从配置名称推断。

actor 在 Stage 1 参考动作上预测增量，输出 head 初始为零。XYZ 保留原归一化空间，不统一施加 ``tanh``；行式 rot6d 投影为有效旋转，夹爪宽度使用配置的物理上下界。该残差策略默认关闭参考输入 dropout。

两个 head 都采用 ``action_parameterization: eef_rot6d``，共用动作转换、人工接管编码、replay 和同步执行流程。TD3 配置沿用当前分支的变体：actor 使用双 Q 的最小值，actor 更新时仍采样动作噪声，每两次 critic 更新进行一次 actor 更新，Q 与 BC 的权重固定为 1.0。该配置保留真机采集流程，不启用仿真的采集和权重调度；这里的 TD3 指仓库已实现的 RLT TD3 变体，并不表示采用了标准 TD3 的所有设置。

TD3 的 ``actor_noise_sigma: 0.1`` 作用于模型预测的残差坐标。采样后的残差先乘 ``eef_residual_scale``，再加到参考动作上：默认 XYZ 和六个旋转分量的尺度为 0.1，夹爪宽度的尺度为 0.005。因此，在旋转投影和夹爪限幅前，加入的噪声标准差分别为归一化 XYZ／旋转分量的 0.01，以及夹爪宽度的 0.0005 m。它不表示给位置或夹爪添加 0.1 m 噪声；XYZ 对应的米制位移还取决于原始归一化统计量。评估时关闭这项采样噪声。AC 继续使用原来的 ``fixed_std: 0.002``，该参数不控制 TD3 head。

安装
----

RLinf 使用 Python 3.11，已有的 CPython 3.10 SDK 二进制继续放在独立 Python 3.10 环境。进入当前分支仓库根目录，安装模型和客户端依赖：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env arx5
   source .venv/bin/activate
   export REPO_PATH="$PWD"
   export EMBODIED_PATH="$REPO_PATH/examples/embodiment"

安装目标包含 OpenPI、UVC 采集和 ZMQ 客户端；下面的 SDK 服务使用独立解释器启动。已有模型环境只需补充依赖时，运行 ``python -m pip install -r requirements/embodied/envs/arx5.txt``。

配置模型与时序
--------------

环境和模型转换必须指向同一份原始统计量，并加载包含 ``rlt_module.*`` 权重的完整 Stage 1 checkpoint。基础 π₀.₅ 权重本身不包含训练好的 RLT 特征。默认 RLT 参数对应 ``umi_rlt_stage1_sft_openpi_pi05.yaml``；若 Stage 1 修改过这些参数，应同步修改 ``rollout.rlt_feature_model.openpi``。

解析配置前设置下列环境变量：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 变量
     - 需要填写的内容
   * - ``RLT_STAGE1_CHECKPOINT``
     - 完整 Stage 1 checkpoint 路径，包含已训练的 RLT 模块。
   * - ``RLT_NORM_STATS``
     - 原 checkpoint 的 ``norm_stats.json`` 文件。
   * - ``ARX_ACTION_DT`` / ``ARX_HISTORY_DT``
     - 从真实数据确认的动作步长和状态历史间隔，单位为秒。
   * - ``ARX_CAMERA_LATENCY``
     - 实测图像延迟，单位为秒；从主机接收图像的时间减去该值。
   * - ``ARX_CAMERA_DEVICE``
     - 明确指定腕部采集设备的 ``/dev/v4l/by-id/...`` 路径。
   * - ``ARX_GRIPPER_MIN`` / ``ARX_GRIPPER_MAX``
     - 标定后的米制夹爪宽度上下界，与 SDK 服务转换一致。
   * - ``ARX_TASK_DESCRIPTION``
     - 模型使用的任务指令。

时序参数没有默认值，转换后数据集的 FPS 元信息也不能单独证明真实动作间隔。用 ``toolkits/realworld_check/measure_camera_latency.py`` 测量图像延迟，并核对抖动、裁切、方向、畸变和 GoPro 镜头模式。根据实测采集规格配置 ``env.train.override_cfg.camera_resolution``、``camera_fps``、``camera_pixel_format``\ （默认 ``MJPG``）和 ``image_crop``，同时修改 ``env.eval.override_cfg`` 中的对应参数。

相机提供主机 monotonic 接收时间，并非曝光时间。环境减去配置的延迟，再插值该时刻的机械臂反馈。图像过期或反馈时间间隔过大时停止本轮执行。UVC 子进程用于隔离底层采集阻塞，不负责模型推理，也不会启用 RTC。

启动机械臂服务
--------------

启动 SDK 前，先确认机械臂固定、负载、CAN 接口及工具坐标标定。同一 CAN 接口只能有一个控制进程，先停止原有 SDK 服务。将验证过的参数保存为 JSON 文件，字段如下：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 配置
     - 含义
   * - ``robot`` / ``controller``
     - 显式覆盖 SDK 配置字段，包括验证过的重力补偿设置。
   * - ``controller.default_preview_time``
     - 验证过的 SDK 目标过渡时间（秒），必须大于 ``controller_dt``；它与模型动作步长不同。
   * - ``adapter.eef_to_tcp``
     - 从 SDK EEF 到模型工具坐标的 4 × 4 标定变换。
   * - ``adapter.gripper_scale`` / ``gripper_offset``
     - 转换到 checkpoint 使用的米制夹爪宽度。
   * - ``adapter.drag_kd`` / ``drag_gripper``
     - 六个已验证的拖动阻尼值；夹爪行为选 ``hold`` 或 ``manual``。
   * - ``limits``
     - 每条命令允许的正数上限：``translation``\ （米）、``rotation``\ （弧度）、``gripper``\ （米）、``joint``\ （弧度）。
   * - ``service``
     - 状态采样频率、历史长度及状态和命令超时。
   * - ``calibration_confirmed``
     - 完成当前设备的参数验证后，设为 ``true``。

先用 SDK 解释器启动默认禁止策略运动的服务：

.. code-block:: bash

   /media/lenovo/data/miniconda3/envs/arx-py310/bin/python \
     toolkits/realworld_check/arx5_server.py \
     --sdk-path /home/lenovo/kian/arx5-sdk --model X5 --interface can0 \
     --config /path/to/verified-arx5.json

默认绑定 ``tcp://127.0.0.1:5557``。SDK 构造会启动 CAN 和控制循环，可能施加重力补偿或阻尼，因此即使禁止策略运动，也属于主动控制连接。另开终端运行 ``python toolkits/realworld_check/arx5_server.py --operation state`` 查看反馈。关闭 SDK 所属进程前，应支撑机械臂和负载，因为 SDK 退出可能改变阻尼。进入有人监督的硬件验证和运行阶段时，再加 ``--allow-motion`` 重启服务；该选项要求显式运动限幅、已确认标定和有效的 SDK preview 时间。

设置 ``service.command_timeout`` 前，测量模型命令之间的最大正常间隔，包括同步推理和 learner 更新。默认值为两秒；超时后服务保持实测位姿并切换控制 epoch，读取状态不会续期。查明原因后显式恢复控制，旧动作 chunk 不能自行重新启动。

开始训练
--------

先选择配套的训练与评估配置。原 AC 入口保持可用；使用 TD3 时选择第二组：

.. code-block:: bash

   TRAIN_CONFIG=realworld_arx5_rlt_stage2_ac_mlp
   EVAL_CONFIG=realworld_arx5_rlt_stage2_eval

.. code-block:: bash

   TRAIN_CONFIG=realworld_arx5_rlt_stage2_td3_mlp
   EVAL_CONFIG=realworld_arx5_rlt_stage2_td3_eval

后续启动、恢复和评估沿用选定的配置。先解析完整配置，检查变量和维度；此命令不连接机器人，也不加载 GPU 模型：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py \
     --config-name "$TRAIN_CONFIG" --cfg job --resolve

依次验证状态、图像、小幅模型动作、拖动进入与从实测位置交还控制后，启动同步训练：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh "$TRAIN_CONFIG"

每轮先等待操作者摆好初始位姿并按 ``s``。训练默认执行 Stage 1 参考策略，按 ``b`` 后切换为 Stage 2 actor。环境的键盘监听独立于模型推理，按键如下：

.. list-table::
   :header-rows: 1

   * - 按键
     - 操作
   * - ``s``
     - 从当前实测位姿保持，启用策略控制并开始本轮。
   * - ``b``
     - 后续预测选择 Stage 2 actor。
   * - ``v``
     - 后续预测切回 Stage 1 参考策略。
   * - ``g``
     - 进入重力补偿拖动，作废待执行的模型命令。
   * - ``r``
     - 在实测当前位置保持，重新观测后恢复策略控制。
   * - ``c`` / ``a`` / ``q``
     - 分别标记成功、失败、中止，并保持位置。等待开始时，``q`` 取消等待，``c`` 和 ``a`` 被忽略。

复位由人工完成：重新摆好起始状态，再按 ``s``。等待开始时，``r`` 只保持实测位姿，不启用策略运动。环境不会自动执行关节回零。键盘控制通过 Linux evdev 读取本机键盘；运行用户需要能读取对应的 ``/dev/input`` 设备，并在启动 RLinf 前指定稳定路径：

.. code-block:: bash

   export RLINF_KEYBOARD_DEVICE='/dev/input/by-id/<keyboard>-event-kbd'

选择连接在机器人主机上的实际键盘；监听器读取设备事件，在 SSH 终端输入字符不会触发这些操作。模型推理等待期间，也可通过独立终端的 ``--operation enter_drag`` 或 ``--operation hold`` 直接操作服务，这些服务命令不负责 episode 结果标注。

实测拖动轨迹转换到与 actor 一致的归一化动作空间，通过已有 replay、critic 和人工 BC 流程学习。``algorithm.replay_buffer.filter_invalid_transitions: true`` 排除中断或过期 transition，成功或失败标注会立即保持机械臂，随后按原时间间隔采样剩余 chunk 的保持位姿，并在 chunk 末尾只记录一次终止奖励。这些实测保持动作让结果进入 replay；中止、故障及控制权过期的数据仍作为无效样本处理。操作事件和实测 step 记录以 JSONL 写入 ``runner.logger.log_path/arx5_raw``；该记录不等同于完整图像数据集。

保存、恢复与评估
----------------

恢复训练应保留完整 RLinf checkpoint，其中包括优化器、target network 和当前保留的 replay 窗口。``auto_save: false`` 只关闭 trajectory 的逐条自动保存，actor 保存 checkpoint 时仍会将缓存中的 replay trajectory 写入 ``actor/sac_components/replay_buffer``。默认每十个训练 epoch 保存一次，恢复时指定 ``global_step_<N>`` 目录：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh "$TRAIN_CONFIG" \
     runner.resume_dir=/path/to/checkpoints/global_step_10

恢复时保持 checkpoint 对应的算法和模型维度。AC 与 TD3 的 head 不同，Stage 2 checkpoint 不能互换；两者使用相同的冻结 Stage 1 checkpoint 和归一化统计量。

固定权重评估时，将 ``RLT_STAGE2_WEIGHTS`` 指向所选 head 的 ``actor/model_state_dict/full_weights.pt`` 文件，保留相同的 Stage 1 资产和时序参数，然后运行：

.. code-block:: bash

   export RLT_STAGE2_WEIGHTS=/path/to/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt
   bash evaluations/run_eval.sh "$EVAL_CONFIG"

评估配置包含完整 rollout 模型定义，按 ``s`` 后默认使用 Stage 2。两种算法的观测、同步动作执行和拖动接管都与各自训练时一致，RTC 保持关闭。checkpoint 布局参见 :doc:`恢复训练 </rst_source/guides/resume>`，已有学习目标参见 :doc:`RL Token <rlt>`。

验证范围
--------

CPU 测试覆盖动作转换、服务协议、模拟控制器、UVC 进程生命周期、环境集成和 Hydra 配置。真实轨迹跟踪、当前负载下的重力补偿、GoPro 延迟以及完整 checkpoint 的 GPU 显存占用仍需真机验收。完成这些检查后，再执行短程训练、保存、恢复及评估。当前配置让冻结的 Stage 1 和 Stage 2 共用 GPU，应为二者预留足够显存，不能将小模型测试占用当成完整模型的容量结果。
