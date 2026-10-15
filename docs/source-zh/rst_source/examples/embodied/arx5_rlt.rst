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
     - 人工标注成功或失败，结束后平滑返回配置的起始位姿
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

RLinf 使用 Python 3.11，ARX SDK 服务使用独立的 Python 3.10 环境。进入当前分支仓库根目录，安装模型和客户端依赖：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env arx5
   source .venv/bin/activate
   export REPO_PATH="$PWD"
   export EMBODIED_PATH="$REPO_PATH/examples/embodiment"

安装目标包含 OpenPI、UVC 采集和 ZMQ 客户端；下面的 SDK 服务使用独立解释器启动。已有模型环境只需补充依赖时，运行 ``python -m pip install -r requirements/embodied/envs/arx5.txt``。

仓库在 ``third_party/arx5-sdk`` 中包含基于官方提交 ``c160db7`` 的 ARX SDK，并加入了本地 emergency 处理修复，来源、许可证和基础版本记录在 :download:`third_party/README.md <../../../../../third_party/README.md>`。用 SDK 的 Python 3.10 环境构建扩展；另开终端，从仓库根目录运行：

.. code-block:: bash

   conda activate arx-py310
   (
   set -e
   cd third_party/arx5-sdk
   mkdir build && cd build
   cmake ..
   make -j
   )

以上按上游 README 创建新的构建目录。已有 ``build`` 目录时，进入 ``third_party/arx5-sdk/build``，依次运行 ``cmake ..`` 和 ``make -j``。命令使用 ``arx-py310`` 中的 C++ 依赖构建全部默认目标，并将 Python 扩展编译到 ``third_party/arx5-sdk/python``。保留 SDK 的 ``models``、``lib`` 和 ``python`` 目录结构。需要新建 SDK 环境时，参照 ``third_party/arx5-sdk/README.md`` 及其中的 ``conda_environments/py310_environment.yaml``。后面的服务命令在这个 SDK 终端执行，训练命令在 RLinf 终端执行。

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
   * - ``ARX_RESET_POSE``
     - 固定回程目标，格式为服务 TCP 坐标系下的 ``[x,y,z,rx,ry,rz]``：XYZ 单位为米，后三项为弧度制旋转向量。
   * - ``ARX_RESET_GRIPPER``
     - 回程后的夹爪宽度，单位为米；默认 0.085，必须位于配置的夹爪上下界内。

时序参数没有默认值，转换后数据集的 FPS 元信息也不能单独证明真实动作间隔。用 ``toolkits/realworld_check/measure_camera_latency.py`` 测量图像延迟，并核对抖动、裁切、方向、畸变和 GoPro 镜头模式。根据实测采集规格配置 ``env.train.override_cfg.camera_resolution``、``camera_fps``、``camera_pixel_format``\ （默认 ``MJPG``）和 ``image_crop``，同时修改 ``env.eval.override_cfg`` 中的对应参数。

相机提供主机 monotonic 接收时间，并非曝光时间。环境减去配置的延迟，再插值该时刻的机械臂反馈。图像过期或反馈时间间隔过大时停止本轮执行。UVC 子进程用于隔离底层采集阻塞，不负责模型推理，也不会启用 RTC。

配置每个 episode 完成后机械臂应返回的位姿。启动前，将下面的占位符替换为当前设备已确认的目标：

.. code-block:: bash

   export ARX_RESET_POSE='[<x>,<y>,<z>,<rx>,<ry>,<rz>]'
   export ARX_RESET_GRIPPER='<initial gripper width in metres>'

目标使用与服务实测反馈、绝对位姿命令相同的 TCP 坐标系。默认 ``adapter.eef_to_tcp`` 为单位矩阵，此时使用 SDK EEF 坐标系。迁移原有部署的位姿时，需要先应用其中的原点、工具偏移和朝向变换；UMI TCP 的旋转向量不能直接作为 SDK EEF 朝向。这个固定回程目标与每个 episode 的观测参考位姿分别保存。

以下环境选项可配置在 ``env.train.override_cfg`` 和 ``env.eval.override_cfg`` 下。默认启用自动回程，需要填写明确的目标：

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 选项
     - 默认值与行为
   * - ``return_on_episode_end``
     - ``true``：成功、失败或达到 episode 步数上限后回程；设为 ``false`` 时由人工恢复位姿。
   * - ``reset_pose``
     - 从 ``ARX_RESET_POSE`` 读取六个有限数值，不预设其他设备的目标位姿。
   * - ``reset_gripper_width``
     - 从 ``ARX_RESET_GRIPPER`` 读取，默认 0.085 米。
   * - ``reset_duration`` / ``reset_frequency``
     - 默认按 4.0 秒、20.0 Hz 生成目标；位置采用平滑加减速，姿态采用最短路径插值。逐次发送会计入通信等待，实际回程至少持续 4.0 秒。
   * - ``reset_settle_timeout``
     - 最后一个目标发出后，最多再等待 2.0 秒到达目标，然后保持。

实测位置误差不超过 5 毫米、姿态误差不超过 0.05 弧度、夹爪宽度误差不超过 5 毫米时，回程完成。超过末段等待时间仍未到达，会停止并报错。回程全程使用相同的服务命令限幅和实时反馈检查。

启动机械臂服务
--------------

启动 SDK 前，先确认机械臂固定、负载、CAN 接口及工具坐标标定。同一 CAN 接口只能有一个控制进程，先停止原有 SDK 服务。RLinf 在 ``tcp://127.0.0.1:5557`` 上使用 JSON 服务协议；SDK 原有的 ``python/communication/zmq_server.py`` 使用端口 ``8765`` 和另一套协议，不能直接供 RLinf 客户端连接。

服务默认加载已加入修复的 ``third_party/arx5-sdk``，也可以通过 ``--sdk-path`` 指定其他 SDK 目录；外部目录不会获得这些本地修复。模型和 CAN 接口既可写成位置参数 ``X5 can0``，也可写成 ``--model X5 --interface can0``。先查看实际生效的服务配置：

.. code-block:: bash

   "$CONDA_PREFIX/bin/python" toolkits/realworld_check/arx5_server.py \
     X5 can0 --print-config

``--print-config`` 输出合并后的软件默认值和 JSON 覆盖项即退出，不导入 SDK，也不连接硬件。服务以 SDK 的机械臂和控制器默认值为基础，采用 ``toolkits/realworld_check/arx5_server_defaults.json`` 中的软件参数，见下表。已有部署如果在 Python 脚本中设置了参数，将这些显式覆盖项写入 JSON，再通过 ``--config /path/to/arx5.json`` 加载。JSON 按 section 内的字段覆盖默认值，未填写的字段保留默认值。

服务从 SDK 配置工厂读取 ``cartesian_controller`` 参数，创建 ``Arx5CartesianController``，并通过 ``set_eef_cmd`` 下发末端目标。IK 重试、关节目标生成和插值由 SDK 处理。本地 emergency 修复保留正常 Cartesian IK 行为和 SDK 默认参数。服务默认配置中的 ``controller`` 为空，因此沿用 SDK 的 Cartesian controller 增益和 0.1 秒插值预览时间。

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 配置
     - 含义
   * - ``robot`` / ``controller``
     - 覆盖 SDK 配置字段，包括已有部署中的夹爪标定、增益和重力补偿设置。
   * - ``controller.default_preview_time``
     - SDK 目标过渡时间，沿用 ``cartesian_controller`` 的 0.1 秒默认值，必须大于 ``controller_dt``；它与模型动作步长不同。
   * - ``adapter.eef_to_tcp``
     - 从 SDK EEF 到模型工具坐标的 4 × 4 变换，默认为单位矩阵。
   * - ``adapter.gripper_scale`` / ``gripper_offset``
     - 转换到 checkpoint 使用的米制夹爪宽度，默认分别为 1.0 和 0.0，即保留 SDK 的米制数值。
   * - ``adapter.drag_kd`` / ``drag_gripper``
     - 六个拖动阻尼值，默认采用控制器的 ``default_kd``；夹爪行为选 ``hold``\ （默认）或 ``manual``。
   * - ``limits``
     - 每条命令允许的正数上限，默认值：``translation`` 为 0.05 米、``rotation`` 为 0.2 弧度、``gripper`` 为 0.02 米。
   * - ``service``
     - 状态采样频率、历史长度及状态和命令超时；此入口的 ``command_timeout`` 默认值为 10 秒。
   * - ``calibration_confirmed``
     - 可选的部署状态；显式填写 ``false`` 会阻止 ``--allow-motion``，省略时允许通过运动开关启动。

单位矩阵表示直接使用 SDK EEF 坐标系。应核对该坐标系和 SDK 夹爪宽度是否与训练数据一致；这些默认值并不表示已经匹配 UMI checkpoint。运动限幅也需结合当前设备和任务检查。原部署如果在 Python 脚本中修改了夹爪 readout、宽度或增益，应将这些覆盖项一并迁移到 JSON。

先用 SDK 解释器启动默认禁止 policy 运动的服务：

.. code-block:: bash

   "$CONDA_PREFIX/bin/python" toolkits/realworld_check/arx5_server.py X5 can0

默认绑定 ``tcp://127.0.0.1:5557``。SDK 构造会启动 CAN 和控制循环，可能施加重力补偿或阻尼，因此即使禁止 policy 运动，也属于主动控制连接。另开终端运行 ``python toolkits/realworld_check/arx5_server.py --operation state`` 查看反馈。关闭 SDK 所属进程前，应支撑机械臂和负载，因为 SDK 退出可能改变阻尼。进入有人监督的硬件验证和运行阶段时，再加 ``--allow-motion`` 重启服务；有部署覆盖项时，同时传入 ``--config``。RLinf 在调用 SDK 前检查末端单步变化、夹爪行程、命令时限、控制 epoch 和实时反馈，运动开关不会绕过这些检查或命令 watchdog。正常 IK 与关节处理沿用官方 Cartesian controller 的实现。

设置 ``service.command_timeout`` 前，测量模型命令之间的最大正常间隔，包括同步推理和 learner 更新。此入口默认值为 10 秒；超时后服务保持实测位姿并切换控制 epoch，读取状态不会续期。查明原因后显式恢复控制，旧动作 chunk 不能自行重新启动。

本地 SDK 修复会在进入 emergency 处理前释放状态与命令锁，初始化校验失败时抛出 ``RuntimeError``，后台 emergency 阻尼循环可在析构时退出。进入 fault 后，Joint 和 Cartesian controller 都会拒绝后续运动命令；处理原始故障后需要重建 controller 或重启服务。

SDK 在服务进程内直接运行。服务运行期间按 ``Ctrl+C`` 或发送 ``SIGTERM`` 会请求 ``service.stop()``，正常退出由 SDK 完成清理；退出前先支撑机械臂和负载。使用 ``third_party/arx5-sdk`` 中编译的 SDK 才能获得 emergency 修复，引发 emergency 的夹爪标定错误仍需处理后再启动。

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

环境启动后，在机器人主机的浏览器中打开 `本机相机预览 <http://127.0.0.1:8765/>`_。日志会打印这个地址，程序不会自动弹出窗口。AC 和 TD3 训练默认开启预览，评估默认关闭。页面显示按原比例缩小的相机画面、EP 编号和当前状态，等待 ``s``、模型推理和回程期间仍持续刷新；画面过期或连接中断时会显示提示。

EP 从 1 开始，每次成功开始一个新 episode 才递增，等待开始时显示即将开始的编号。重复按 ``s``、使用 ``g``、``r``、``b``、``v`` 或完成一个 64 步 rollout 都不会增加编号。进程重启或从 checkpoint 恢复后，编号重新从 1 开始，不能据此读取 runner 的 ``global_step`` 或 replay trajectory 数量。

关闭页面不会停止采集。通过 ``env.train.override_cfg.enable_camera_preview=false`` 关闭预览，通过 ``env.train.override_cfg.camera_preview_port`` 修改默认端口 ``8765``。评估使用对应的 ``env.eval.override_cfg`` 参数；训练占用同一端口时，保持评估预览关闭。端口冲突会产生警告，预览不可用时采集仍继续。预览中的缩放和 EP 显示不会改变模型输入或 replay 中的图像。视频录制由 ``video_cfg.save_video`` 单独控制，这些配置中仍为 ``false``。

通过 ``algorithm.update_epoch`` 和 ``algorithm.train_actor`` 选择要更新的网络。后者在两种 ARX 配置中均默认为 ``true``；切换阶段时需要显式设置这两个参数：

.. list-table::
   :header-rows: 1

   * - ``algorithm.update_epoch``
     - ``algorithm.train_actor``
     - 行为
   * - ``0``
     - ``false``
     - 采集和保存数据，actor 与 Q 均不更新。
   * - 正数
     - ``false``
     - replay 达到训练门槛后更新 Q。
   * - 正数
     - ``true``
     - replay 达到训练门槛后，按配置的比例更新 Q 和 actor。

设置 ``algorithm.train_actor=false`` 后，会跳过 actor 和熵温度 alpha 的优化器及学习率调度；Q 优化器、Q 学习率调度和 target model 的 EMA 仍按原配置执行。当前 RLT MLP 的 actor 与 ``q_head`` 参数独立，因此在线 actor 保持初始化或恢复时的权重，Stage 1 继续冻结。AC 仍使用在线 actor 生成 TD 目标动作，并保留 ``fixed_std: 0.002`` 的采样噪声；TD3 继续使用 target actor，其参数遵循原有 target model 更新规则。``start_with_actor=false`` 选择机械臂执行的参考动作分支，不会将 TD 目标动作改为精确的参考动作。

发生学习更新时，``train/actor/updated`` 是各次更新是否实际执行 actor 优化的 0/1 均值，仅训练 Q 时应为 0。启用 RLT 调度后，``train/rlt/actor_updates_run`` 记录本轮实际 actor 更新次数。runner 的 ``global_step`` 表示采集与训练迭代进度，不是优化器更新次数。

首次启动时，先摆好机械臂和物体的初始状态，再按 ``s``。训练默认执行 Stage 1 参考 policy，按 ``b`` 后切换为 Stage 2 actor。环境的键盘监听独立于模型推理，按键如下：

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
   * - ``c`` / ``a``
     - 分别标记成功或失败，保持到当前 chunk 结束后返回配置的起始位姿；等待开始时忽略这两个按键。
   * - ``q``
     - 中止并保持位置，不执行回程；等待开始时取消等待。

标记成功或失败后，环境先记录终止 transition，再平滑回程并保持到达的位姿。达到 episode 步数上限时也会回程。恢复物体的初始状态后，再按 ``s`` 开始下一个 episode。等待开始时，``r`` 只保持当前实测位姿。

回程期间，``g`` 取消回程并进入拖动，``q`` 或 ``r`` 取消回程并保持当前实测位姿；此时忽略 ``s``、``b``、``v``、``c`` 和 ``a``。取消后不会自动重试回程。通过 ``g`` 接管后的拖动状态会保留到自动 reset 阶段，直到按 ``s`` 或 ``r`` 恢复保持。中止 episode、发生故障或关闭环境，都不会主动发起回程。

回程在终止 step 返回前完成，``auto_reset=false`` 时也会执行。默认 ``auto_reset=true`` 会进入下一次 reset 并等待 ``s``，即使刚结束的 episode 位于最后一个 rollout chunk 中，这次等待仍可能推迟当前迭代的保存。训练 rollout 为 64 步，短于配置的 320 步 episode 上限；只结束一个 rollout 不会触发回程。

键盘控制默认通过 Linux evdev 同时监听机器人主机上所有可读取的键盘。按 :doc:`franka` 的说明授予运行用户读取键盘事件设备的权限，并在启动 RLinf 或 Ray 前清除旧的单设备选择：

.. code-block:: bash

   unset RLINF_KEYBOARD_DEVICE

连接在机器人主机上的任意实体键盘都可以触发这些操作；在 SSH 终端输入字符不会产生相应的设备事件。监听器定期扫描新接入且可读取的键盘，拔掉一个键盘后，其余键盘仍可使用。启动时至少需要一个可读取的键盘，新接入或重新插拔的设备可能需要重新授予读取权限。需要只监听一个设备时，仍可设置 ``RLINF_KEYBOARD_DEVICE`` 为它的事件设备路径。模型推理等待期间，也可通过独立终端的 ``--operation enter_drag`` 或 ``--operation hold`` 直接操作服务，这些服务命令不负责 episode 结果标注。

实测拖动轨迹转换到与 actor 一致的归一化动作空间，通过已有 replay、critic 和人工 BC 流程学习。``algorithm.replay_buffer.filter_invalid_transitions: true`` 排除中断或过期 transition，成功或失败标注会立即保持机械臂，随后按原时间间隔采样剩余 chunk 的保持位姿，并在 chunk 末尾只记录一次终止奖励。这些实测保持动作让结果进入 replay；中止、故障及控制权过期的数据仍作为无效样本处理。终态观测取自回程前，回程动作不进入 replay。操作事件、实测 step 和独立回程事件以 JSONL 写入 ``runner.logger.log_path/arx5_raw``；该记录不等同于完整图像数据集。

保存、恢复与评估
----------------

恢复训练应保留完整 RLinf checkpoint，其中包括优化器、target network 和 replay 数据。保存 replay checkpoint 时，会先等待已提交的 trajectory 写入完成，再将相互匹配的元数据、索引和 trajectory 文件保存到 ``actor/sac_components/replay_buffer``。启用 ``auto_save: true`` 时，checkpoint 包含索引中的全部已持久化 trajectory；即使使用 ``algorithm.update_epoch=0`` 采集、尚未进行采样，也会保存这些数据。``auto_save: false`` 关闭 trajectory 的逐条自动写入，checkpoint 仍会保存当前保留在缓存中或可从磁盘读取的数据。从未写入磁盘且已从内存淘汰的 trajectory 不会进入 checkpoint，恢复后的 trajectory 数量和样本总数按实际保存的数据计算。恢复后再次保存时，会按各条 trajectory 的来源读取数据，同时包含从原 checkpoint 加载的 trajectory 和仍保留的新采集数据。

后台写入失败或所需 trajectory 文件缺失时，checkpoint 保存会报错。归档前应等待保存成功完成；仅看到 checkpoint 目录不能证明保存已完成。这个过程不会补齐旧 checkpoint 中缺失的文件，也不保证强制退出后仍能完成待处理的写入。环境尚未发送到 replay 的 transition 不在快照范围内。

默认每十个训练 epoch 保存一次。从采集 checkpoint 开始训练 Q 时，将下面的路径替换为实际的 ``global_step_1`` 目录。此示例再执行一个迭代并保存结果。Q 更新仍需达到 ``algorithm.replay_buffer.min_buffer_size: 64`` 的门槛，这里的 64 指 trajectory 条目数；只采集了首轮时，下一轮仍可能不执行任何 Q 更新。此命令用于演示恢复参数，实际 Q 训练需要设置相应的累计迭代上限。请在工程根目录执行：

.. code-block:: bash

   python examples/embodiment/train_embodied_agent.py --config-name "$TRAIN_CONFIG" \
     algorithm.update_epoch=8 \
     algorithm.train_actor=false \
     runner.resume_dir=/path/to/checkpoints/global_step_1 \
     runner.max_epochs=2 \
     runner.save_interval=1

``runner.max_epochs`` 是累计迭代上限，必须大于加载的 ``global_step``；从更晚的 checkpoint 恢复时，应同步增大该值。这个入口仍会初始化真机环境，每轮先采集 rollout，再从 replay 学习。准备同时训练 actor 时，需要显式设置 ``algorithm.train_actor=true``。

恢复时保持 checkpoint 对应的算法和模型维度。AC 与 TD3 的 head 不同，Stage 2 checkpoint 不能互换；两者使用相同的冻结 Stage 1 checkpoint 和归一化统计量。

固定权重评估时，将 ``RLT_STAGE2_WEIGHTS`` 指向所选 head 的 ``actor/model_state_dict/full_weights.pt`` 文件，保留相同的 Stage 1 资产和时序参数，然后运行：

.. code-block:: bash

   export RLT_STAGE2_WEIGHTS=/path/to/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt
   bash evaluations/run_eval.sh "$EVAL_CONFIG"

评估配置包含完整 rollout 模型定义，按 ``s`` 后默认使用 Stage 2。两种算法的观测、同步动作执行和拖动接管都与各自训练时一致，RTC 保持关闭。checkpoint 布局参见 :doc:`恢复训练 </rst_source/guides/resume>`，已有学习目标参见 :doc:`RL Token <rlt>`。

验证范围
--------

CPU 测试覆盖动作转换、服务协议、模拟控制器、UVC 进程生命周期、环境集成和 Hydra 配置。真实轨迹跟踪、当前负载下的重力补偿、GoPro 延迟以及完整 checkpoint 的 GPU 显存占用仍需真机验收。完成这些检查后，再执行短程训练、保存、恢复及评估。当前配置让冻结的 Stage 1 和 Stage 2 共用 GPU，应为二者预留足够显存，不能将小模型测试占用当成完整模型的容量结果。
