# OpenPI 部署与训练的 EEF 数据处理核查

本记录以 `umi-arx-deploy` 的部署调用链为主，对照用户指定的 OpenPI V4 训练配置、数据转换脚本与 YAML，确定 RLinf 适配时应保留的 EEF 语义。这里只做源码和纯函数核查，未修改部署工程、训练代码或启动硬件。

## 核查结论

在观测和动作均使用 `relative`、原始 EEF 参考点一致的条件下，原 V4 部署 helper 与训练转换的核心坐标变换一致，训练动作经部署逆变换可以恢复原轨迹。完整部署与训练尚不能判定为一致：历史入口有额外动作修正，频率与表示依赖外部配置，当前入口则已采用另一套模型观测。

已核查的依据：

- 用户训练配置：[config.py](/home/lenovo/mason/code/openpi-umi/src/openpi/training/config.py:4782)，名称为 `pi05_umi_32d_80k_95_real_umi_batch_72_v4`。
- 用户数据转换：[convert_umi_data_to_lerobot_add_target_value.py](/home/lenovo/mason/code/openpi-umi/examples/umi/convert_umi_data_to_lerobot_add_target_value.py:300) 与 [dataset_config.yaml](/home/lenovo/mason/code/openpi-umi/examples/umi/config/dataset_config.yaml:5)。
- 当前部署：[eval_arx5_pi_hzl.py](/home/lenovo/mason/code/umi-arx-deploy/scripts/eval_arx5_pi_hzl.py:51)、其 `_v2.py` 版本及被调用的 `shellgame_openpi.py`。
- 原 V4 风格入口：部署仓库历史提交 `117bae51e9fda3f102430b0770843bc84d7ea73f` 中的 `scripts/eval_arx5_pi_hzl.py`。后续提交 `6b0c408` 将该入口替换为 ShellGame 流程。历史入口是否就是用户实际使用的版本，仍需启动命令确认。
- 历史入口调用的 [real_inference_util.py](/home/lenovo/mason/code/umi-arx-deploy/utils/real_inference_util.py:53)、`pose_util.py`、`pose_repr_util.py` 与当前仓库对应文件没有版本差异，可用当前文件核查数学实现。

历史源码可只读查看，无需切换工作区：

```bash
git -C /home/lenovo/mason/code/umi-arx-deploy show \
  117bae51e9fda3f102430b0770843bc84d7ea73f:scripts/eval_arx5_pi_hzl.py
```

## 原 V4 部署调用链

1. **SDK 状态转换为部署使用的 TCP 位姿。** ZMQ 客户端将 SDK 的 RPY 转成矩阵，右乘固定工具轴旋转矩阵，再转成 rotvec；这一步保留 XYZ。命令回传时使用逆旋转恢复 SDK 的 RPY。该轴变换属于硬件表达适配，不能遗漏或重复应用。[ee2tcp / tcp2ee](/home/lenovo/mason/code/umi-arx-deploy/modules/arx5_zmq_client.py:105)

2. **按相机时间对齐机械臂历史。** 环境取得相机时间，按 `horizon`、`down_sample_steps` 和 `frequency` 形成观测时间点；对机械臂位姿插值、夹爪线性插值，产生绝对 XYZ、rotvec 和夹爪宽度。两帧历史应对应两个时间点。[get_obs](/home/lenovo/mason/code/umi-arx-deploy/modules/arx5_env.py:397)

3. **构造模型观测。** 历史入口调用 `get_real_umi_obs_dict`，传入同一份 `obs`、外部配置的 `obs_pose_repr` 和 episode 起点。设置为 `relative` 时，历史位姿变为 `inverse(T_current) @ T_history`；另计算相对 episode 起点的旋转，编码为矩阵前两行的 rot6d；夹爪原值传递。[观测编码](/home/lenovo/mason/code/umi-arx-deploy/utils/real_inference_util.py:90)

4. **调用 OpenPI 服务。** 客户端将观测交给 `openpi_policy.infer`。XYZ 归一化和反归一化属于服务端模型 transforms；应使用与训练一致的统计量，不能在客户端重复执行。旧入口使用的服务配置和 checkpoint 尚未确认。

5. **用预测前的同一份观测解码动作。** `get_real_umi_action(raw_action, obs, action_pose_repr)` 取前 9 维组成相对位姿矩阵，最后 1 维为夹爪。`relative` 的逆变换是 `T_current @ T_action_relative`，同一个 chunk 使用固定的 `T_current`，输出绝对 XYZ、rotvec 和夹爪，共 7 维。[动作解码](/home/lenovo/mason/code/umi-arx-deploy/utils/real_inference_util.py:194)

6. **执行前的额外修正和调度。** 历史入口在上述解码之后修改夹爪和 Z，再生成动作时间戳。环境过滤过期目标，将剩余点送入轨迹调度；控制器插值后，经 `tcp2ee` 转回 SDK 命令。额外修正与模型坐标解码必须分别记录。

历史入口的表示方式并非写死为 `relative`：它从 `--input` 对应的 `checkpoints/latest.yaml` 读取 `cfg.task.pose_repr.obs_pose_repr` 和 `action_pose_repr`。helper 本身的默认值为 `abs`，不能只依据函数存在就认定实际启用了相对表示。

## 与指定训练流程逐项对照

| 项目 | 训练侧 | 原 V4 部署侧 | 结论 |
| --- | --- | --- | --- |
| 动作标签来源 | YAML 不加载独立 `action`；sampler 由记录的 XYZ、rotvec、夹爪组成轨迹 | 模型预测作为控制目标 | 标签来自记录的 EEF 轨迹；其上游是 FK、测量还是估计，转换脚本无法证明 |
| 观测基准 | 历史位姿相对当前最后一帧 | `relative` 模式采用相同公式 | 数学一致；实际 YAML 模式待确认 |
| 旋转表示 | 原始 rotvec → 矩阵 → 前两行 rot6d | 同样采用行式 rot6d | 一致，不能换成列式表示 |
| 状态结构 | 两帧，每帧 XYZ3 + 相对旋转6 + 起点相对旋转6 + 夹爪1 | helper 能生成这些字段；形状由外部配置决定 | 必须确认旧部署配置为两帧 |
| 起点相对旋转 | 每个样本对起始 pose 的 6 个分量加入标准差 0.05 的高斯扰动 | 使用实际保存的 episode 起点，不加噪声 | 存在训练增强与推理的预期差异；扰动起点不能用于动作解码 |
| 动作解码 | 标签为 `inverse(T_current) @ T_future` | 解码为 `T_current @ T_relative` | 核心变换互逆 |
| 夹爪数据 | 当前 YAML 未启用二值化；模型归一化保留原宽度 | helper 保留宽度，但历史入口随后做阈值映射 | 完整执行动作不等于模型原输出 |
| 位置修正 | 训练转换没有固定世界 Z 偏移 | 历史入口对所有解码目标 `Z -= 0.008` | 额外 8 mm 修正，不是模型原坐标变换 |
| XYZ 归一化 | 原 checkpoint 的 min/max，缺失时 q01/q99；只处理 XYZ | 应由配套 OpenPI 服务做正逆变换 | 原理可对齐，实际服务配置与统计文件待确认 |

历史入口中实际生效的额外修正（提交 `117bae51`，第 764–770 行）：

```python
action_np[:, -1] = np.where(action_np[:, -1] < 0.05, 0.011, 0.085)
action_np[:, 2] += float(-0.008)
```

之后还会按夹爪上下限裁剪。RLinf 方案应将其列为待确认的部署补偿，不能默认为原 OpenPI 数据约定。

## 时序与字段尚未完全对应

指定 sampler 在原始索引 `t` 取低维观测 `[t-3, t]`，动作取 `[t, t+3, …, t+45]`。所有 latency steps 为 0，因此第一步动作对应当前位姿。开头历史不足时重复首帧；动作尾部不足时重复最后一个仍在步长 3 网格内的有效动作。[采样实现](/home/lenovo/mason/code/openpi-umi/examples/umi/sampler_v2.py:258)

历史部署入口 `--frequency` 默认 8 Hz，history horizon 来自外部 YAML，未覆盖环境的 `robot_down_sample_steps=1`。因此历史间隔是 `1/frequency`。历史入口动作时间戳使用 `obs_timestamp + k/frequency`，并由环境过滤过期点。若原始数据是 60 Hz，则训练的三帧间隔是 0.05 s，与 20 Hz 部署相符；不能据此假定用户没有覆盖 8 Hz 默认值，也不能直接把默认值当作训练频率。

数据转换 YAML 的 `fps: 20` 不会覆盖转换函数的 `fps` 参数。命令行 `--fps` 默认 60，控制写出的 metadata 和 timestamp；转换脚本仍逐原始帧生成样本。因此最终物理间隔应核对采集频率和实际转换命令。[写出时间戳](/home/lenovo/mason/code/openpi-umi/examples/umi/convert_umi_data_to_lerobot_add_target_value.py:414)、[命令行默认值](/home/lenovo/mason/code/openpi-umi/examples/umi/convert_umi_data_to_lerobot_add_target_value.py:871)

当前 `dataset_config.yaml` 的 `img_obs_horizon: 1` 只会输出 `observation.left_wrist_0_rgb_0`，而指定 V4 训练配置重排 `_0`、`_1`，模型使用 `_1`。原部署 helper 则交付 `camera0_rgb` 两帧数组，由配套推理输入 transform 选择最新帧。这三处命名需要核对实际训练数据 schema 与服务配置，不能擅自复制一帧来掩盖差异。[数据字段生成](/home/lenovo/mason/code/openpi-umi/examples/umi/convert_umi_data_to_lerobot_add_target_value.py:270)、[V4 训练重排](/home/lenovo/mason/code/openpi-umi/src/openpi/training/config.py:694)、[V4 推理输入](/home/lenovo/mason/code/openpi-umi/src/openpi/policies/umi_policy.py:51)

## 当前部署入口与原 V4 的区别

当前 `eval_arx5_pi_hzl.py` 与 `_v2.py` 调用 `ShellGamePolicyClient` / `ShellGameHistory`，没有调用原 V4 helper。其观测是相对 episode 起点的单帧 XYZ3 + rot6d6 + 夹爪1，配合 241 帧历史和当前图像，支持服务端历史缓存。这与指定 V4 的两帧 16D 状态不同。[当前观测构造](/home/lenovo/mason/code/umi-arx-deploy/utils/shellgame_openpi.py:395)

当前入口另有沿工具 Z 轴的 `link6 ↔ EEF` 位移（默认 0.145 m）及可选夹爪量程转换。SDK 侧固定工具轴旋转、工具参考点位移、采集坐标原点平移是三件不同的事，不能合并或重复执行。是否需要这些数值，须以这份 V4 训练数据的采集参考点为准；不自动继承当前 ShellGame 的标定。[参考点转换](/home/lenovo/mason/code/umi-arx-deploy/utils/shellgame_openpi.py:26)、[当前夹爪转换](/home/lenovo/mason/code/umi-arx-deploy/scripts/eval_arx5_pi_hzl.py:110)

## 已完成的纯函数验证与后续核对

使用同一组非零位姿、变化旋转和夹爪宽度，直接调用指定训练采样/转换函数及原部署 helper，明确设置 `relative`；未导入设备类：

- 相对 XYZ、相对 rot6d 和夹爪的训练/部署差异为 0。
- 对齐同一扰动起点后，`wrt_start` 最大误差约 `2.59e-8`；使用真实起点时存在增强导致的差异，符合源码。
- 训练动作经部署逆变换恢复原采样轨迹，XYZ 最大误差约 `2.98e-8`，旋转矩阵最大误差约 `6.58e-8`，夹爪误差为 0。
- 验证范围不包含历史入口额外的夹爪二值化、Z 偏移，也不包含真实模型服务、相机时间同步或机械臂执行。

继续对齐所需信息是：用户实际 V4 部署启动命令及其 `latest.yaml`、模型服务配置/checkpoint/统计文件、训练数据 schema、原始采样频率，以及采集时的 EEF 参考点和夹爪标定。确认后，在 RLinf 内复现这条经核对的处理链；旧部署工程作为行为参考，不作为运行依赖。
