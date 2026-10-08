RL Token：借助视觉-语言-动作模型启动在线强化学习
================================================

**RL Token: Bootstrapping Online RL with Vision-Language-Action Models** 在冻结的 VLA 特征模型之上训练一个轻量级强化学习策略。在 RLinf 的配置和代码中，这套流程简称为 **RLT**。整个流程分为两个阶段：

1. 在示范数据上联合训练 VLA 检查点和 RLT token transformer。
2. 冻结第一阶段得到的特征模型，用提取出的 RLT 状态训练一个轻量级 off-policy actor-critic。

当前仓库中的示例配置覆盖 UMI V4 Stage 1 训练、Franka peg insertion 和 ManiSkill ``PegInsertionSideWideClearance-v1`` joint-control 仿真。只要示范数据、环境配置、动作维度、状态语义和 OpenPI dataconfig 对齐，就可以复用相同的两阶段结构。

官方项目页：`Precise Manipulation with Efficient Online RL <https://www.pi.website/research/rlt>`_。

概览
----

RLT 将表示学习和在线 RL 控制拆开。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Stage 1
      :text-align: center

      VLA SFT + RLT token transformer

   .. grid-item-card:: Stage 2
      :text-align: center

      轻量级 off-policy actor-critic

   .. grid-item-card:: 状态
      :text-align: center

      ``z_rl`` + proprio + reference chunk

   .. grid-item-card:: 部署
      :text-align: center

      Franka 真机 / ManiSkill 仿真

| **你将完成：** 准备示范数据 -> 训练 Stage 1 -> 在 Stage 2 中加载 Stage 1 检查点 -> 启动 actor-critic 训练 -> 观察 replay buffer 与任务成功率指标。
| **前置条件：** 准备好 `OpenPI π₀.₅ <https://huggingface.co/lerobot/pi05_base>`__ 基座模型，并按所选示例配置 :doc:`Franka 真机环境 <../embodied/franka>` 或 :doc:`ManiSkill 仿真环境 <../embodied/maniskill>` （二选一）。

提供的配置文件
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - 阶段
     - 配置
     - 作用
   * - UMI Stage 1
     - ``examples/sft/config/umi_rlt_stage1_sft_openpi_pi05.yaml``
     - 使用 UMI V4 示范数据，通过 FP32 短训练检查 Pi0.5 与 RLT 联合训练。
   * - Franka Stage 1
     - ``examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml``
     - 在 Franka 示范数据上联合 SFT π₀.₅ 和 RLT token transformer。
   * - Franka Stage 2
     - ``examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml``
     - 使用冻结的 Stage 1 特征模型，在真机上训练 RLT actor-critic。
   * - ManiSkill Stage 1
     - ``examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml``
     - 联合训练 ManiSkill OpenPI 基座和 RLT token transformer。
   * - ManiSkill Stage 2
     - ``examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml``
     - 使用自动 ``rlt_policy_switch`` 和 transition replay 训练仿真 RLT actor-critic。
   * - ManiSkill Stage 2（TD3）
     - ``examples/embodiment/config/maniskill_rlt_stage2_td3_mlp.yaml``
     - 使用相同的冻结 Stage 1 特征模型和 replay 路径运行仿真 TD3-MLP 变体。

安装
----

1. 克隆 RLinf 仓库
~~~~~~~~~~~~~~~~~~

.. code:: bash

   # 为提高国内下载速度，可以使用：
   # git clone https://gh-proxy.com/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

2. 安装依赖
~~~~~~~~~~~

**方式一：Docker 镜像**

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 20g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-maniskill_libero
      # 为提高国内下载速度，可以使用：
      # infinigence-ai-registry.cn-beijing.cr.aliyuncs.com/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

进入容器后，切换到 OpenPI 虚拟环境：

.. code:: bash

   source switch_env openpi

**方式二：自建环境**

.. code:: bash

   # 为提高国内依赖安装速度，可以添加 `--use-mirror` 到下面的 install.sh 命令

   bash requirements/install.sh embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

下面的 RLT 配置使用 ``model_type: openpi_rlinf``。安装命令仍使用
``--model openpi``，因为 vendored PyTorch 模型复用 OpenPI 运行环境，并且 RLT
Stage 1 dataloader 为了保持 ManiSkill 和真机行为一致，仍使用 OpenPI 数据管线。
这里的 ``openpi_rlinf`` 是 RLinf vendored、已对齐 JAX OpenPI 参考实现精度的
PyTorch Pi0.5 路径，不是旧的官方 OpenPI PyTorch 路径。

RLT 如何工作
------------

Stage 1：学习 RLT 特征模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 从一个 VLA 检查点开始，在同一批示范数据上优化两个目标：

1. 普通 VLA 动作预测目标，日志中记为 ``vla_loss``。
2. RLT token 目标，日志中记为 ``rlt_loss``。

Stage 1 的总损失为：

.. code:: text

   total_loss = rlt_loss + rlt_alpha * vla_loss

OpenPI 模型会暴露 VLA prefix hidden states。RLT token transformer 读取这些 prefix states，并输出紧凑向量 ``z_rl``。Stage 2 使用 ``z_rl`` 作为学习到的 RL 表示，而不是直接在图像观测上训练 actor-critic。

Stage 1 中比较关键的字段：

.. code:: yaml

   # examples/sft/config/realworld_rlt_stage1_sft_openpi_pi05.yaml
   data:
     train_data_paths: "/path/to/data"

   actor:
     model:
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"
       model_type: "openpi_rlinf"
       is_lora: False
       model_path: "/path/to/model"
       num_action_chunks: 20
       action_dim: 7
       num_steps: 4
       add_value_head: False
       openpi:
         task: sft
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         # Network prediction horizon (Pi0Config.action_horizon). Matches official
         # TrainConfig for config_name pi05_franka_state. Distinct from action_chunk
         # / num_action_chunks, which is the env-executed (RLT) window.
         action_horizon: 20
         action_chunk: ${actor.model.num_action_chunks}
         action_env_dim: ${actor.model.action_dim}
         num_steps: ${actor.model.num_steps}
         model_action_dim: 32
         paligemma_variant: "gemma_2b"
         action_expert_variant: "gemma_300m"
         max_token_len: 200
         discrete_state_input: True
         use_rlt: True
         rlt_alpha: 1.0
         rlt_prefix_seq_len: 1024
         rlt_image_only: False
         rlt_use_mask: True

``repo_id``、``config_name`` 和归一化统计需要与 Stage 2 特征模型配置保持一致。
``openpi_data`` 应写在 ``actor.model`` 下。

需要从 VLA 目标中排除 padding action 时，将 ``actor.model.openpi.action_loss_mask`` 配置为长度等于 ``model_action_dim`` 的非负权重。VLA loss 先将各维平方误差加权求和，再除以权重和，最后对 batch 和 action horizon 取平均。权重必须有限且总和大于零；省略 mask 时仍对模型全部 action 维度取平均。普通 SFT 和 RLT Stage 1 都使用这个 mask，RLT 重建目标保持不变。

.. note::

   ManiSkill joint 示例不依赖 ``state_indices`` 手工切片。``pi05_rlt_maniskill_joint``
   dataconfig 将 LeRobot 数据中的 ``state`` 映射到 OpenPI ``observation.state``；
   Stage 2 rollout 也使用 OpenPI 处理后的 ``observation.state`` 作为 ``proprio``。
   这样 Stage 1、Stage 2 和 OpenPI normalization 看到的是同一种状态语义。

Stage 2：训练 Actor-Critic 策略
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 2 冻结 Stage 1 特征模型，只训练轻量级 RLT MLP actor 和 critic。

.. note::

   当前 Stage 2 实现不是标准 maximum-entropy SAC。

rollout 时：

1. 环境返回原始观测和任务元信息。
2. ``rollout.rlt_feature_model`` 运行冻结的 Stage 1 模型，将原始观测转换为：

   - ``z_rl``：紧凑的 RLT 表示。
   - ``proprio``：选中的机器人或仿真状态。
   - ``ref_chunk``：VLA reference action chunk。

3. Stage 2 actor 的输入是 ``ref_chunk``、``z_rl`` 和 ``proprio``。
4. replay buffer 存储 RLT transition：

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = 实际发送给环境的动作
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

Franka 真机配置中，``keyboard_reward_wrapper: rlt_policy_switch`` 会提供
``rlt_switch_flags`` 标记。在操作员按下 ``b`` 之前，实际执行 VLA 的
``ref_chunk``；按下 ``b`` 之后，实际执行 Stage 2 actor 的动作。

ManiSkill joint 配置中，``env.*.rlt_policy_switch`` 使用任务信息自动产生 ``rlt_switch_flags``\ （actor/ref 阶段）和 ``intervene_flag``\ （expert 接管请求）。HF rollout route 根据这些标记在整 chunk 级别选择 actor action、VLA ``ref_chunk`` 或 expert action，并把 ``record_transition``、``actor_switch`` 和 ``intervention_requested`` 写入 ``forward_inputs``，供 replay 和监控使用。

critic 使用 chunked rewards 上的 TD target。一个 action chunk 内的奖励会先按折扣累计，然后用下一状态 Q 值 bootstrap：

.. code:: text

   target_q = discounted_chunk_reward + gamma ** chunk_horizon * Q_target(next_obs, next_action)

如果 episode 已终止，并且 ``bootstrap_type`` 为 ``standard``，则不会加入 bootstrap 项。

Stage 2 中比较关键的字段：

.. code:: yaml

   # examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml
   algorithm:
     loss_type: rlt_ac
     q_weight: 0.1
     bc_weight: 5
     reference_dropout_prob: 0.5
     gamma: 0.96
     entropy_tuning:
       alpha_type: fixed_alpha
       initial_alpha: 0.0

   rollout:
     collect_transitions: True
     model:
       model_path: null
       precision: ${actor.model.precision}
       action_dim: ${actor.model.action_dim}
       num_action_chunks: ${actor.model.num_action_chunks}
       ref_num_action_chunks: ${actor.model.ref_num_action_chunks}
     rlt_feature_model:
       model_type: "openpi_rlinf"
       precision: bf16
       is_lora: False
       num_action_chunks: 20
       action_dim: 7
       num_steps: 4
       add_value_head: False
       model_path: "/path/to/stage1/checkpoint/actor"
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"
         norm_stats_path: /path/to/lerobot_dataset/norm_stats.json
       openpi:
         task: eval
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         # Network prediction horizon (Pi0Config.action_horizon). Matches official
         # TrainConfig for config_name pi05_franka_state. Distinct from action_chunk
         # / num_action_chunks, which is the env-executed (RLT) window.
         action_horizon: 20
         action_chunk: ${actor.model.ref_num_action_chunks}
         action_env_dim: ${rollout.rlt_feature_model.action_dim}
         num_steps: ${rollout.rlt_feature_model.num_steps}
         model_action_dim: 32
         paligemma_variant: "gemma_2b"
         action_expert_variant: "gemma_300m"
         max_token_len: 200
         discrete_state_input: True
         state_indices: []      # 保留完整 raw state；例如 19D state
         use_rlt: True
         rlt_prefix_seq_len: 1024
         rlt_image_only: False
         rlt_use_mask: True

   actor:
     model:
       model_type: "rlt_mlp_policy"
       precision: fp32
       add_value_head: False
       add_q_head: True
       q_head_type: "default"
       fixed_std: 0.002
       is_lora: False
       z_dim: 2048
       proprio_dim: 19
       action_dim: 7
       num_action_chunks: 10
       ref_num_action_chunks: 20

   env:
     train:
       keyboard_reward_wrapper: rlt_policy_switch

Stage 2 的 actor loss 为：

.. code:: text

   actor_loss = -q_weight * Q(obs, pi(obs)) + bc_weight * BC(pi(obs), target_action)

普通 policy step 中，BC target 是 VLA reference action。如果某一步存了
human intervention action，那么这一步的 BC target 会切换成人类动作。ManiSkill
默认关闭 ``expert_takeover``，因此默认路线只使用 VLA reference 和 actor action。

运行 UMI Stage 1
----------------

使用 ``examples/sft/config/umi_rlt_stage1_sft_openpi_pi05.yaml``，在 UMI V4 示范数据上联合训练转换后的原生 PyTorch Pi0.5 与 RLT token transformer。``pi05_umi_v4`` dataconfig 要求 LeRobot 数据集的每一行已经包含 16 步、每步 10 维的 action chunk。

每条 observation 包含两帧 16 维 state 和最新的左腕 RGB 图像。数据加载器在 prompt 和离散 state 的 tokenization 阶段保留 ``(2, 16)`` 的 state shape，完成后再展平为 32 维。归一化使用原 checkpoint 的 min/max 统计，仅处理 state 和 action 的 XYZ；旋转与夹爪数值保持原有尺度。模型输入的 action 从 ``(16, 10)`` padding 到 ``(16, 32)``。示例的 ``action_loss_mask`` 前 10 项为 1、后 22 项为 0，VLA 目标仅对 10 个真实 action 维度取平均。

启动前，在示例配置中填写数据集、转换后的 checkpoint 及其原始归一化文件路径：

.. code:: yaml

   data:
     train_data_paths: /path/to/umi_lerobot_dataset
   actor:
     model:
       model_path: /path/to/pi05_umi_fp32
       precision: fp32
       openpi_data:
         repo_id: umi_v4
         norm_stats_path: /path/to/pi05_umi_fp32/assets/norm_stats.json
       openpi:
         config_name: pi05_umi_v4

示例使用 8 张 GPU 执行 20 个 optimizer step，每张 GPU 的 micro batch size 为 1，global batch size 为 8；模型参数和 FSDP 通信均使用 FP32。在 OpenPI 环境中，从仓库根目录启动：

.. code:: bash

   bash examples/sft/run_vla_sft.sh umi_rlt_stage1_sft_openpi_pi05

短训练用于检查 ``vla_loss`` 和 ``rlt_loss`` 是否有限，以及保存的模型能否重新加载。FSDP 会同步 FP32 图像 stem 以及其他可训练 VLA 和 RLT 参数的梯度。对比 DCP 与完整 checkpoint 中的模型 tensor，检查各副本是否一致，其中应包含 ``img.stem.weight`` 和 ``img.stem.bias``。

最终的 ``actor/model_state_dict/full_weights.pt`` 同时包含 VLA 和 ``rlt_module`` 权重。应保留完整 checkpoint、解析后的训练配置和原始归一化文件，供 Stage 2 特征模型使用。

确认短训练正常后，使用 ``examples/sft/config/umi_rlt_stage1_sft_openpi_pi05_20k.yaml`` 执行 20,000 个 optimizer step。该配置使用 8 张 GPU，每卡 micro batch size 为 9，global batch size 为 72；每个 optimizer step 中，每个 rank 执行一次前向与反向计算。每个 rank 启动 1 个数据加载 worker，共 8 个。

20k 配置通过 ``actor.fsdp_config.strategy: fsdp`` 和 ``actor.fsdp_config.sharding_strategy: full_shard``，将参数、梯度和 Adam 状态分片到各 rank。原生 ``openpi_rlinf`` 模型的这一 FSDP1 模式要求 ``actor.model.openpi.task: sft``、FP32 参数存储和 ``actor.model.is_lora: false``。配置同时启用 ``actor.fsdp_config.gradient_checkpointing: true`` 和 ``actor.fsdp_config.gradient_checkpointing_use_reentrant: false``，在反向计算时重新计算 Gemma 与 SigLIP 的 activation，以降低 activation 占用的显存。

20k 配置将主干网络的 BF16 计算与 FP32 训练状态分开，沿用源 OpenPI 训练 pipeline 的精度设置：``actor.model.precision: fp32`` 保留 FP32 参数，``actor.model.openpi.compute_dtype: bf16`` 启用 BF16 计算。Gemma、SigLIP encoder 与 projector、RLT transformer 使用 BF16 计算；参数、梯度、Adam 状态和 FSDP 通信保持 FP32。图像 stem、action 与 time projection、Gemma attention logits 和 loss 也使用 FP32。显式指定 ``compute_dtype`` 时必须使用 FP32 参数存储；不指定该字段时沿用模型原有的精度行为。

该配置沿用 OpenPI-UMI 中 ``pi05_umi_32d_80k_95_real_umi_batch_72_v4`` 的 observation 和 action pipeline，并通过 ``actor.model.openpi.umi_v4_image_augmentation: true`` 在训练期间启用 UMI V4 图像增强。按照源环境 Augmax 0.4.1 的实际行为，每张左腕图像以 0.5 的概率执行颜色扰动，亮度参数范围为 ``[-0.3, 0.3]``，对比度为 ``[-0.4, 0.4]``，色相为 ``[-0.1, 0.1]``，饱和度保持不变。评估时保留原始颜色。PyTorch 与 JAX 使用不同的随机数序列，相同 seed 不会产生完全相同的增强图像。

优化器使用 RLT Stage 1 的独立设置：峰值学习率为 ``1e-5``，warmup 为 1,000 步，随后按 cosine 衰减，在第 20,000 步降至 ``2.5e-6``。原 SFT 配置的学习率计划和 EMA 不属于这份 Stage 1 配置；checkpoint 保存当前训练中的 VLA 和 RLT 权重。

在 20k 配置中填写数据集、模型和归一化文件路径，为缓存和输出选择容量充足的目录，然后在 OpenPI 环境中从仓库根目录启动：

.. code:: bash

   export EMBODIED_PATH="$PWD/examples/sft"
   export PYTHONPATH="$PWD"
   export HF_DATASETS_CACHE=/path/to/dataset_cache
   export JAX_PLATFORMS=cpu
   export NVIDIA_TF32_OVERRIDE=0
   python examples/sft/train_vla_sft.py \
       --config-name umi_rlt_stage1_sft_openpi_pi05_20k \
       runner.logger.log_path=/path/to/rlt_runs

该命令将 loader 的 JAX 运算限制在 CPU，并关闭 PyTorch 训练中的 TF32。训练每隔 2,000 个 optimizer step 保存完整 checkpoint，第 20,000 步也会保存。应为数据缓存和全部 10 个 checkpoint 预留磁盘空间；保存间隔由 ``runner.save_interval`` 控制，已保存的 checkpoint 会保留。训练期间观察 ``vla_loss``、``rlt_loss``、梯度范数和学习率。该配置没有启用独立验证；下面的验证配置可以在训练期间检查动作质量和 token 重建效果，训练 loss 本身不能证明这两项结果。

20k 配置通过 ``data.save_openpi_resume_state: true`` 支持在首个数据 epoch 内恢复采样位置。每个 checkpoint 的 ``actor/openpi_resume_state.pt`` 保存各 rank 已读取的样本数和训练 RNG 状态。将 ``runner.resume_dir`` 指向 ``checkpoints/global_step_<N>`` 目录，即可随模型、优化器和 scheduler 一起恢复这些状态。恢复时先重放确定性的 UMI V4 loader 到已保存的位置，再恢复各 rank 的训练 RNG 状态。

恢复时应保持数据集、loader 中的确定性 transform、sampler 设置和 world size 不变。若各 rank 已读取的样本数都能被新的 micro batch size 整除，可以调整 micro batch，并相应调整梯度累积次数以保持目标 global batch size。该恢复格式仅支持首个数据 epoch。改变 batch 分组或计算精度可能改变后续随机数的消耗方式和计算结果，因此不保证调整前后逐位一致。

.. warning::

   官方 OpenPI loader 只有在启用 UMI V4 恢复选项且 checkpoint 包含对应状态文件时，才会恢复采样位置和训练 RNG。若选项未启用或状态文件缺失，``runner.resume_dir`` 仅恢复模型、优化器和 scheduler，数据采样会重新开始。UMI 恢复格式不恢复数据加载 worker 内部随机 transform 的状态。

在保留的 episode 上验证 UMI Stage 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

若要用未参与 Stage 1 更新的示范选择 checkpoint，应在训练开始前按完整 episode 划分数据。重复录制应归入同一侧，必要时按采集来源分层。将两组互不重叠的原始 episode ID 写成 JSON 列表，例如 ``[0, 3, 7]``。从原始基础 checkpoint 开始，并设置 ``runner.resume_dir: null``；继续使用已经训练过这些验证 episode 的模型，会使这次划分失去作用。

使用 ``examples/sft/config/umi_rlt_stage1_sft_openpi_pi05_holdout.yaml``，按上文配置数据集、基础模型和归一化路径。训练和验证路径指向同一个 LeRobot 目录，再提供两组 episode 列表：

.. code:: yaml

   data:
     train_data_paths: /path/to/umi_lerobot_dataset
     val_data_paths: /path/to/umi_lerobot_dataset
     train_episode_ids: /path/to/train_episode_ids.json
     val_episode_ids: /path/to/val_episode_ids.json
     val_frames_per_episode: 36
     val_num_batches: null
     eval_seed: 20260924
     eval_num_steps: 10
     eval_token_shuffle: true
   actor:
     eval_batch_size: 9
   runner:
     resume_dir: null
     eval_at_start: true
     val_check_interval: 1000
     save_interval: 1000
     checkpoint_selection:
       metric: action_mse
       mode: min
       keep_last: 1

Loader 复用完整数据集的 Arrow 缓存，通过 subset 选择帧，不复制图像数据。训练使用所选 episode 的全部帧。验证从每个 episode 均匀抽取 36 帧并包含首尾，再将不同 episode 交错排列，使每个 rank 内的 token 循环移位来自其他 episode；loader 会检查这一条件。60 个验证 episode、8 个 rank、每卡 batch 为 9 时，2,160 帧恰好构成每 rank 30 批。其他规模下不足一个 global batch 的尾部会被丢弃，数量记录在 loader 的选择 metadata 中。设置 ``val_frames_per_episode: null`` 可验证保留 episode 的全部帧；若该顺序无法保证 donor 来自其他 episode，应同时设置 ``eval_token_shuffle: false``。``val_num_batches`` 可限制诊断批次数；保持 null 则遍历完整的已选验证样本。

验证关闭图像增强，使用固定 RNG seed，并在结束后恢复训练 RNG 和各模块模式。指标包括 ``vla_loss``、``rlt_loss``、打乱 token 后的重建 loss 与比例，以及 10 步采样得到的归一化动作 MAE/MSE。动作 mask 排除 padding 维度。重建 loss 降低本身不能证明下游 RL policy 改善。

Runner 在第一次更新前评估初始模型，并选择 ``action_mse`` 最低的 checkpoint。``checkpoints/validation_selection.json`` 记录其步数、分数和路径，``validation_metrics.jsonl`` 按 optimizer step 记录每次验证。保存管理会保留最佳与最近一次 checkpoint；只有新保存成功后，才会清理这个 tracker 已登记的其他 checkpoint。保存过程中还应为额外一个 checkpoint 预留空间。已有 20k 配置的保留规则保持不变。

使用与 20k 训练相同的环境变量启动此配置：

.. code:: bash

   python examples/sft/train_vla_sft.py \
       --config-name umi_rlt_stage1_sft_openpi_pi05_holdout \
       runner.logger.log_path=/path/to/rlt_holdout_runs

从最近保留的 ``global_step_<N>`` 目录恢复时，保持 split 文件、sampler 设置和输出目录一致。恢复位置还会检查 episode ID 和帧顺序签名。已有选择记录的输出目录不能直接重新从零训练；回退到更早 checkpoint 应使用新输出目录。归一化应与基础模型一致；Stage 1 的保留划分并不证明预训练 VLA 从未见过这些录制。用于选择 checkpoint 的数据属于验证集，最终泛化与闭环测试仍需另备数据。

在验证中评估 token 的动作信息
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

保留验证集的配置可以在验证阶段拟合诊断用动作预测头，评估 token 是否提供了超出机器人状态的信息。先使用相同的 OpenPI transforms 和归一化、关闭图像增强，准备固定输入，再启用以下可选配置：

.. code:: yaml

   runner:
     action_probe:
       enabled: true
       input_dir: /path/to/prepared_probe_inputs
       interval: 5000
       at_start: false
       batch_size: 9
       methods: [ridge, mlp]
       cpu_threads: 4
       max_epochs: 100
       patience: 15
       random_seed: 271828
       image_seed: 20260925

准备目录包含观测、状态、动作和帧标识的 NumPy 数组，以及 ``probe_split.json``、``samples.jsonl``、``provenance.json`` 和标记通过的 ``input_quality_check.json``。``rlinf/data/datasets/openpi_rlinf/action_probe.py`` 中的 ``PreparedActionProbeInputs`` 会核对这些文件与训练的数据集、归一化和 episode 划分是否一致。从训练 episode 内部分出预测头的拟合集和调参集；验证 episode 仅用于最终评分，不参与拟合或选参。每个 episode 保留固定帧，保证不同训练步的结果可比较。

所有 actor 使用已加载的模型提取特征，然后由 driver 在 CPU 上拟合 ridge 和三个 seed 的 MLP 预测头。此过程在单节点共享文件系统中同步执行，不额外加载 backbone，也不更新训练模型；结束后恢复 RNG、模块模式并保留训练数据游标。启动前设置 ``OMP_NUM_THREADS``、``MKL_NUM_THREADS`` 和 ``OPENBLAS_NUM_THREADS`` 以限制 BLAS 线程数；``cpu_threads`` 控制拟合期间的 Torch 线程数。

对比包括仅状态、状态加训练后的 token、状态加固定随机 encoder 的 token。打乱 token、均值 token 和跨 episode 图像替换进一步检查模型是否依赖当前样本的信息。报告使用 episode 等权动作误差和配对 episode bootstrap 区间。``eval/rlt_probe/mlp/state_z_vs_state_only/chunk/all/relative_mse_improvement`` 为正，表示加入训练后的 token 降低了误差；对应的 ``relative_mse_improvement_ci95_low``、``relative_mse_improvement_ci95_high`` 表示不确定性。同一命名空间也包含 ridge 和其他对照的结果。数值为比例，不是百分数。离线预测改善不能直接证明闭环 RL 成功。

新增指标与普通验证一起写入 TensorBoard 和 ``validation_metrics.jsonl``；特征和详细报告保存在 ``<log_path>/<experiment_name>/action_probe/global_step_<N>/``。执行间隔必须是 ``val_check_interval`` 的整数倍，最后一步也会执行。``at_start`` 控制训练开始时的验证；单独调用 ``SFTRunner.run_eval()`` 总会执行已启用的 probe。默认仍按 ``action_mse`` 选择 checkpoint；若改用 probe 指标，则每次验证都必须执行 probe，包括已启用的初始验证。

如果保存 checkpoint 后又执行了未保存的更新，恢复时将 ``runner.logger.tensorboard_purge_step`` 设为第一个需要重新计算的步数。TensorBoard 会隐藏旧的后续事件并保留之前的历史；若有意回退 checkpoint 选择记录，应使用新的输出目录。

运行当前 Franka 示例
--------------------

数据：采集 Franka 示范并计算归一化统计
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 需要 LeRobot 格式的 Franka 示范数据，数据目录应直接包含
``data/`` 和 ``meta/``。在控制节点上按照 :doc:`Franka 真机文档
<../embodied/franka>` 的数据采集流程准备机器人和目标位姿，并在采集配置中导出
LeRobot 格式数据：

.. code:: yaml

   env:
     data_collection:
       enabled: True
       export_format: "lerobot"

然后启动采集：

.. code:: bash

   bash examples/embodiment/collect_data.sh realworld_collect_data

采集完成后，将 LeRobot 数据集放到训练节点，并为当前 RLT OpenPI dataconfig
计算归一化统计。``repo_id`` 需要与 Stage 1 / Stage 2 配置中的
``actor.openpi_data.repo_id`` 和 ``rollout.rlt_feature_model.openpi_data.repo_id`` 保持一致：

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_franka_state \
       --repo-id realworld_peg_insertion_rlt_stage1

后续将 Stage 1 配置中的 ``data.train_data_paths`` 指向该 LeRobot 数据集目录。若 checkpoint 未包含 norm stats，可在 ``openpi_data`` 中设置 ``norm_stats_path`` 指向 ``norm_stats.json``。

Stage 1：训练 RLT 特征模型
~~~~~~~~~~~~~~~~~~~~~~~~~~

启动前先修改 Stage 1 配置中的路径：

.. code:: yaml

   data:
     train_data_paths: /path/to/lerobot_dataset

   actor:
     openpi_data:
       repo_id: "realworld_peg_insertion_rlt_stage1"
     model:
       model_path: /path/to/model
       openpi:
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         rlt_prefix_seq_len: 1024

启动 SFT：

.. code:: bash

   bash examples/sft/run_vla_sft.sh realworld_rlt_stage1_sft_openpi_pi05

保存出的检查点目录通常形如：

.. code:: text

   logs/<run-name>/checkpoints/global_step_<step>/actor

Stage 2 中需要将这个 ``actor`` 目录填到 ``rollout.rlt_feature_model.model_path``。
不要把 Stage 1 checkpoint 填到 ``rollout.model.model_path`` 或
``actor.model.model_path``；这两个位置不负责加载 Stage 1 特征模型。

Stage 2：运行 RLT Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改 Stage 2 配置：

.. code:: yaml

   rollout:
     model:
       model_path: null
     rlt_feature_model:
       model_path: /path/to/stage1/checkpoint/actor
       openpi_data:
         repo_id: "realworld_peg_insertion_rlt_stage1"
       openpi:
         config_name: "pi05_franka_state"
         num_images_in_input: 1
         state_indices: []
         rlt_prefix_seq_len: 1024

   cluster:
     node_groups:
       - label: <gpu_node_group>
         node_ranks: <gpu_node_rank>
       - label: <env_node_group>
         node_ranks: <env_node_rank>
         hardware:
           type: <robot_type>
           configs:
             - robot_ip: <robot_ip>

   env:
     train:
       keyboard_reward_wrapper: rlt_policy_switch
       override_cfg:
         target_ee_pose: [<target_x>, <target_y>, <target_z>, <target_roll>, <target_pitch>, <target_yaw>]

在 master node 上启动异步训练：

.. code:: bash

   bash examples/embodiment/run_realworld_async.sh realworld_rlt_stage2_ac_mlp

当前默认键盘模块实现了 RLT 算法中的关键阶段切换：按 ``b`` 进入 Stage 2 actor
控制阶段。其他功能可根据具体任务需求进行定制
（``rlinf/envs/real/wrappers/episode/policy_switch.py``）。

运行 ManiSkill Joint 示例
-------------------------

ManiSkill joint 示例使用 ``PegInsertionSideWideClearance-v1``、RGB 双视角观测、
Panda 前 9 维 qpos 和 8 维 ``pd_joint_delta_pos`` action。每个 Stage 2 action
是 10-step chunk，语言指令固定为 ``insert the peg in the hole``。

数据：准备 joint-control LeRobot 数据
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage 1 和 Stage 2 的 OpenPI feature model 使用
``pi05_rlt_maniskill_joint`` dataconfig。LeRobot 数据集至少需要包含：

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - 含义
   * - ``image``
     - 主视角 RGB 图像。
   * - ``wrist_image``
     - wrist RGB 图像。
   * - ``state``
     - Panda joint proprio，当前示例使用 9 维状态。
   * - ``actions``
     - 8 维 ``pd_joint_delta_pos`` action。
   * - ``task``
     - 语言指令；也可以通过 ``default_prompt`` 固定为 ``insert the peg in the hole``。

可以直接使用参考数据集
`RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ
<https://huggingface.co/datasets/RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ>`__。
它包含 ManiSkill ``PegInsertionSideWideClearance-v1`` 的成功演示，动作空间为
``pd_joint_delta_pos``，并使用上表中的 joint-control LeRobot 字段。

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   huggingface-cli download RLinf/rlt-maniskill-PegInsertionSide-v1-400-succ \
       --repo-type dataset \
       --local-dir ${HF_LEROBOT_HOME}/maniskill_peginsertionside_joint

也可以用当前仓库中的采集脚本重新生成数据。该脚本会先调用 ManiSkill Panda
motion-planning solver 生成成功轨迹，再把 ``pd_joint_pos`` solver action 转成
``pd_joint_delta_pos`` action，并只保存 replay 成功的 episode：

.. code:: bash

   python toolkits/lerobot/collect_maniskill_peg_lerobot_joint.py \
       --repo-id maniskill_peginsertionside_joint \
       --num-episodes 400 \
       --seed 0 \
       --max-attempts 4000 \
       --overwrite

.. note::

   采集脚本位于 ``toolkits/lerobot``，用于 RLT joint 数据准备。
   它依赖 ManiSkill 的 PegInsertionSide Panda motion-planning solver；如果运行时报
   solver import 错误，请先确认当前 ManiSkill 安装包含 motion-planning 示例。

计算归一化统计时，``--config-name`` 和 ``--repo-id`` 要与训练配置保持一致：

.. code:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_rlt_maniskill_joint \
       --repo-id maniskill_peginsertionside_joint

.. warning::

   Stage 1 checkpoint、Stage 2 ``rollout.rlt_feature_model`` 和 OpenPI assets
   必须使用同一套 ``norm_stats.json``。如果 SFT 基座和 Stage 2 加载了不同
   ``repo_id`` 下的 norm stats，VLA reference action 的尺度会发生偏移。可通过 ``openpi_data.norm_stats_path`` 显式指定上述 ``norm_stats.json`` 路径。

Stage 1：联合训练 ManiSkill OpenPI + RLT 特征模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改 ``examples/sft/config/maniskill_rlt_stage1_sft_openpi_pi05.yaml`` 中的数据路径：

.. code:: yaml

   data:
     train_data_paths:
       - dataset_path: /path/to/maniskill_peginsertionside_joint
         weight: 1.0

   actor:
     model:
       model_path: /path/to/pi05_base
       openpi:
         config_name: pi05_rlt_maniskill_joint
       openpi_data:
         repo_id: maniskill_peginsertionside_joint
         default_prompt: insert the peg in the hole
         norm_stats_path: /path/to/maniskill_peginsertionside_joint/norm_stats.json

启动训练：

.. code:: bash

   bash examples/sft/run_vla_sft.sh maniskill_rlt_stage1_sft_openpi_pi05

保存出的 FSDP 检查点通常形如：

.. code:: text

   logs/<run-name>/checkpoints/global_step_<step>/actor

Stage 2 中需要将这个 ``actor`` 目录直接填到
``rollout.rlt_feature_model.model_path``，例如
``/path/to/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_<step>/actor``。

Stage 2：运行 ManiSkill RLT Actor-Critic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改 ``examples/embodiment/config/maniskill_rlt_stage2_ac_mlp.yaml`` 中的
Stage 1 checkpoint：

.. code:: yaml

   env:
     train:
       wrap_obs_mode: rlt_openpi_joint
       rlt_policy_switch:
         enable: True
         task_mode: full_task
         trigger_mode: auto
     eval:
       wrap_obs_mode: rlt_openpi_joint
       rlt_policy_switch:
         enable: True
         task_mode: full_task
         trigger_mode: auto

   rollout:
     rlt_feature_model:
       model_path: /path/to/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_<step>/actor
       openpi_data:
         repo_id: maniskill_peginsertionside_joint
         norm_stats_path: /path/to/maniskill_peginsertionside_joint/norm_stats.json
       openpi:
         config_name: pi05_rlt_maniskill_joint
     expert_model:
       model_path: /path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_<step>/actor
       precision: null
       openpi:
         use_rlt: False

启动训练：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_rlt_stage2_ac_mlp

如需运行 TD3-MLP 变体，在 ``maniskill_rlt_stage2_td3_mlp.yaml`` 中配置相同的
Stage 1 checkpoint，然后执行：

.. code:: bash

   bash examples/embodiment/run_embodiment.sh maniskill_rlt_stage2_td3_mlp

目前该变体只提供 ManiSkill 仿真配置。它保留冻结的 Stage 1 特征和 transition
replay 路径，仅将 Stage 2 策略和更新目标替换为直接 TD3 actor 与 twin-Q critic。

这个配置会启动 actor、rollout 和 ManiSkill env。rollout 侧冻结
``rollout.rlt_feature_model``，只同步和执行 Stage 2 MLP actor。ManiSkill route
在 ``ready_for_online`` 之前执行 VLA ``ref_chunk``；达到
``algorithm.rlt_schedule.warmup_post_collect_updates`` 后，才允许 actor 在自动
critical phase 中接管。

调节 ManiSkill Critical Phase 和 Intervention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ManiSkill 的切换逻辑是仿真专用的。它读取 peg insertion 任务信息来打开
``rlt_switch_flags`` / ``in_critical_phase``；如果启用 expert takeover，它还会在
actor 连续几个 chunk 没有进展时请求 expert 接管。下面的注释只用于文档说明，
不要写进实际 YAML 文件。

.. code:: yaml

   algorithm:
     rlt_schedule:
       enable: True                    # 使用 ManiSkill rollout / update budget。
       warmup_post_collect_updates: 30000  # actor 上线前需要完成的 update 数。
       train_every_transitions: 5       # 每 N 条已记录 transition 增加训练预算。
     actor_weight_schedule:
       enable: True                    # 使用 BC/Q 权重 ramp，而不是固定权重。
       warmup_updates: 20000           # warmup BC/Q 权重保持的 update 数。
       ramp_updates: 50000             # 逐步过渡到 online BC/Q 权重。

   env:
     train:
       rlt_policy_switch:
         task_mode: full_task          # critical phase 前用 VLA，里面用 actor。
         trigger_mode: auto            # 根据任务信息自动计算 critical phase。
         latch_until_done: True        # 一旦进入 critical phase 就保持 actor 控制。
         auto_gate:
           require_grasp: True         # 抓住 peg 后才允许进入。
           require_not_success: True   # 已成功时不再切到 actor。
           near_hole_x_min: -0.16      # 值越大，x 方向进入门控越窄。
           near_hole_yz_margin: 1.5    # 值越大，y/z 进入窗口越宽。
         expert_takeover:
           enable: True                # 只在 train rollout 中启用仿真 expert 干预。
           trigger_mode: stalled_progress  # 连续无进展 chunk 后才接管。
           gate:
             near_hole_x_min: -0.10    # 足够靠近洞口后才检查是否卡住。
             near_hole_yz_margin: 2.0  # y/z 足够接近后才检查是否卡住。
             stuck_chunks_before_takeover: 3  # 接管前需要的无进展 chunk 数。
             min_x_progress: 0.003     # 能重置 stall 计数的前向进展。
             min_yz_progress: 0.0015   # 能重置 stall 计数的对齐进展。
             min_score_progress: 0.002 # 能重置 stall 的综合分数进展。
             progress_yz_weight: 1.0   # 综合进展分数里的 y/z 惩罚权重。

     eval:
       rlt_policy_switch:
         expert_takeover:
           enable: False               # eval 只测 base policy + actor，不用 expert。

可选：启用 ManiSkill Expert Takeover
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ManiSkill expert takeover 默认关闭。需要在 critical phase 中用更强的 SFT expert
替换 actor action，并把 expert action 作为 intervention target 写入 replay 时，
再打开这个路径：

.. code:: yaml

   rollout:
     expert_model:
       model_path: /path/to/rlt_maniskill_joint_pi05_sft/checkpoints/global_step_<step>/actor
       precision: null
       openpi:
         use_rlt: False

   env:
     train:
       rlt_policy_switch:
         expert_takeover:
           enable: True
           trigger_mode: stalled_progress
           gate:
             stuck_chunks_before_takeover: 3
             min_x_progress: 0.003
             min_yz_progress: 0.0015
             min_score_progress: 0.002
             progress_yz_weight: 1.0

也可以把 ``rollout.expert_model.model_path`` 换成你自己训练得更充分的
joint-control SFT checkpoint。expert 的 OpenPI dataconfig 和 norm stats 需要和
Stage 2 数据保持一致。expert 只用于 train rollout；eval rollout 不会执行 expert takeover，评测的是学到的 actor。

Replay Buffer 逻辑
------------------

当 ``loss_type: rlt_ac`` 时，replay buffer 不会把原始图像观测当作 RL 状态存储。
rollout worker 会返回 RLT 特征，learner 侧把这些特征组装成 transition：

.. code:: text

   curr_obs = {z_rl, proprio, ref_chunk}
   action   = 实际发送给环境的 action chunk
   next_obs = {next_z_rl, next_proprio, next_ref_chunk}

这意味着：

- 真机切换前的 step 仍然会用于训练。这些 step 的执行动作是 VLA reference action，transition 也会进入同一个 replay buffer。
- 切换后的 step 执行 actor action，并以相同的 RLT observation 格式存储。
- ManiSkill route 在整 chunk 级别选择 actor action 或 VLA ``ref_chunk``，replay 中的 ``action`` 始终是实际执行的动作。
- ManiSkill learner 会把 rollout chunk 切成 1-sample transition trajectory，再放进 RLinf ``TrajectoryReplayBuffer``。
- ``sample_window_size`` 控制从 replay buffer 近期窗口中采样的范围，不需要和 ``max_steps_per_rollout_epoch`` 一致。
- ``max_steps_per_rollout_epoch`` 控制一次 rollout flush 到训练侧之前收集多少环境 step。

监控指标
--------

指标定义见 :doc:`训练指标 <../../reference/metrics>`。RLT 中比较有用的信号包括：

- Stage 1 SFT：

  - ``vla_loss``：OpenPI 动作预测损失。
  - ``rlt_loss``：RLT token 重建 / 压缩损失。

- Stage 2 actor-critic：

  - ``critic/critic_loss``：Q 函数 TD loss。
  - ``actor/actor_loss``：组合后的 ``-Q + BC`` actor 目标。
  - ``q_pi`` 和 ``q_value_*``：actor 和 critic heads 上的 Q 值。
  - ``actor/bc_loss``、``actor/bc_ref_loss``、``actor/bc_human_loss``：BC 正则相关损失。
  - ``actor/human_mask_ratio``：当前训练 batch 中被标记为 expert 干预的比例（来自 ``trajectory.intervene_flags``）。
  - ``train/replay_buffer/size``：当前 replay transitions 数量。
  - ``env/success_once`` 和 ``env/episode_len``：任务结果指标。
  - ``eval/success_once``：固定 eval reset ids 上的成功率。

  ManiSkill rollout / replay 诊断（由 actor 收到 trajectory 后统计）：

  - ``train/replay/record_transition_rate``：收集到的 step 中被保存为 RLT transition 的比例（来自 ``forward_inputs.record_transition``）。
  - ``train/replay/actor_switch_rate``：收集到的 step 中实际由 actor/student action 控制环境的比例（来自 ``forward_inputs.actor_switch``）。
  - ``train/replay/intervention_requested_rate``：env 请求 expert 接管的比例（来自 ``forward_inputs.intervention_requested``）。
  - ``train/replay/intervention_rate``：route 实际应用 expert action 的比例（来自 ``trajectory.intervene_flags``）。
  - ``train/replay/transition_count``、``train/replay/reward_mean``、``train/replay/reward_positive_rate``、``train/replay/done_rate``：当前 collect step 的 ManiSkill transition-replay 入库统计。

  RLT schedule / learner backlog（ManiSkill ``algorithm.rlt_schedule.enable``）：

  - ``train/rlt/ready_for_online``：learner ``update_step`` 是否已超过 ``warmup_post_collect_updates``。
  - ``train/rlt/actor_updates_run``、``train/rlt/critic_updates_run`` 和 ``train/rlt/pending_update_budget``：本 step 执行的 actor/critic 更新次数，以及剩余 learner backlog。
  - ``train/rlt/updates_to_run``、``train/rlt/should_train`` 和 ``train/rlt/skip_reason``：本 step 计划训练量，以及跳过训练的原因。
  - ``train/rlt/global_transitions_since_train`` 和 ``train/rlt/global_total_transitions_added``：自上次训练 burst 以来的全局 replay 入库计数。

实验结果
--------

在 RLinf 中的 peg_insertion 任务中利用 RL Token 训练结果如下图。

.. raw:: html

   <div style="display: flex; justify-content: center; margin: 20px 0;">
      <div style="flex: 0.5; text-align: center;">
        <img src="https://raw.githubusercontent.com/RLinf/misc/main/pic/RLT_real.png" style="width: 100%;"/>
       <p><em>RLinf peg_insertion 任务中的 RL Token 训练结果</em></p>
     </div>
   </div>

实践建议
--------

- Stage 1 和 Stage 2 的数据配置必须保持一致：``repo_id``、``config_name``、``action_dim``、``proprio_dim``、``ref_num_action_chunks`` 和 ``z_dim`` 都要对齐；如果保留完整 raw state，可使用 ``state_indices: []``。
- ``rollout.rlt_feature_model`` 指向 Stage 1 检查点；``actor.model`` 是 actor-critic worker 会更新的 Stage 2 MLP 策略。
- ``rollout.model`` 是 Stage 2 MLP 在 rollout worker 上的同步副本。Stage 2 从头训练时保持 ``rollout.model.model_path: null``；恢复 Stage 2 训练使用 ``runner.resume_dir``，加载单个 Stage 2 权重文件使用 ``runner.ckpt_path``。
- 不要配置 ``actor.model.model_path`` 来加载 Stage 1；``actor.model`` 只描述 Stage 2 MLP 的输入输出维度和 Q-head 设置。
- Stage 2 MLP 配置直接内联在各个 Stage 2 YAML 的 ``actor.model`` 下，不再使用单独的 model defaults 文件。
- ``keyboard_reward_wrapper: rlt_policy_switch`` 只在需要人工控制关键阶段切换时使用。
- ManiSkill joint 示例使用 ``env.*.rlt_policy_switch``，不要再使用真机的 keyboard wrapper。
- ManiSkill 的 ``proprio`` 来自 OpenPI processed ``observation.state``。如果新建仿真 dataconfig，需要同时检查数据集 ``state``、OpenPI transform 和 Stage 2 ``proprio_dim``。
- Stage 1、Stage 2 和 checkpoint assets 中的 ``norm_stats.json`` 必须来自同一套数据语义和同一个 ``repo_id``。推荐通过 ``openpi_data.norm_stats_path`` 显式指定，避免 Stage 1 checkpoint 未写入 norm stats 时加载失败。
- ``rollout.rlt_feature_model.model_path`` 应指向 Stage 1 FSDP 检查点下的 ``actor`` 目录，例如 ``.../checkpoints/global_step_<step>/actor``。不要把 RLT Stage 1 checkpoint 转成裸 ``model.safetensors`` 再给 Stage 2 使用，因为 RLT token module 保存在完整 wrapper checkpoint 中。
- 添加仿真示例时，可以新建仿真环境配置，保留 ``loss_type: rlt_ac`` 和 ``rollout.rlt_feature_model``，再把真机阶段切换逻辑替换成适合仿真的逻辑。
