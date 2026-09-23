# 本机 RLinf RL Token 环境安装文档

> 当前更新：ARX SDK/UVC 适配及 Stage 2 的 AC、TD3 训练和配套评估配置已经加入本分支；SDK 服务使用 `/media/lenovo/data/miniconda3/envs/arx-py310`。下文原始安装记录保留供追溯，最新启动、标定、接管与测试边界见 [ARX Stage 2 指南](../docs/source-zh/rst_source/examples/embodied/arx5_rlt.rst)。真机与完整 checkpoint 验收仍需完成。

这份文档说明如何使用和重建本机 RLinf RLT 训练环境。安装对象是 `codex/umi-rlt-sft-resume` 分支，目标为 Stage 2 在线强化学习。当前 ARX/UVC 适配和 AC／TD3 配置已加入，安装入口见第 6 节；实现前的环境核查保留在 [UMI + ARX 历史记录](UMI_ARX_INTEGRATION.md)。

## 1. 使用已安装环境

在新终端执行：

```bash
cd /home/lenovo/mason/code/rlinf_evo
source local_setup/activate.sh
which python
python local_setup/verify_environment.py
```

Python 应指向 `/home/lenovo/mason/code/rlinf_evo/.venv/bin/python`。激活脚本同时将 JAX 限制到 CPU，设置本环境的 OpenPI 资源目录，并隔离登录 shell 中继承的 ROS/Python 3.10 搜索路径；模型训练与推理使用 CUDA PyTorch。退出 Python 虚拟环境可执行 `deactivate`；若要切换到其他项目，建议使用新终端，避免继续沿用这些环境变量。

当前 ARX SDK 服务使用 `/media/lenovo/data/miniconda3/envs/arx-py310/bin/python`，通过本分支 ZMQ 接口与 RLinf 交换数据。SDK 的本地二进制是 CPython 3.10 扩展，不能直接导入 Python 3.11 训练环境。早期核查的 `/media/lenovo/data/miniconda3/envs/umi-arx`（Python 3.10.21）保留在历史记录中。

## 2. 本机与关键版本

安装日期：2026-09-23。仓库提交：`173594b8`。

| 项目 | 配置 |
| --- | --- |
| 操作系统 | Ubuntu 22.04.5 LTS，x86_64 |
| GPU | NVIDIA GeForce RTX 4090，24 GB |
| 驱动 | 595.84；`nvidia-smi` 显示最高支持 CUDA 13.2 |
| 系统 CUDA 编译器 | `nvcc` 11.5 |
| 新虚拟环境 | 仓库内 `.venv`，Python 3.11.16 |
| PyTorch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| TorchCodec | 0.11.1，CPU 视频解码 |
| RLinf / Ray | 0.4.0（editable 安装本仓库）/ 2.58.0 |
| OpenPI | `rlinf-openpi==0.1.1` |
| Transformers | `transformers==4.53.2` + `rlinf-transformer-openpi==4.53.2` + OpenPI overlay |
| LeRobot | 0.1.0，Git commit `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |
| JAX / jaxlib / Flax | 0.5.3 / 0.5.3 / 0.10.2 |
| Orbax / NumPy | 0.11.13 / 1.26.4 |
| PEFT / Diffusers | 0.17.1 / 0.33.1 |
| FFmpeg 命令 | imageio-ffmpeg 提供的 7.0.2，链接至 `.venv/bin/ffmpeg` |

完整已安装版本记录在 [installed.freeze](installed.freeze)，关键兼容约束记录在 [runtime-overrides.constraints](runtime-overrides.constraints)。虚拟环境约占 9.5 GB；uv 下载缓存额外占用磁盘。

这里明确选择 CUDA 12.8 wheel，与本机驱动兼容。PyTorch 2.11 的 PyPI 默认 CUDA 构建为 13.0，12.8 wheel 通过 PyTorch 专用索引提供，因此安装时固定 `cu128`。[PyTorch 发布说明](https://github.com/pytorch/pytorch/releases/tag/v2.11.0)

本机已有 FFmpeg 共享库（如 `libavcodec.so.58`、`libavformat.so.58`、`libavutil.so.56`）及 GL/EGL 库，视频自检已通过。`imageio-ffmpeg` 提供的静态命令不能替代 TorchCodec 所需的 FFmpeg 共享库；迁移到其他机器时，使用 `--no-root` 前需要确认这些系统依赖。

系统 `nvcc` 与 PyTorch wheel 携带的 CUDA runtime 是不同组件。此次使用预编译 wheel，保留系统编译器；所验证的 RLT 原生 PyTorch 路径使用 PyTorch attention 运算，未安装 FlashAttention、Apex 或 NATTEN。若未来需要编译 CUDA 扩展，应另行配置与所选 PyTorch 相匹配的工具链。

## 3. 原始安装过程（历史记录）

以下命令记录最初在本机使用 uv 创建独立环境的过程。已有环境直接使用第 1 节；当前 ARX 安装入口见第 6 节。复现原始安装时，在仓库根目录按顺序执行；安装过程需要访问 GitHub、PyPI、PyTorch wheel 索引和 Google Cloud Storage。

先安装 RLinf embodied 基础依赖：

```bash
cd /home/lenovo/mason/code/rlinf_evo
UV_TORCH_BACKEND=cu128 UV_HTTP_TIMEOUT=120 \
  bash requirements/install.sh embodied \
  --env dummy --python 3.11.16 --venv .venv \
  --no-root --install-rlinf
```

此分支的 `--model openpi` 不接受 `--env dummy`。这里用不带 `--model` 的 dummy 目标建立基础环境，再安装真机模型依赖。官方 RLT 文档中的 `--model openpi --env maniskill_libero` 面向仿真；本机的 ARX 控制组件由已有环境提供。[RLinf RLT 安装说明](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/rlt.html)

安装 OpenPI、固定版本的 LeRobot，以及 ZMQ/websocket 通信和 embodied 公共依赖：

```bash
UV_LINK_MODE=copy UV_HTTP_TIMEOUT=120 \
  uv pip install --python .venv/bin/python \
  --override local_setup/runtime-overrides.constraints \
  --torch-backend cu128 \
  'rlinf-openpi==0.1.1' \
  -r requirements/embodied/models/openpi.txt \
  -r requirements/embodied/envs/common.txt \
  'transformers==4.53.2' 'peft==0.17.1' \
  'diffusers==0.33.1' 'ml-dtypes==0.5.4' \
  'chex==0.1.90' 'rerun-sdk==0.22.1' \
  pyzmq websockets msgpack 'protobuf>=6.33.5,<7'
```

这里使用 `--override`，因为 OpenPI 的 wheel 元数据固定 Torch 2.7.1，而该 RLinf 分支在 `pyproject.toml` 中选择 Torch 2.11.0。普通 constraint 无法解决这两个精确版本要求。LeRobot 固定到源码 commit，以保留 OpenPI 使用的 `lerobot.common.datasets` 接口。JAX override 选择 CPU 基础包；该环境面向 PyTorch RLT，JAX 用于 OpenPI 数据转换与资源处理。

OpenPI 的 Transformers fork 与标准 distribution 共用 `transformers/` 目录，需要在标准包安装后重铺 fork，再应用仓库安装脚本要求的 overlay：

```bash
UV_LINK_MODE=copy uv pip install --python .venv/bin/python \
  --reinstall --no-deps 'rlinf-transformer-openpi==4.53.2'
.venv/bin/python local_setup/patch_openpi.py
```

`copy` 模式及补丁脚本的 unlink 操作可避免改写 uv 缓存的共享文件。今后升级、卸载或重装任一 Transformers distribution 后，需要重新执行这一步并运行自检。

下载并校验分词器，然后给新环境提供 FFmpeg 命令：

```bash
mkdir -p .venv/assets/openpi/big_vision
curl --fail --location --retry 3 \
  https://storage.googleapis.com/big_vision/paligemma_tokenizer.model \
  -o .venv/assets/openpi/big_vision/paligemma_tokenizer.model
sha256sum -c local_setup/tokenizer.sha256

.venv/bin/python - <<'PY'
from pathlib import Path
import imageio_ffmpeg

link = Path('.venv/bin/ffmpeg')
if not link.exists() and not link.is_symlink():
    link.symlink_to(imageio_ffmpeg.get_ffmpeg_exe())
PY
source local_setup/activate.sh
```

实际 tokenizer 查找路径为 `OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model`。只把文件放在缓存根目录会导致 OpenPI 再次联网下载。

## 4. 安装验证与记录

执行独立自检：

```bash
source local_setup/activate.sh
python local_setup/verify_environment.py
uv pip check --python .venv/bin/python
```

自检使用合成输入和小模型，不加载真实 Stage 1 checkpoint，也不连接 ARX 或相机。它检查模型与 worker 导入、CUDA 运算、RLT 特征和 actor-critic 的前向/反向、tokenizer、视频编解码及独立 Ray 任务。`--skip-ray` 可用于只检查 Python/模型依赖；完整验证应保留 Ray 检查。

`uv pip check` 当前报告一条已知元数据冲突：`rlinf-openpi==0.1.1` 声明要求 `torch==2.7.1`，实际使用 `2.11.0+cu128`。这是上述显式 override 的结果，原始包元数据仍保留。判断可用性需结合实际自检；若出现其他冲突，应排查，不能将其归入这一例外。

2026-09-23 本机实测的 6 项检查全部通过：版本读取、Stage 2/SFT 模块导入及 UMI 配置、CUDA/RLT 前向反向、tokenizer、视频编解码、Ray 远程任务。RLT 测试输出 action shape 为 `(2, 160)`、双 Q shape 为 `(2, 2)`，PyTorch 缓存显存峰值为 28 MiB。该数值只代表小模型测试，不代表完整 Pi0.5 所需显存。Ray worker 使用本环境 Python，并在检查结束后关闭临时实例。

原始自检输出见 [verification.log](../logs/environment-install/verification.log)，当前依赖检查共扫描 284 个包，仅报告上文所述的 Torch 声明冲突。

安装日志保存在 `logs/environment-install/`。已保存 Stage 2 示例的 Hydra 配置解析结果 `stage2-config.yaml`；解析配置仅验证入口和依赖，不表示 Franka 示例已适用于 ARX。包版本快照在安装后由 `uv pip freeze --python .venv/bin/python` 生成；其中 editable 路径针对本机。

## 5. 安装完成时的接入缺口（历史记录）

最初完成训练软件环境配置并核查已有 ARX 控制环境时，记录了以下接入缺口。软件适配和本机配置现已完成，模型资产与真机验收仍需按当前指南确认：

- 与 UMI V4 对齐且包含 RLT 模块的完整 Stage 1 checkpoint、配置和归一化统计。
- RLinf 与现有 ARX 控制接口的 adapter，包括观测、动作、reward、reset、人工接管和 transition 回传。
- 本机单卡 actor/rollout placement 及实际任务参数。现有真机示例为双节点 Franka，不能直接执行它来控制 ARX。
- 为冻结特征模型和 Stage 2 训练留出显存。安装时已有 OpenPI 服务占用约 18 GB；本次保留该服务，验证只使用小规模张量，因此尚未证明完整模型可与其同时驻留。

具体接口、UMI 维度和已有控制端验证命令见 [UMI_ARX_INTEGRATION.md](UMI_ARX_INTEGRATION.md)。

## 6. 新增 ARX 适配的安装入口

新建完整模型环境可使用：

```bash
bash requirements/install.sh embodied --model openpi --env arx5
```

在当前已有模型环境中，仅补客户端依赖：

```bash
.venv/bin/python -m pip install -r requirements/embodied/envs/arx5.txt
```

这些命令只安装训练端和 UVC/ZMQ 客户端依赖；SDK 控制服务继续使用独立 Python 3.10。启动命令、必填的原始统计量和时序参数、人工复位与拖动操作、恢复训练和固定权重评估均见上面的新指南。配置解析与 fake 硬件测试不会启动真实机械臂或相机。

选择 Stage 2 算法时，使用配套的配置和 checkpoint：

| 算法 | 训练配置 | 固定权重评估配置 |
| --- | --- | --- |
| AC | `realworld_arx5_rlt_stage2_ac_mlp` | `realworld_arx5_rlt_stage2_eval` |
| 当前 RLT TD3 变体 | `realworld_arx5_rlt_stage2_td3_mlp` | `realworld_arx5_rlt_stage2_td3_eval` |

两者通过同样的 `run_embodiment.sh` 和 `evaluations/run_eval.sh` 入口运行，复用同一 Stage 1 checkpoint、EEF 数据处理、拖动接管和 replay，保持同步推理与 RTC 关闭。恢复训练使用原算法的 `runner.resume_dir`；评估的 `RLT_STAGE2_WEIGHTS` 指向对应算法的 `actor/model_state_dict/full_weights.pt`，AC 与 TD3 权重不能互换。TD3 的 `actor_noise_sigma=0.1` 在残差坐标中采样后再乘 EEF 各维尺度，不是直接向米制动作添加 0.1 的噪声；具体尺度和命令见新指南。
