# Measure Camera Capture Latency / 测量相机取帧延时

## 中文

用 [measure_camera_latency.py](measure_camera_latency.py) 测量 GoPro 经 HDMI 采集卡到程序拿到画面的近似延时。脚本在接收电脑上显示时间戳，并给收到的画面添加接收时间；保存图片后，读取画面中的两个数字并计算差值。

### 启动测量

在连接采集卡、可以打开桌面窗口的电脑上运行。Python 环境需要支持窗口显示的 OpenCV 和 NumPy；`opencv-python-headless` 不能用于交互测量。在仓库根目录执行：

```bash
python -m pip install opencv-python numpy
python toolkits/realworld_check/measure_camera_latency.py --device 0
```

程序打开 `Timer` 和 `Capture` 窗口。将 GoPro 对准 `Timer` 中 `SOURCE` 下方的毫秒数字，调整位置和焦点，让数字在 `Capture` 中清晰可读。两个时间使用同一进程的单调时钟，以相同起点计时。

需要指定 Linux 设备和采集格式时，可以运行：

```bash
python toolkits/realworld_check/measure_camera_latency.py \
  --device /dev/video0 --backend v4l2 --width 1920 --height 1080 --fps 30
```

`--device` 接受设备编号或设备路径，默认是 `0`。`--backend` 默认为 `auto`，也可以选择 `v4l2`、`dshow`、`msmf` 或 `avfoundation`。`--width`、`--height`、`--fps` 和四字符的 `--fourcc` 用于请求采集格式，未指定时使用驱动默认值，实际支持情况取决于设备和驱动。测量时尽量使用实际应用的采集格式。

### 保存并读取结果

等待画面稳定后，在窗口获得键盘焦点时按 `S` 或空格保存样本，按 `Q` 或 `Esc` 退出。每个样本包含 PNG 图片和 `samples.csv` 中的一条接收时间记录，默认保存到 `camera_latency_results`；使用 `--output-dir` 可以更改目录。

图片保留原始画面，在外加边框中标注 `RECEIVED ... ms`。读取 GoPro 拍到的 `SOURCE` 数字，在 CSV 中补填 `source_ms`，按以下公式计算后填入 `latency_ms`：

```text
近似延时（ms）= received_ms - source_ms
例如：123638.0 - 123456.0 = 182.0 ms
```

`received_ms` 在读取帧成功返回后立即记录，与按键保存时间无关。文字识别和差值计算需要人工完成。保存多个样本后，可以统计延时的中位数、P95 和最大值。

### 理解测量范围

这个差值包含屏幕渲染和刷新误差，以及相机、HDMI、采集卡、驱动和取帧环节的延时，不能视为精确的曝光到接收延时。若数字因刷新或曝光出现重影，应舍弃该样本。

脚本在后台持续取帧，预览使用最新收到的帧，因此测得的是本工具取帧方式下的延时。已有应用内部的排队和 policy 取帧等待需要在实际应用中另外测量。

可以先检查图像生成、导出和接收时间的保存：

```bash
python toolkits/realworld_check/measure_camera_latency.py --self-test
```

该检查使用合成数据，可以在没有相机和桌面窗口的环境运行；实际 HDMI 链路仍需在设备上验证。

## English

Use [measure_camera_latency.py](measure_camera_latency.py) to estimate latency from a GoPro through an HDMI capture card to frame receipt in this program. The script displays a timer on the receiving computer and adds a receipt timestamp to captured images. Read both numbers from a saved image to calculate their difference.

### Start a Measurement

Run on the computer connected to the capture card, with a desktop session and Python containing NumPy and an OpenCV build with GUI support. Interactive measurement requires `opencv-python`, rather than `opencv-python-headless`. From the repository root, run:

```bash
python -m pip install opencv-python numpy
python toolkits/realworld_check/measure_camera_latency.py --device 0
```

The program opens `Timer` and `Capture` windows. Point the GoPro at the millisecond digits below `SOURCE` in `Timer`, then adjust framing and focus until the digits are readable in `Capture`. Both timestamps use the same process's monotonic clock and the same origin.

To select a Linux device and request a capture format, run:

```bash
python toolkits/realworld_check/measure_camera_latency.py \
  --device /dev/video0 --backend v4l2 --width 1920 --height 1080 --fps 30
```

`--device` accepts a device index or path and defaults to `0`. `--backend` defaults to `auto`; other choices are `v4l2`, `dshow`, `msmf`, and `avfoundation`. Use `--width`, `--height`, `--fps`, and a four-character `--fourcc` to request a format supported by your device and driver; omitted settings retain driver defaults. Match the format used by your application when possible.

### Save and Read Results

Once capture has settled, focus either window and press `S` or Space to save a sample. Press `Q` or Esc to exit. Each sample produces a PNG and a receipt timestamp row in `samples.csv` under `camera_latency_results`. Change the destination with `--output-dir`.

The saved image preserves the captured pixels and adds a border labeled `RECEIVED ... ms`. Read the filmed `SOURCE` value, enter it in the CSV's `source_ms` field, and fill `latency_ms` using:

```text
Approximate latency (ms) = received_ms - source_ms
Example: 123638.0 - 123456.0 = 182.0 ms
```

`received_ms` is recorded immediately after a successful frame read, independently of when you press Save. Reading the digits and calculating differences are manual steps. Collect multiple samples to calculate the median, P95, and maximum latency.

### Interpret the Measurement

The difference includes screen rendering and refresh error, plus camera, HDMI, capture card, driver, and frame acquisition latency. It is an approximation of exposure-to-receipt latency. Discard samples with digits blurred or mixed by exposure and display refresh.

The script continuously reads frames in the background and previews the latest received frame. Its result describes acquisition in this tool; measure existing application queues and waits before a policy consumes a frame inside that application.

To check image generation, export, and receipt-timestamp preservation, run:

```bash
python toolkits/realworld_check/measure_camera_latency.py --self-test
```

This check uses synthetic data and runs without a camera or desktop window. Validate the actual HDMI capture chain on your hardware.
