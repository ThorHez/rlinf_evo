# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure camera-to-program latency by filming a clock on the receiving computer.

Run ``python measure_camera_latency.py --device 0`` on a desktop with OpenCV
and NumPy. Aim the camera at Timer, then press S or Space to save Capture.
Subtract the photographed SOURCE value from the RECEIVED value above the image.
Both values use milliseconds since this process started, on the same clock.
See camera_latency.md for setup, measurement limits, and a headless self-test.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import queue
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger("camera_latency")
TIMER_WINDOW = "Timer"
CAPTURE_WINDOW = "Capture"


@dataclass(frozen=True)
class ReceivedFrame:
    """A decoded frame and its read-completion time in session milliseconds."""

    image: np.ndarray
    received_ms: float


def _draw_text(image: np.ndarray, text: str, y: int, scale: float) -> None:
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0]
    scale *= min(1.0, (image.shape[1] - 40) / max(width, 1))
    cv2.putText(
        image,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def render_timer(source_ms: float) -> np.ndarray:
    """Return a clock target whose time is supplied by the session clock."""
    image = np.zeros((300, 1000, 3), dtype=np.uint8)
    _draw_text(image, "SOURCE (ms) - point the camera at this window", 50, 0.8)
    _draw_text(image, f"{source_ms:012.1f}", 175, 3.6)
    _draw_text(image, "S / Space: save sample     Q / Esc: quit", 260, 0.8)
    return image


def annotate_frame(frame: np.ndarray, received_ms: float) -> np.ndarray:
    """Add a receipt-time header without covering or modifying camera pixels."""
    image = cv2.copyMakeBorder(
        frame,
        100,
        0,
        0,
        max(0, 800 - frame.shape[1]),
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    _draw_text(image, f"RECEIVED: {received_ms:012.1f} ms", 40, 1.0)
    _draw_text(image, "Latency ~= RECEIVED - photographed SOURCE", 80, 0.75)
    return image


def save_sample(output_dir: Path, frame: np.ndarray, received_ms: float) -> Path:
    """Save an annotated PNG and append its receipt time to samples.csv.

    Args:
        output_dir: Directory created as needed for samples from any session.
        frame: Original BGR camera image, which is left unchanged.
        received_ms: Time recorded at read completion, not at save time.

    Returns:
        Path to the PNG. The CSV leaves source_ms and latency_ms blank for
        manual transcription and subtraction of the photographed clock value.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"sample_{uuid.uuid4().hex}.png"
    if not cv2.imwrite(str(path), annotate_frame(frame, received_ms)):
        raise OSError(f"Could not write sample image: {path}")
    csv_path = output_dir / "samples.csv"
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if needs_header:
            writer.writerow(["image", "received_ms", "source_ms", "latency_ms"])
        writer.writerow([path.name, f"{received_ms:.1f}", "", ""])
    return path


def _publish(
    mailbox: queue.Queue[ReceivedFrame | Exception], item: ReceivedFrame | Exception
) -> None:
    # Only the capture thread publishes. Replace stale frames instead of queuing.
    try:
        mailbox.get_nowait()
    except queue.Empty:
        pass
    mailbox.put_nowait(item)


def _capture_frames(
    args: argparse.Namespace,
    origin_ns: int,
    stop: threading.Event,
    mailbox: queue.Queue[ReceivedFrame | Exception],
) -> None:
    capture = None
    try:
        backend = {
            "auto": cv2.CAP_ANY,
            "v4l2": cv2.CAP_V4L2,
            "dshow": cv2.CAP_DSHOW,
            "msmf": cv2.CAP_MSMF,
            "avfoundation": cv2.CAP_AVFOUNDATION,
        }[args.backend]
        device = int(args.device) if args.device.isdecimal() else args.device
        capture = cv2.VideoCapture(device, backend)
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open capture device {args.device!r}.")
        settings = (
            [(cv2.CAP_PROP_FOURCC, "fourcc", cv2.VideoWriter_fourcc(*args.fourcc))]
            if args.fourcc
            else []
        )
        settings.extend(
            [
                (cv2.CAP_PROP_FRAME_WIDTH, "width", args.width),
                (cv2.CAP_PROP_FRAME_HEIGHT, "height", args.height),
                (cv2.CAP_PROP_FPS, "fps", args.fps),
            ]
        )
        for prop, name, value in settings:
            if value is not None and not capture.set(prop, value):
                LOGGER.warning("Backend rejected requested %s=%s", name, value)
        LOGGER.info(
            "Capture: backend=%s, %.0fx%.0f, fps=%.3f",
            capture.getBackendName(),
            capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            capture.get(cv2.CAP_PROP_FPS),
        )
        while not stop.is_set():
            ok, frame = capture.read()
            received_ns = time.perf_counter_ns()
            if not ok or frame is None:
                raise RuntimeError(
                    "Capture read failed. Check the HDMI signal and device."
                )
            _publish(mailbox, ReceivedFrame(frame, (received_ns - origin_ns) / 1e6))
    except Exception as exc:
        _publish(mailbox, exc)
    finally:
        if capture is not None:
            capture.release()


def _run(args: argparse.Namespace) -> None:
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "Run on the receiving computer's graphical desktop; no display is available."
        )
    stop = threading.Event()
    mailbox: queue.Queue[ReceivedFrame | Exception] = queue.Queue(maxsize=1)
    origin_ns = time.perf_counter_ns()
    reader = threading.Thread(
        target=_capture_frames,
        args=(args, origin_ns, stop, mailbox),
        daemon=True,
    )
    sample = None
    pending_save: Future[Path] | None = None
    saver = ThreadPoolExecutor(max_workers=1)
    reader.start()
    try:
        cv2.namedWindow(TIMER_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TIMER_WINDOW, 1000, 300)
        cv2.namedWindow(CAPTURE_WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CAPTURE_WINDOW, 960, 640)
        placeholder = np.zeros((300, 800, 3), dtype=np.uint8)
        _draw_text(placeholder, "Waiting for the capture device...", 150, 1.0)
        cv2.imshow(CAPTURE_WINDOW, placeholder)
        LOGGER.info("Aim at Timer. Press S / Space to save, Q / Esc to quit.")
        LOGGER.info("Samples: %s", args.output_dir.resolve())
        while True:
            if pending_save is not None and pending_save.done():
                LOGGER.info("Saved %s", pending_save.result())
                pending_save = None
            source_ms = (time.perf_counter_ns() - origin_ns) / 1e6
            cv2.imshow(TIMER_WINDOW, render_timer(source_ms))
            try:
                item = mailbox.get_nowait()
            except queue.Empty:
                item = None
            if isinstance(item, Exception):
                raise item
            if item is not None:
                sample = item
                cv2.imshow(
                    CAPTURE_WINDOW, annotate_frame(sample.image, sample.received_ms)
                )
            last_received_ms = sample.received_ms if sample else 0.0
            if source_ms - last_received_ms > 10000:
                raise RuntimeError(
                    "No frame received for 10 seconds. Check the capture device."
                )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if any(
                cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) == 0
                for name in (TIMER_WINDOW, CAPTURE_WINDOW)
            ):
                break
            if key in (ord("s"), ord("S"), 32) and sample is not None:
                if pending_save is None:
                    # Encoding and disk I/O must not freeze the filmed timer.
                    pending_save = saver.submit(
                        save_sample, args.output_dir, sample.image, sample.received_ms
                    )
                else:
                    LOGGER.info(
                        "Previous sample is still being saved; try again shortly."
                    )
    finally:
        stop.set()
        reader.join(timeout=2.0)
        cv2.destroyAllWindows()
        saver.shutdown(wait=True)
        if pending_save is not None:
            LOGGER.info("Saved %s", pending_save.result())
        if reader.is_alive():
            # TODO(agent): Some capture drivers cannot interrupt a blocked read.
            # The daemon exits with this standalone process; do not release the
            # capture concurrently with its owner's read call.
            LOGGER.warning(
                "Capture driver is still blocked; process exit will close the device."
            )


def _self_test() -> None:
    source_ms, received_ms = 123456.0, 123638.0
    frame = render_timer(source_ms)
    original = frame.copy()
    with tempfile.TemporaryDirectory(prefix="camera_latency_") as directory:
        root = Path(directory)
        first = save_sample(root, frame, received_ms)
        second = save_sample(root, frame, received_ms + 10)
        saved = cv2.imread(str(first))
        if not np.array_equal(saved[100:, : frame.shape[1]], original):
            raise RuntimeError("Self-test failed: camera pixels changed during export.")
        if not np.array_equal(frame, original) or first == second:
            raise RuntimeError(
                "Self-test failed: source mutation or duplicate filename."
            )
        with (root / "samples.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 2 or float(rows[0]["received_ms"]) - source_ms != 182.0:
            raise RuntimeError("Self-test failed: receipt timestamp was not preserved.")
    LOGGER.info(
        "Self-test passed: rendering, PNG export, CSV timestamps, and unique samples."
    )


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def main() -> int:
    """Run the desktop measurement or a headless export self-test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device", default="0", help="Camera index or device path, e.g. /dev/video2"
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "v4l2", "dshow", "msmf", "avfoundation"],
    )
    parser.add_argument("--width", type=_positive_int, help="Requested capture width")
    parser.add_argument("--height", type=_positive_int, help="Requested capture height")
    parser.add_argument(
        "--fps", type=_positive_float, help="Requested capture frames per second"
    )
    parser.add_argument("--fourcc", help="Requested capture format, e.g. MJPG or YUYV")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("camera_latency_results")
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Test rendering/export without a camera or display",
    )
    args = parser.parse_args()
    if args.fourcc is not None and (len(args.fourcc) != 4 or not args.fourcc.isascii()):
        parser.error("--fourcc must contain exactly four ASCII characters")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        if args.self_test:
            _self_test()
        else:
            _run(args)
    except KeyboardInterrupt:
        return 0
    except (RuntimeError, OSError, cv2.error) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
