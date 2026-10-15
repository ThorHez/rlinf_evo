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

import json
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

from rlinf.utils.logging import get_logger

_PAGE = """<!doctype html>
<html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ARX5 采集预览</title>
<style>
body{margin:24px;background:#131820;color:#edf2f7;font:18px system-ui,sans-serif}
main{max-width:960px;margin:auto}h1{font-size:24px}
.viewer{position:relative}
header{position:absolute;top:12px;left:12px;right:12px;display:flex;gap:20px;
align-items:center;flex-wrap:wrap;padding:10px 14px;border-radius:8px;
background:rgba(8,11,16,.85)}
#episode{font-weight:bold}#health{color:#ffcf70;font-size:15px}
img{display:block;width:100%;min-height:160px;background:#080b10;border-radius:8px}
.stale{opacity:.4}p{font-size:14px;color:#bbc5d2}
</style><main><h1>ARX5 采集预览</h1><div class="viewer">
<img id="frame" class="stale" alt="相机画面"><header>
<span id="episode">EP —</span><span id="status">等待相机画面</span>
<span id="health">连接中</span></header></div>
<p>仅显示采集画面；机械臂操作仍使用实体键盘。</p></main>
<script>
const picture=document.getElementById('frame');
const episode=document.getElementById('episode');
const status=document.getElementById('status');
const health=document.getElementById('health');
let receivedAt=0, frameAge=0, connected=false, currentURL=null;
function freshness(){
  const age=receivedAt ? frameAge+(performance.now()-receivedAt)/1000 : Infinity;
  picture.classList.toggle('stale', !connected || age>1);
  health.textContent=!connected ? '连接断开或等待画面' :
    age>1 ? `画面已过期 · ${age.toFixed(1)} 秒前` : `画面更新于 ${age.toFixed(1)} 秒前`;
}
async function refresh(){
  const controller=new AbortController();
  const timeout=setTimeout(()=>controller.abort(),2000);
  try{
    const response=await fetch('/frame.jpg',{cache:'no-store',signal:controller.signal});
    if(!response.ok) throw new Error('No frame');
    const metadata=JSON.parse(response.headers.get('X-Preview-Metadata'));
    const blob=await response.blob();
    const nextURL=URL.createObjectURL(blob);
    picture.src=nextURL;
    if(currentURL) URL.revokeObjectURL(currentURL);
    currentURL=nextURL;
    await picture.decode();
    episode.textContent=`EP ${metadata.episode}`;
    status.textContent=metadata.status;
    receivedAt=performance.now();
    frameAge=metadata.age_seconds;
    connected=true;
  }catch(error){connected=false;}
  finally{
    clearTimeout(timeout);
    freshness();
    setTimeout(refresh,100);
  }
}
window.addEventListener('pagehide',()=>{
  if(currentURL) URL.revokeObjectURL(currentURL);
});
setInterval(freshness,250);
refresh();
</script></html>""".encode()


@dataclass(frozen=True)
class _Frame:
    image: np.ndarray
    episode: int
    status: str
    timestamp: float


class _PreviewServer(HTTPServer):
    preview: "Arx5CameraPreview"

    def get_request(self) -> tuple[socket.socket, tuple[str, int]]:
        connection, address = super().get_request()
        connection.settimeout(1.0)
        return connection, address


class _PreviewHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        preview = self.server.preview
        try:
            if self.path == "/":
                self._respond(200, "text/html; charset=utf-8", _PAGE)
            elif self.path == "/frame.jpg":
                self._frame(preview)
            else:
                self._respond(404, "text/plain", b"Not found")
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except Exception:
            preview._fail()

    def _frame(self, preview: "Arx5CameraPreview") -> None:
        with preview._lock:
            frame = preview._latest
        if frame is None:
            self._respond(503, "text/plain", b"Waiting for camera")
            return
        image = frame.image
        height, width = image.shape[:2]
        if width > 960:
            image = cv2.resize(image, (960, max(1, round(height * 960 / width))))
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not success:
            raise RuntimeError("Could not encode the ARX camera preview.")
        metadata = json.dumps(
            {
                "episode": frame.episode,
                "status": frame.status,
                "age_seconds": max(0.0, time.monotonic() - frame.timestamp),
            }
        )
        self._respond(200, "image/jpeg", encoded.tobytes(), metadata=metadata)

    def _respond(
        self,
        code: int,
        content_type: str,
        content: bytes,
        *,
        metadata: str | None = None,
    ) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        if metadata is not None:
            self.send_header("X-Preview-Metadata", metadata)
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args) -> None:
        """Keep polling requests out of collection logs."""


class Arx5CameraPreview:
    """Display existing BGR camera frames through a localhost browser page.

    Publishing replaces one retained frame and never waits for HTTP clients.
    JPEG encoding runs in a daemon server thread. A service error stops preview
    updates without raising on the collection thread.
    """

    def __init__(self, port: int) -> None:
        self._logger = get_logger()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: _Frame | None = None
        self._server = _PreviewServer(("127.0.0.1", port), _PreviewHandler)
        self._server.preview = self
        self._server.timeout = 0.2
        self.url = f"http://127.0.0.1:{self._server.server_port}/"
        self._thread = threading.Thread(
            target=self._serve, name="arx5-camera-preview", daemon=True
        )
        try:
            self._thread.start()
        except Exception:
            self._server.server_close()
            raise

    def publish(self, frame: np.ndarray, *, episode: int, status: str) -> None:
        """Retain a BGR uint8 frame that the caller will no longer modify.

        This only replaces memory references; contended or closed updates are
        skipped. Encoding and resizing never modify the supplied array.
        """
        if self._stop.is_set() or not self._lock.acquire(blocking=False):
            return
        try:
            if self._stop.is_set():
                return
            if (
                not isinstance(frame, np.ndarray)
                or frame.ndim != 3
                or frame.shape[2] != 3
                or frame.dtype != np.uint8
                or not frame.shape[0]
                or not frame.shape[1]
            ):
                return
            self._latest = _Frame(frame, episode, status, time.monotonic())
        except Exception:
            self._stop.set()
        finally:
            self._lock.release()

    def _serve(self) -> None:
        try:
            while not self._stop.is_set():
                self._server.handle_request()
        except Exception:
            if not self._stop.is_set():
                self._fail()
        finally:
            self._server.server_close()

    def _fail(self) -> None:
        if not self._stop.is_set():
            self._stop.set()
            self._logger.warning(
                "ARX camera preview stopped after a service error.", exc_info=True
            )

    def close(self) -> None:
        """Stop accepting requests and wait at most 1.5 seconds for the thread."""
        self._stop.set()
        self._server.server_close()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.5)
