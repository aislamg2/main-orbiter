"""Live camera preview served as a web page (MJPEG), viewable from any browser.

Works over SSH: open http://<pi-ip>:<port>/ on your laptop. The page shows the
video with a center crosshair and the live pan/tilt angles drawn on top.
"""

from __future__ import annotations

import io
import json
import socket
import threading
from http import server
from socketserver import ThreadingMixIn

from .config import CameraConfig

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orbiter camera</title>
<style>
  body { margin:0; background:#111; color:#eee; font:14px/1.4 system-ui,sans-serif; }
  .wrap { position:relative; max-width:100vw; max-height:100vh; display:inline-block; }
  img { display:block; max-width:100vw; max-height:100vh; }
  .hud { position:absolute; top:10px; left:10px; background:rgba(0,0,0,.6); padding:8px 12px;
         border-radius:6px; font-family:ui-monospace,Menlo,monospace; white-space:pre; }
  .warn { color:#ffb347; }
  .cross::before, .cross::after { content:""; position:absolute; background:rgba(0,255,120,.7); }
  .cross::before { left:50%; top:calc(50% - 15px); width:1px; height:30px; }
  .cross::after  { top:50%; left:calc(50% - 15px); height:1px; width:30px; }
</style></head>
<body><div class="wrap cross"><img src="/stream.mjpg" alt="camera"><div class="hud" id="hud">connecting...</div></div>
<script>
async function tick() {
  try {
    const s = await (await fetch('/status.json', {cache: 'no-store'})).json();
    const lines = Object.entries(s).map(([k, v]) =>
      k.padEnd(6) + ' ' + (typeof v === 'number' ? v.toFixed(1) : v));
    document.getElementById('hud').textContent = lines.join('\\n');
  } catch (e) { document.getElementById('hud').textContent = 'status unavailable'; }
}
setInterval(tick, 150); tick();
</script></body></html>
"""


class _Frames(io.BufferedIOBase):
    """Receives JPEG frames from the encoder and hands the latest one to clients."""

    def __init__(self):
        self.frame: bytes | None = None
        self.cond = threading.Condition()

    def write(self, buf):
        with self.cond:
            self.frame = bytes(buf)
            self.cond.notify_all()
        return len(buf)


class _HTTPServer(ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def local_ip() -> str:
    """Best guess at the Pi's LAN address (no packets are actually sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname()


def camera_transform(cam: CameraConfig):
    """libcamera Transform for the configured mounting (180 = flip both axes)."""
    from libcamera import Transform

    return Transform(hflip=1, vflip=1) if cam.rotate_180 else Transform()


class CameraStream:
    def __init__(self, cam: CameraConfig, status=None, port: int | None = None):
        self.cam = cam
        self.port = port or cam.preview_port
        self.status = status or (lambda: {})
        self.frames = _Frames()
        self.picam2 = None
        self.httpd = None
        self.stopping = False

    @property
    def url(self) -> str:
        return f"http://{local_ip()}:{self.port}/"

    def start(self) -> str:
        self._start_camera()
        self._start_http()
        return self.url

    def _start_camera(self):
        from picamera2 import Picamera2
        from picamera2.encoders import JpegEncoder
        from picamera2.outputs import FileOutput

        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (self.cam.preview_width, self.cam.preview_height)},
            transform=camera_transform(self.cam),
        )
        self.picam2.configure(config)
        # Software JPEG encoder: the Pi 5 has no hardware MJPEG encoder.
        self.picam2.start_recording(JpegEncoder(q=self.cam.preview_quality), FileOutput(self.frames))

    def _start_http(self):
        self.httpd = _HTTPServer(("", self.port), _make_handler(self))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self):
        self.stopping = True
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.picam2:
            self.picam2.stop_recording()
            self.picam2.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def _make_handler(stream: CameraStream):
    frames = stream.frames

    class Handler(server.BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal UI clean
            pass

        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif path == "/status.json":
                self._send(json.dumps(stream.status()).encode(), "application/json")
            elif path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()
                try:
                    while not stream.stopping:
                        with frames.cond:
                            frames.cond.wait(timeout=1)
                            frame = frames.frame
                        if frame is None:
                            continue
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # browser tab closed
            else:
                self.send_error(404)

    return Handler
