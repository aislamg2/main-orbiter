"""Live preview for aiming the mount and choosing a crop. No model required.

The page shows three things:

  * the camera view with a green crosshair at the frame centre,
  * a red outline marking the square the classifier would take at `crop`, and
  * a thumbnail of the exact 224 x 224 array that square becomes - cropped,
    greyscaled and nearest-neighbour resized, the real model input.

The thumbnail is the point. The outline shows where the model looks; the
thumbnail shows what survives the downscale, which is usually less than the
outline suggests. This builds that input with the same `preprocess()` the
classifier uses, but never opens the Hailo, so you can tune framing, lighting
and crop without the model in the way.

The raw stream is pinned to the full sensor array. A 720p video mode left to
itself reads out a *cropped* region of the sensor, so its field of view is
narrower than what `measure-k` and `survey` capture - framing on it would lie
to you.

    python3 scripts/preview.py                # crop from [model] crop
    python3 scripts/preview.py --crop 0.3     # try another, nothing saved

Leave it running and drive the mount from a second terminal with
`python3 -m pantilt keys`. The two don't fight: `keys` only talks to the servos
over I2C, it never opens the camera. Everything else that uses the camera does,
so stop this before `live`, `measure-k` or `survey`.
"""

from __future__ import annotations

import argparse
import io
import sys
import threading
import time

from pantilt.classify import preprocess
from pantilt.config import CameraConfig, load_config
from pantilt.stream import CameraStream, camera_transform

FULL_ARRAY = (4608, 2592)  # IMX708 full frame
THUMB_MIN_INTERVAL = 0.25  # seconds between real grabs, however often asked


class FullFovStream(CameraStream):
    """CameraStream reading the whole sensor, with cached metadata and thumbnail.

    Both the exposure readout and the thumbnail are cached rather than computed
    inside the HTTP handler: picamera2 waits on the capture loop, and a browser
    polling every 150 ms would otherwise stack up grabs until requests time out
    - which is what "status unavailable" in the HUD means.
    """

    def __init__(self, cam: CameraConfig, crop: float, port: int | None = None):
        super().__init__(cam, port=port)
        self.crop = crop
        self.meta: dict = {"camera": "starting"}
        self.status = lambda: self.meta
        self.thumb = self._thumb
        self._jpg: bytes | None = None
        self._jpg_at = 0.0
        self._lock = threading.Lock()

    def _start_camera(self):
        from picamera2 import Picamera2
        from picamera2.encoders import JpegEncoder
        from picamera2.outputs import FileOutput

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_video_configuration(
            main={"size": (self.cam.preview_width, self.cam.preview_height)},
            raw={"size": FULL_ARRAY},
            transform=camera_transform(self.cam),
        ))
        self.picam2.start_recording(
            JpegEncoder(q=self.cam.preview_quality), FileOutput(self.frames))

    def poll(self):
        """Refresh the cached exposure readout. Called from the main loop."""
        try:
            md = self.picam2.capture_metadata()
        except Exception as exc:
            self.meta = {"camera": type(exc).__name__}
            return
        self.meta = {
            "crop": f"{self.crop:.2f}",
            "gain": float(md.get("AnalogueGain", 0.0)),
            "exp_ms": float(md.get("ExposureTime", 0)) / 1000.0,
            "lux": float(md.get("Lux", 0.0)),
        }

    def _thumb(self) -> bytes | None:
        now = time.time()
        if now - self._jpg_at < THUMB_MIN_INTERVAL:
            return self._jpg
        with self._lock:
            if time.time() - self._jpg_at < THUMB_MIN_INTERVAL:
                return self._jpg
            try:
                from PIL import Image

                arr = preprocess(self.picam2.capture_array("main"), crop=self.crop)[0]
                buf = io.BytesIO()
                Image.fromarray(arr).save(buf, format="JPEG", quality=85)
                self._jpg = buf.getvalue()
            except Exception:
                self._jpg = None
            self._jpg_at = time.time()
            return self._jpg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crop", type=float, help="crop to show (default: [model] crop)")
    ap.add_argument("--port", type=int, help="override [camera] preview_port")
    ap.add_argument("--config", help="path to config.toml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    crop = args.crop if args.crop is not None else cfg.model.crop
    if not 0 < crop <= 1:
        print("crop must be in (0, 1]", file=sys.stderr)
        return 1

    stream = FullFovStream(cfg.camera, crop, port=args.port)
    with stream:
        print(f"preview   {stream.url}")
        print(f"red box   what the model sees at crop = {crop:.2f}")
        print("thumb     the actual 224x224 input, bottom left")
        print("HUD       live gain / exposure - aim for gain ~1, exp_ms < 20")
        print("Ctrl-C to stop\n")
        try:
            while True:
                stream.poll()
                time.sleep(0.4)
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
