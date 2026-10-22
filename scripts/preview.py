"""Live preview for aiming the mount, with the same field of view as a still.

`pantilt live` also shows a preview, but it loads the Hailo model and classifies
every frame, which you don't need while you're just pointing the camera. This
doesn't touch the model.

The raw stream is pinned to the full sensor array on purpose. A 720p video mode
left to itself reads out a *cropped* region of the sensor, so its field of view
is narrower than what `measure-k` and `survey` actually capture - framing on it
would lie to you about what's in the shot. Forcing the full array means the
preview edges are the still's edges.

    python3 scripts/preview.py

Leave it running, open the URL it prints, and drive the mount from a second
terminal with `python3 -m pantilt keys`. The two don't fight: `keys` only talks
to the servos over I2C, it never opens the camera.
"""

from __future__ import annotations

import sys
import time

from pantilt.config import load_config
from pantilt.stream import CameraStream, camera_transform

FULL_ARRAY = (4608, 2592)  # IMX708 full frame


class FullFovStream(CameraStream):
    """CameraStream, but reading out the whole sensor rather than a crop."""

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


def exposure_status(stream: CameraStream):
    """What auto-exposure is choosing, live - the lighting readout.

    `survey` freezes these at the start of a run, so whatever they read while
    you're aiming is roughly what every tile will be captured at. Gain near 1
    and exposure under 20 ms is the target; gain in the double digits means the
    board is underlit and the classifier will be looking at sensor noise.
    """

    def status():
        try:
            md = stream.picam2.capture_metadata()
        except Exception:
            return {"camera": "starting"}
        return {
            "gain": float(md.get("AnalogueGain", 0.0)),
            "exp_ms": float(md.get("ExposureTime", 0)) / 1000.0,
            "lux": float(md.get("Lux", 0.0)),
        }

    return status


def main() -> int:
    cfg = load_config()
    stream = FullFovStream(cfg.camera)
    stream.status = exposure_status(stream)
    with stream:
        print(f"preview  {stream.url}")
        print("crosshair = frame centre, which is the point survey classifies")
        print("HUD shows live gain / exposure - aim for gain ~1, exp_ms < 20")
        print("Ctrl-C to stop\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
