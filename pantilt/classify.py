"""Real-time Mars landmark classification on the Hailo-8 (AI HAT+).

The model comes from the mars-landmark-classification project
(hailo-inference branch): a MobileNetV2 classifier compiled to a HEF with the
MobileNetV2 normalization built in, so it takes raw uint8 RGB, 224 x 224.

    from pantilt.classify import HailoClassifier, preprocess
    with HailoClassifier("models/mars_landmark.hef") as clf:
        probs = clf.classify(preprocess(frame))
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

CLASS_NAMES = [
    "other",
    "crater",
    "dark dune",
    "slope streak",
    "bright dune",
    "impact ejecta",
    "swiss cheese",
    "spider",
]

INPUT_SIZE = 224


def preprocess(frame: np.ndarray, size: int = INPUT_SIZE, crop: float = 1.0) -> np.ndarray:
    """Camera frame (H x W x 3 or 4, uint8) -> model input (1 x size x size x 3, uint8).

    Matches how the training tiles look: a square tile, grayscale (the HiRISE
    images are single-channel, copied into R, G and B), nearest-neighbour resize.

    crop: fraction of the frame's short side to use, centered (1.0 = largest
    centered square). Smaller values zoom in on the crosshair.
    """
    from PIL import Image

    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError(f"expected an H x W x 3/4 image, got shape {frame.shape}")
    if not 0 < crop <= 1:
        raise ValueError("crop must be in (0, 1]")

    h, w = frame.shape[:2]
    side = max(1, int(min(h, w) * crop))
    top, left = (h - side) // 2, (w - side) // 2
    tile = frame[top:top + side, left:left + side, :3]

    # Grayscale as a plain channel mean: works the same whether picamera2 hands
    # us RGB or BGR order.
    gray = tile.astype(np.float32).mean(axis=2).round().astype(np.uint8)
    gray = np.asarray(Image.fromarray(gray).resize((size, size), Image.NEAREST))

    # np.array (not asarray): HailoRT needs a writeable, contiguous buffer.
    rgb = np.array(np.repeat(gray[:, :, None], 3, axis=2), dtype=np.uint8, order="C")
    return rgb[None]


class HailoClassifier:
    """Keeps the Hailo pipeline open so each frame costs one inference call."""

    def __init__(self, hef_path: str):
        self.hef_path = hef_path
        self._stack = None
        self._pipeline = None

    def open(self) -> "HailoClassifier":
        from contextlib import ExitStack

        from hailo_platform import (
            HEF, ConfigureParams, FormatType, HailoStreamInterface, InferVStreams,
            InputVStreamParams, OutputVStreamParams, VDevice,
        )

        hef = HEF(self.hef_path)
        self.input_name = hef.get_input_vstream_infos()[0].name
        self.output_name = hef.get_output_vstream_infos()[0].name

        stack = ExitStack()
        try:
            target = stack.enter_context(VDevice())
            params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            network_group = target.configure(hef, params)[0]
            in_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
            out_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            self._pipeline = stack.enter_context(InferVStreams(network_group, in_params, out_params))
            stack.enter_context(network_group.activate(network_group.create_params()))
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        return self

    def classify(self, x: np.ndarray) -> np.ndarray:
        """x from preprocess() -> class probabilities, shape (len(CLASS_NAMES),)."""
        if self._pipeline is None:
            raise RuntimeError("classifier is closed - call open() before classify()")
        out = self._pipeline.infer({self.input_name: x})
        return np.asarray(out[self.output_name], dtype=np.float32).reshape(-1)

    def close(self) -> None:
        if self._stack:
            self._stack.close()
            self._stack = None
            self._pipeline = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


@dataclass
class Prediction:
    label: str = "-"
    index: int = -1
    confidence: float = 0.0           # 0-1, after smoothing
    probs: np.ndarray | None = None
    infer_ms: float = 0.0             # Hailo call only
    fps: float = 0.0                  # whole loop: grab + preprocess + infer
    frames: int = 0
    top3: list = field(default_factory=list)


class LiveClassifier:
    """Background thread: grab the newest camera frame, classify it, repeat.

    grab:     callable returning an H x W x 3/4 uint8 frame (e.g. picam2.capture_array)
    classify: callable taking preprocess() output and returning probabilities
    smooth:   average probabilities over this many frames to stop the label flickering
    """

    def __init__(self, grab, classify, smooth: int = 5, crop: float = 1.0,
                 min_confidence: float = 0.0):
        self.grab = grab
        self.classify = classify
        self.crop = crop
        self.min_confidence = min_confidence
        self._history: deque = deque(maxlen=max(1, smooth))
        self._lock = threading.Lock()
        self._latest = Prediction()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    @property
    def latest(self) -> Prediction:
        with self._lock:
            return self._latest

    def step(self) -> Prediction:
        """One grab -> preprocess -> infer cycle. Also usable without the thread."""
        t0 = time.perf_counter()
        x = preprocess(self.grab(), crop=self.crop)
        t1 = time.perf_counter()
        probs = np.asarray(self.classify(x), dtype=np.float32).reshape(-1)
        t2 = time.perf_counter()

        self._history.append(probs)
        avg = np.mean(self._history, axis=0)
        order = np.argsort(avg)[::-1]
        best = int(order[0])
        conf = float(avg[best])
        label = CLASS_NAMES[best] if best < len(CLASS_NAMES) else str(best)
        if conf < self.min_confidence:
            label = "uncertain"

        prev = self.latest
        loop_s = time.perf_counter() - t0
        fps = 1.0 / loop_s if loop_s > 0 else 0.0
        if prev.frames:
            fps = 0.8 * prev.fps + 0.2 * fps  # smooth the readout
        pred = Prediction(
            label=label, index=best, confidence=conf, probs=avg,
            infer_ms=(t2 - t1) * 1000, fps=fps, frames=prev.frames + 1,
            top3=[(CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i), float(avg[i]))
                  for i in order[:3]],
        )
        with self._lock:
            self._latest = pred
        return pred

    def _run(self):
        try:
            while not self._stop.is_set():
                self.step()
        except BaseException as e:  # surfaced to the caller via .error
            self.error = e

    def start(self) -> "LiveClassifier":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def status(self) -> dict:
        """Fields for the preview page's on-screen display."""
        p = self.latest
        d = {"class": p.label, "conf%": p.confidence * 100, "fps": p.fps, "hailo": f"{p.infer_ms:.1f} ms"}
        for i, (name, prob) in enumerate(p.top3[1:], start=2):
            d[f"#{i}"] = f"{name} {prob * 100:.0f}%"
        return d
