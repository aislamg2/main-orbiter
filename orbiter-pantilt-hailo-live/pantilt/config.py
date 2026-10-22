"""Configuration: defaults match the ArduCAM pan-tilt kit on a PCA9685 at 0x40."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass
class DriverConfig:
    i2c_bus: int = 1
    address: int = 0x40
    frequency_hz: float = 50
    oscillator_hz: float = 25_000_000


@dataclass
class ServoConfig:
    channel: int
    min_angle: float = 0
    max_angle: float = 180
    center: float = 90
    min_pulse_us: float = 500   # pulse at 0 degrees
    max_pulse_us: float = 2500  # pulse at 180 degrees
    invert: bool = False        # flip direction (e.g. servo mounted upside down)

    def __post_init__(self):
        if not 0 <= self.min_angle < self.max_angle <= 180:
            raise ValueError("need 0 <= min_angle < max_angle <= 180")
        if not self.min_angle <= self.center <= self.max_angle:
            raise ValueError("center must be between min_angle and max_angle")
        if not self.min_pulse_us < self.max_pulse_us:
            raise ValueError("min_pulse_us must be less than max_pulse_us")


@dataclass
class MotionConfig:
    speed_dps: float = 60   # degrees per second for smooth moves
    step_deg: float = 1     # interpolation step for smooth moves
    settle_s: float = 0.3   # pause after a move before taking a picture


@dataclass
class CameraConfig:
    rotate_180: bool = False     # camera mounted upside down
    preview_width: int = 1280
    preview_height: int = 720
    preview_quality: int = 70    # JPEG quality for the live preview (1-100)
    preview_port: int = 8000


@dataclass
class ModelConfig:
    hef_path: str = "models/mars_landmark.hef"  # compiled Hailo model (mars-landmark-classification)
    crop: float = 1.0          # fraction of the frame's short side to classify (centered)
    smooth_frames: int = 5     # average predictions over N frames to steady the label
    min_confidence: float = 0.0  # below this, show "uncertain" (0-1)
    log_every_s: float = 1.0   # live mode: seconds between rows in the predictions log

    def __post_init__(self):
        if not 0 < self.crop <= 1:
            raise ValueError("crop must be in (0, 1]")
        if self.smooth_frames < 1:
            raise ValueError("smooth_frames must be >= 1")


@dataclass
class Config:
    driver: DriverConfig = field(default_factory=DriverConfig)
    pan: ServoConfig = field(default_factory=lambda: ServoConfig(channel=1))
    tilt: ServoConfig = field(
        default_factory=lambda: ServoConfig(channel=0, min_angle=15, max_angle=145)
    )
    motion: MotionConfig = field(default_factory=MotionConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


_SECTIONS = {
    "driver": DriverConfig,
    "pan": ServoConfig,
    "tilt": ServoConfig,
    "motion": MotionConfig,
    "camera": CameraConfig,
    "model": ModelConfig,
}


def load_config(path: str | Path | None = None) -> Config:
    """Load a TOML config. With no path, uses ./config.toml if present, else defaults."""
    if path is None:
        path = Path("config.toml")
        if not path.exists():
            return Config()
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)

    defaults = Config()
    kwargs = {}
    for name, cls in _SECTIONS.items():
        section = data.pop(name, {})
        allowed = {f.name for f in fields(cls)}
        unknown = set(section) - allowed
        if unknown:
            raise ValueError(f"{path}: unknown key(s) in [{name}]: {', '.join(sorted(unknown))}")
        base = vars(getattr(defaults, name)).copy()
        base.update(section)
        kwargs[name] = cls(**base)
    if data:
        raise ValueError(f"{path}: unknown section(s): {', '.join(sorted(data))}")
    return Config(**kwargs)
