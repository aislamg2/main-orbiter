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
class BoardConfig:
    """The printed map, and where the camera sits relative to it.

    These are measurements, not preferences. `pantilt measure-k` produces
    mm_per_px; the centre angles come from jogging to the middle of the board
    with `pantilt keys` and reading them off.
    """

    distance_mm: float = 1500.0   # camera to board surface
    mm_per_px: float = 0.0        # 0 = not measured yet; run `pantilt measure-k`
    width_mm: float = 1219.0      # the printed map, not the whole sheet
    height_mm: float = 914.0
    overlap: float = 0.20         # fraction shared with each neighbouring tile
    pan_center_deg: float = 90.0  # angles that look at the middle of the board
    tilt_center_deg: float = 90.0
    pan_sign: float = 1.0         # +1 if a larger pan angle moves the view right
    tilt_sign: float = 1.0        # +1 if a larger tilt angle moves the view up

    def __post_init__(self):
        if self.distance_mm <= 0:
            raise ValueError("distance_mm must be positive")
        if self.mm_per_px < 0:
            raise ValueError("mm_per_px must be >= 0 (0 means not yet measured)")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        if self.pan_sign not in (1.0, -1.0) or self.tilt_sign not in (1.0, -1.0):
            raise ValueError("pan_sign and tilt_sign must be 1.0 or -1.0")

    @property
    def measured(self) -> bool:
        return self.mm_per_px > 0


@dataclass
class SweepConfig:
    """`pantilt board-scan`: stop-and-go scan inside corners taught with `pantilt teach`."""

    corners_file: str = "data/board_corners.json"
    hfov_deg: float = 67.0        # horizontal field of view (IMX708 75 deg diagonal lens)
    overlap: float = 0.25         # fraction of the crop square shared with each neighbour
    target: str = "crater"        # class that triggers a photo
    threshold: float = 0.6        # probability of `target` that counts as a detection
    cooldown_tiles: float = 1.2   # no new photo within this many crop widths of a saved one
    cooldown_s: float = 3.0       # and at least this long between any two photos
    max_photos: int = 100         # stop saving after this many per run (0 = no limit)
    speed_dps: float = 30         # slower than [motion] so the mount shakes less
    settle_s: float = 0.4         # always wait this long after a move
    settle_max_s: float = 2.0     # then wait for the image to stop changing, up to this long
    still_diff: float = 2.0       # mean grey-level change between frames that counts as still (0 = off)

    def __post_init__(self):
        if not 0 < self.hfov_deg < 180:
            raise ValueError("hfov_deg must be in (0, 180)")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if self.speed_dps <= 0:
            raise ValueError("speed_dps must be positive")
        if self.settle_max_s < self.settle_s:
            raise ValueError("settle_max_s must be >= settle_s")


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
    board: BoardConfig = field(default_factory=BoardConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)


_SECTIONS = {
    "driver": DriverConfig,
    "pan": ServoConfig,
    "tilt": ServoConfig,
    "motion": MotionConfig,
    "camera": CameraConfig,
    "model": ModelConfig,
    "board": BoardConfig,
    "sweep": SweepConfig,
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
