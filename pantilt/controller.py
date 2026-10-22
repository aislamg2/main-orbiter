"""Servo and pan-tilt controller."""

from __future__ import annotations

import time

from .config import Config, ServoConfig
from .pca9685 import PCA9685


class Servo:
    def __init__(self, driver: PCA9685, cfg: ServoConfig, name: str):
        self.driver = driver
        self.cfg = cfg
        self.name = name
        self.angle: float | None = None  # last commanded angle (unknown at start)

    def clamp(self, angle: float) -> float:
        return max(self.cfg.min_angle, min(self.cfg.max_angle, angle))

    def pulse_for(self, angle: float) -> float:
        a = self.clamp(angle)
        if self.cfg.invert:
            a = 180 - a
        span = self.cfg.max_pulse_us - self.cfg.min_pulse_us
        return self.cfg.min_pulse_us + span * a / 180

    def set(self, angle: float) -> float:
        a = self.clamp(angle)
        self.driver.set_pulse_us(self.cfg.channel, self.pulse_for(a))
        self.angle = a
        return a

    def release(self) -> None:
        self.driver.channel_off(self.cfg.channel)


class PanTilt:
    """Two-axis camera mount.

    The servos have no position feedback, so the first command after start-up
    jumps straight to the target. Later moves are smooth by default.
    """

    def __init__(self, config: Config | None = None, driver: PCA9685 | None = None):
        self.config = config or Config()
        d = self.config.driver
        self.driver = driver or PCA9685(d.i2c_bus, d.address, d.oscillator_hz)
        self.driver.set_frequency(d.frequency_hz)
        self.pan = Servo(self.driver, self.config.pan, "pan")
        self.tilt = Servo(self.driver, self.config.tilt, "tilt")

    # -- state ---------------------------------------------------------------
    @property
    def position(self) -> tuple[float | None, float | None]:
        return self.pan.angle, self.tilt.angle

    # -- motion --------------------------------------------------------------
    def jump(self, pan: float | None = None, tilt: float | None = None):
        """Go straight to the target with no interpolation."""
        if pan is not None:
            self.pan.set(pan)
        if tilt is not None:
            self.tilt.set(tilt)
        return self.position

    def move_to(
        self,
        pan: float | None = None,
        tilt: float | None = None,
        speed_dps: float | None = None,
        settle: bool = False,
        ease: bool = False,
    ):
        """Move smoothly to (pan, tilt). Either axis can be left as None.

        ease=True speeds up and slows down gently instead of stopping dead, so
        the bracket rings less and settles sooner.
        """
        m = self.config.motion
        speed = speed_dps or m.speed_dps

        # Jump any axis whose current position is unknown.
        if pan is not None and self.pan.angle is None:
            self.pan.set(pan)
            pan = None
        if tilt is not None and self.tilt.angle is None:
            self.tilt.set(tilt)
            tilt = None

        p0, t0 = self.position
        p1 = self.pan.clamp(pan) if pan is not None else p0
        t1 = self.tilt.clamp(tilt) if tilt is not None else t0
        dp = (p1 - p0) if p0 is not None else 0
        dt = (t1 - t0) if t0 is not None else 0

        n = int(max(abs(dp), abs(dt)) / m.step_deg)
        delay = m.step_deg / speed
        for i in range(1, n + 1):
            f = i / n
            if ease:
                f = f * f * (3 - 2 * f)  # smoothstep
            if dp:
                self.pan.set(p0 + dp * f)
            if dt:
                self.tilt.set(t0 + dt * f)
            time.sleep(delay)
        # land exactly on target
        if pan is not None:
            self.pan.set(p1)
        if tilt is not None:
            self.tilt.set(t1)

        if settle:
            time.sleep(m.settle_s)
        return self.position

    def nudge(self, dpan: float = 0, dtilt: float = 0):
        """Small relative move (used by keyboard control)."""
        p, t = self.position
        return self.jump(
            pan=(p if p is not None else self.config.pan.center) + dpan,
            tilt=(t if t is not None else self.config.tilt.center) + dtilt,
        )

    def center(self, smooth: bool = True):
        target = dict(pan=self.config.pan.center, tilt=self.config.tilt.center)
        return self.move_to(**target) if smooth else self.jump(**target)

    def release(self) -> None:
        """Stop holding position (servos go limp and stop drawing current)."""
        self.pan.release()
        self.tilt.release()

    def close(self, release: bool = False) -> None:
        if release:
            self.release()
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
