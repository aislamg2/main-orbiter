"""Minimal PCA9685 PWM driver over Linux I2C (smbus2).

Talks to the chip directly through /dev/i2c-<bus>. No GPIO library is used,
so it works on the Raspberry Pi 5 and never touches the pins the AI HAT+
relies on (only I2C1 on header pins 3 and 5).
"""

from __future__ import annotations

import time

# Registers
MODE1 = 0x00
MODE2 = 0x01
LED0_ON_L = 0x06
PRESCALE = 0xFE

# MODE1 bits
RESTART = 0x80
AI = 0x20  # register auto-increment
SLEEP = 0x10

# MODE2 bits
OUTDRV = 0x04  # totem-pole outputs (what servo boards expect)

FULL_OFF = 0x10  # bit 4 of LEDn_OFF_H

NUM_CHANNELS = 16


class PCA9685:
    """16-channel, 12-bit PWM controller."""

    def __init__(
        self,
        bus: int = 1,
        address: int = 0x40,
        oscillator_hz: float = 25_000_000,
        smbus=None,
    ):
        if smbus is None:
            from smbus2 import SMBus  # imported lazily so tests can run anywhere

            smbus = SMBus(bus)
        self._bus = smbus
        self.address = address
        self.oscillator_hz = oscillator_hz
        self.frequency_hz: float | None = None

        self._write(MODE2, OUTDRV)
        self._write(MODE1, AI)  # wake up, auto-increment on, all-call off
        time.sleep(0.001)

    # -- low level -----------------------------------------------------------
    def _write(self, reg: int, value: int) -> None:
        self._bus.write_byte_data(self.address, reg, value & 0xFF)

    def _read(self, reg: int) -> int:
        return self._bus.read_byte_data(self.address, reg)

    @staticmethod
    def _check_channel(channel: int) -> None:
        if not 0 <= channel < NUM_CHANNELS:
            raise ValueError(f"channel must be 0-{NUM_CHANNELS - 1}, got {channel}")

    # -- public API ----------------------------------------------------------
    def set_frequency(self, hz: float) -> float:
        """Set the PWM frequency for all channels. Returns the actual frequency."""
        prescale = round(self.oscillator_hz / (4096 * hz)) - 1
        if not 3 <= prescale <= 255:
            raise ValueError(f"frequency {hz} Hz is out of range for the PCA9685")

        old = self._read(MODE1) & ~RESTART & 0xFF
        self._write(MODE1, old | SLEEP)  # prescale can only change while asleep
        self._write(PRESCALE, prescale)
        self._write(MODE1, old & ~SLEEP)
        time.sleep(0.001)  # oscillator needs 500 us to stabilise
        self._write(MODE1, (old & ~SLEEP) | RESTART | AI)

        self.frequency_hz = self.oscillator_hz / (4096 * (prescale + 1))
        return self.frequency_hz

    def pulse_to_ticks(self, pulse_us: float) -> int:
        if self.frequency_hz is None:
            raise RuntimeError("call set_frequency() before setting pulses")
        ticks = round(pulse_us * self.frequency_hz * 4096 / 1_000_000)
        return max(0, min(4095, ticks))

    def set_pulse_us(self, channel: int, pulse_us: float) -> int:
        """Output a pulse of `pulse_us` microseconds on `channel`. Returns ticks."""
        self._check_channel(channel)
        ticks = self.pulse_to_ticks(pulse_us)
        self._bus.write_i2c_block_data(
            self.address, LED0_ON_L + 4 * channel, [0, 0, ticks & 0xFF, ticks >> 8]
        )
        return ticks

    def channel_off(self, channel: int) -> None:
        """Stop pulses on a channel so the servo goes limp."""
        self._check_channel(channel)
        self._bus.write_i2c_block_data(
            self.address, LED0_ON_L + 4 * channel, [0, 0, 0, FULL_OFF]
        )

    def ping(self) -> int:
        """Read MODE1; raises OSError if nothing answers at this address."""
        return self._read(MODE1)

    def close(self) -> None:
        self._bus.close()
