"""Unit tests that run without hardware, using a fake I2C bus."""

import pytest

import pantilt.controller as controller
from pantilt import Config, PanTilt, PCA9685, load_config, raster
from pantilt.config import ServoConfig


class FakeBus:
    def __init__(self):
        self.regs = {0x00: 0x11}  # MODE1 power-on value (SLEEP | ALLCALL)
        self.blocks = []

    def write_byte_data(self, addr, reg, value):
        self.regs[reg] = value

    def read_byte_data(self, addr, reg):
        return self.regs.get(reg, 0)

    def write_i2c_block_data(self, addr, reg, data):
        self.blocks.append((reg, list(data)))

    def close(self):
        pass


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(controller.time, "sleep", lambda s: None)


def make(cfg=None):
    bus = FakeBus()
    return PanTilt(cfg or Config(), driver=PCA9685(smbus=bus)), bus


def last_ticks(bus, channel):
    reg = 0x06 + 4 * channel
    for r, data in reversed(bus.blocks):
        if r == reg:
            return data[2] | (data[3] << 8)
    return None


def test_frequency_prescale():
    pt, bus = make()
    assert bus.regs[0xFE] == 121          # 25 MHz / (4096 * 50) - 1
    assert pt.driver.frequency_hz == pytest.approx(50.03, abs=0.01)
    assert bus.regs[0x00] & 0x10 == 0     # awake after set_frequency


def test_center_pulse_is_1500us():
    pt, bus = make()
    pt.center(smooth=False)
    assert pt.position == (90, 90)
    ticks = last_ticks(bus, 1)
    assert ticks == 307                   # 1500 us at ~50 Hz


def test_limits_are_clamped():
    pt, _ = make()
    assert pt.jump(pan=999, tilt=-50) == (180, 15)


def test_invert_mirrors_pulse():
    cfg = Config()
    cfg.pan = ServoConfig(channel=1, invert=True)
    pt, _ = make(cfg)
    assert pt.pan.pulse_for(0) == 2500
    assert pt.pan.pulse_for(180) == 500


def test_smooth_move_reaches_target():
    pt, bus = make()
    pt.center(smooth=False)
    n_before = len(bus.blocks)
    assert pt.move_to(pan=120, tilt=60) == (120, 60)
    assert len(bus.blocks) - n_before > 10   # interpolated, not a single jump


def test_release_sends_full_off():
    pt, bus = make()
    pt.release()
    assert (0x06 + 4 * 0, [0, 0, 0, 0x10]) in bus.blocks
    assert (0x06 + 4 * 1, [0, 0, 0, 0x10]) in bus.blocks


def test_raster_is_serpentine():
    poses = list(raster((0, 100), (20, 40), cols=3, rows=2))
    assert poses == [(0, 20), (50, 20), (100, 20), (100, 40), (50, 40), (0, 40)]


def test_load_config(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text("[driver]\naddress = 0x41\n[tilt]\ninvert = true\n")
    cfg = load_config(f)
    assert cfg.driver.address == 0x41
    assert cfg.tilt.invert and cfg.tilt.min_angle == 15   # untouched defaults kept


def test_load_config_rejects_typos(tmp_path):
    f = tmp_path / "c.toml"
    f.write_text("[pan]\nmax_angel = 170\n")
    with pytest.raises(ValueError, match="max_angel"):
        load_config(f)
